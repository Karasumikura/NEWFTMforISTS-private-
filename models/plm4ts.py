import math
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import get_peft_model, LoraConfig, TaskType

# ================= Embeddings =================
class ValueEmbedding_Qwen(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(c_in, d_model)
    def forward(self, x: torch.Tensor):
        return self.projection(x)

class VariableEmbedding_Qwen(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.var_emb = nn.Embedding(c_in, d_model)
    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        idx = torch.arange(D, device=x.device).view(1,1,D).expand(B,L,D)
        return self.var_emb(idx)  # (B,L,D,H)

# ============== CT-RoPE 注入 ==============
def inject_ctr_rope(model, attn_attr_path=("model","layers"), verbose=True):
    """
    反射式遍历所有 decoder layer 的 self_attn，替换 forward 以支持连续时间戳 timestamps。
    不修改类、只包装方法，适配 trust_remote_code 加载的 Qwen2.5。
    """
    # 寻找 decoder 层列表
    current = model
    for attr in attn_attr_path:
        current = getattr(current, attr)
    layers = current  # ModuleList
    n_layers = len(layers)
    if verbose:
        print(f"[inject_ctr_rope] Found {n_layers} layers")

    def make_new_forward(old_attn):
        # 读取必要属性
        q_proj = old_attn.q_proj
        k_proj = old_attn.k_proj
        v_proj = old_attn.v_proj
        o_proj = old_attn.o_proj
        num_heads = getattr(old_attn, "num_heads")
        num_kv_heads = getattr(old_attn, "num_key_value_heads", num_heads)
        head_dim = getattr(old_attn, "head_dim", q_proj.out_features // num_heads)
        rotary_emb = getattr(old_attn, "rotary_emb", None)
        attn_dropout = getattr(old_attn, "attn_dropout", nn.Dropout(0.0))

        def _repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
            if n_rep == 1:
                return kv
            bsz, n_kv, seqlen, hd = kv.shape
            return kv.unsqueeze(2).expand(bsz, n_kv, n_rep, seqlen, hd).reshape(
                bsz, n_kv * n_rep, seqlen, hd
            )

        def new_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            timestamps: Optional[torch.Tensor] = None,   # 新增
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()
            # 投影
            q = q_proj(hidden_states)
            k = k_proj(hidden_states)
            v = v_proj(hidden_states)

            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1,2)      # (B,H,L,hd)
            k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1,2)   # (B,H_kv,L,hd)
            v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1,2)

            kv_seq_len = k.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]

            # Rotary embedding
            if rotary_emb is not None:
                cos, sin = rotary_emb(v, seq_len=kv_seq_len)
                if timestamps is not None:
                    ts = timestamps.unsqueeze(1).unsqueeze(-1).to(k.dtype)  # (B,1,L,1)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, ts)
                elif position_ids is not None:
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

            # 拼接缓存
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)

            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                k = _repeat_kv(k, n_rep)
                v = _repeat_kv(v, n_rep)

            past = (k, v) if use_cache else None

            attn_weights = torch.matmul(q, k.transpose(2,3)) / math.sqrt(head_dim)

            if attention_mask is not None:
                if cache_position is not None:
                    mask_q_len = attention_mask.shape[-2]
                    if mask_q_len != q_len:
                        start_row = mask_q_len - q_len
                        attention_mask = attention_mask[:, :, start_row:, :]
                attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)  # (B,H,L,hd)

            if attn_output.size() != (bsz, num_heads, q_len, head_dim):
                raise ValueError(f"attn_output shape mismatch {attn_output.shape}")

            attn_output = attn_output.transpose(1,2).contiguous().view(bsz, q_len, num_heads*head_dim)
            out = o_proj(attn_output)

            if not output_attentions:
                attn_weights_out = None
            else:
                attn_weights_out = attn_weights

            return out, attn_weights_out, past

        return new_forward

    # 注入
    for i, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            continue
        old_attn = layer.self_attn
        # 包装 forward
        layer.self_attn.forward = make_new_forward(old_attn)
        if verbose:
            print(f"[inject_ctr_rope] Layer {i} attention patched. Type={type(old_attn)}")

    return model


# ============== LoRA 配置 ==============
def configure_lora(args, model):
    target_modules = ["q_proj","k_proj","v_proj","o_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    lora_model = get_peft_model(model, lora_cfg)
    for n,p in lora_model.named_parameters():
        if "lora" not in n:
            p.requires_grad = False
    return lora_model


# ============== 主任务模型 ==============
class istsplm_forecast(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.d_model = args.d_model
        self.num_types = args.num_types
        self.use_lora = args.use_lora
        self.plm_name = args.plm_path
        self.n_plm_layers = 2
        self.prompt_len = args.prompt_len
        self.patch_len = args.patch_len

        self.value_embedding = ValueEmbedding_Qwen(1, self.d_model)
        self.var_embedding = nn.Embedding(self.num_types, self.d_model)
        self.prompt_embed = nn.Parameter(torch.randn(1, self.prompt_len, self.d_model))

        # 加载 Qwen2.5 (远程代码)
        self.config = AutoConfig.from_pretrained(self.plm_name, trust_remote_code=True)
        self.config.use_cache = True

        gpts = []
        for _ in range(self.n_plm_layers):
            base = AutoModelForCausalLM.from_pretrained(
                self.plm_name,
                trust_remote_code=True,
                torch_dtype="auto"
            )
            # 只取其基础 model 部分 (Qwen 结构一般是 .model)
            core = getattr(base, "model", base)

            # 注入连续时间 CT-RoPE
            core = inject_ctr_rope(core, attn_attr_path=("layers",), verbose=False)

            if self.use_lora:
                core = configure_lora(args, core)

            gpts.append(core)

        self.gpts = nn.ModuleList(gpts)
        self.ln_proj = nn.LayerNorm(self.d_model)

        self.forecasting_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LeakyReLU(),
            nn.Linear(self.d_model // 2, self.num_types)
        )

    def forecasting(self, batch_dict, n_vars_to_predict=None):
        observed_data = batch_dict["observed_data"].to(self.device)      # (B,L_obs,D)
        observed_mask = batch_dict["observed_mask"].to(self.device)      # (B,L_obs,D)
        observed_tp_raw = batch_dict["observed_tp"].to(self.device)      # (B,L_obs) or (B,1,L_obs)
        tp_to_predict = batch_dict["tp_to_predict"].to(self.device)      # (B,L_pred)

        B, L_obs, D = observed_data.shape
        L_pred = tp_to_predict.shape[1]

        # Value embedding
        x_flat = observed_data.permute(0,2,1).reshape(B*D, L_obs, 1)
        value_embed = self.value_embedding(x_flat)   # (B*D, L_obs, d_model)
        L_val = value_embed.size(1)

        # 时间戳对齐
        observed_tp = observed_tp_raw.reshape(B, -1)
        L_time = observed_tp.size(1)
        if L_time != L_val:
            if L_time >= L_val:
                tp_aligned = observed_tp[:, -L_val:]
            else:
                pad = torch.zeros(B, L_val-L_time, device=self.device, dtype=observed_tp.dtype)
                tp_aligned = torch.cat([observed_tp, pad], dim=1)
        else:
            tp_aligned = observed_tp

        tt = tp_aligned.unsqueeze(1).expand(B, D, L_val).reshape(B*D, L_val)

        # Prompt + concat
        prompt = self.prompt_embed.expand(B*D, -1, -1)
        inputs_embeds = torch.cat([prompt, value_embed], dim=1)          # (B*D, prompt_len+L_val, d_model)
        tt_prompt = torch.zeros(B*D, self.prompt_len, device=self.device)
        tt_with_prompt = torch.cat([tt_prompt, tt], dim=1)               # (B*D, prompt_len+L_val)

        # 时间 mask
        time_mask_flat = observed_mask.permute(0,2,1).reshape(B*D, L_obs)
        if L_obs != L_val:
            if L_obs >= L_val:
                time_mask_flat = time_mask_flat[:, -L_val:]
            else:
                padm = torch.zeros(B*D, L_val-L_obs, device=self.device, dtype=time_mask_flat.dtype)
                time_mask_flat = torch.cat([time_mask_flat, padm], dim=1)

        time_mask_prompt = torch.ones(B*D, self.prompt_len, device=self.device)
        time_mask = torch.cat([time_mask_prompt, time_mask_flat], dim=1)   # (B*D, total_len)
        additive_mask = (1.0 - time_mask) * torch.finfo(torch.float32).min
        additive_mask = additive_mask.unsqueeze(1).unsqueeze(1)            # (B*D,1,1,total_len)

        # 时间 PLM 前向 (gpts[0])
        out_time = self.gpts[0](
            inputs_embeds=inputs_embeds,
            attention_mask=additive_mask,
            timestamps=tt_with_prompt
        ).last_hidden_state   # (B*D, total_len, d_model)

        # 池化（去掉 prompt）
        mask_pooled = observed_mask.permute(0,2,1).reshape(B*D, L_obs, 1)
        if L_obs != L_val:
            if L_obs >= L_val:
                mask_pooled = mask_pooled[:, -L_val:, :]
            else:
                padm3 = torch.zeros(B*D, L_val-L_obs, 1, device=self.device, dtype=mask_pooled.dtype)
                mask_pooled = torch.cat([mask_pooled, padm3], dim=1)

        out_val = out_time[:, self.prompt_len:]  # (B*D, L_val, d_model)
        denom = mask_pooled.sum(dim=1, keepdim=True) + 1e-8
        pooled = (out_val * mask_pooled).sum(dim=1) / denom.squeeze(1)  # (B*D, d_model)
        pooled = self.ln_proj(pooled).view(B, D, -1)                    # (B,D,d_model)

        # 变量级 PLM (gpts[1])
        var_emb = self.var_embedding.weight.unsqueeze(0).expand(B, D, -1)
        var_inputs = pooled + var_emb
        # 构造一个“全连接” attention mask：这里用 0 (无惩罚)，保持全可见
        var_attn_mask = torch.zeros(B, D, D, device=self.device).unsqueeze(1)  # (B,1,D,D)
        # Qwen 通常需要 position_ids：用简单整数序列
        position_ids = torch.arange(D, device=self.device).unsqueeze(0).expand(B, D)

        out_var = self.gpts[1](
            inputs_embeds=var_inputs,
            attention_mask=var_attn_mask,
            position_ids=position_ids
        ).last_hidden_state   # (B,D,d_model)

        # 预测
        forecast_vars = self.forecasting_head(out_var)  # (B,D,D)
        regression_value = forecast_vars[:, :, 0]       # (B,D)
        forecast = regression_value.unsqueeze(1).expand(-1, L_pred, -1)  # (B,L_pred,D)

        if n_vars_to_predict is not None:
            forecast = forecast[:, :, :n_vars_to_predict]
        return forecast

    def forward(self, batch_dict, n_vars_to_predict=None):
        return self.forecasting(batch_dict, n_vars_to_predict)
