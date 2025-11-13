import math
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

# 仅使用 Qwen2.5
from transformers.models.qwen2_5.modeling_qwen2_5 import (
    Qwen2_5Attention,
    Qwen2_5DecoderLayer,
    Qwen2_5Model,
    apply_rotary_pos_emb,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)

from peft import get_peft_model, LoraConfig, TaskType


# ====================== 基础数值/变量 Embedding ======================

class ValueEmbedding_Qwen(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(c_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class VariableEmbedding_Qwen(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.var_emb = nn.Embedding(c_in, d_model)
        self.c_in = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        var_idx = torch.arange(D, device=x.device).view(1, 1, D).expand(B, L, D)
        return self.var_emb(var_idx)  # (B,L,D,H)


# ====================== CT-RoPE Attention (Qwen2.5) ======================

class CTRoPEQwen2Attention(Qwen2_5Attention):
    def __init__(self, config, layer_idx: int):
        # 父类初始化
        super().__init__(config, layer_idx)

        # 显式补齐关键属性，防止未来版本签名改变
        if not hasattr(self, "num_heads"):
            self.num_heads = getattr(config, "num_attention_heads")
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = getattr(
                config, "num_key_value_heads", self.num_heads
            )
        if not hasattr(self, "hidden_size"):
            self.hidden_size = config.hidden_size
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_heads

        # rotary embedding 若未创建则补齐
        if not hasattr(self, "rotary_emb"):
            self.rotary_emb = Qwen2RotaryEmbedding(
                self.head_dim, base=10000, interleaved=False
            )

        # 投影层补齐（一般已存在）
        need_proj = any(
            not hasattr(self, name)
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        if need_proj:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        if not hasattr(self, "attn_dropout"):
            self.attn_dropout = nn.Dropout(getattr(config, "attn_dropout", 0.0))

        self.layer_idx = layer_idx

    def _repeat_kv(self, kv: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return kv
        bsz, n_kv, seqlen, hd = kv.shape
        return kv.unsqueeze(2).expand(bsz, n_kv, n_rep, seqlen, hd).reshape(
            bsz, n_kv * n_rep, seqlen, hd
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        timestamps: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Rotary embedding with continuous time
        if timestamps is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            ts = timestamps.unsqueeze(1).unsqueeze(-1).to(key_states.dtype)  # (B,1,L,1)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, ts)
        elif position_ids is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Append cached K/V
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # GQA repeat
        if self.num_key_value_heads != self.num_heads:
            n_rep = self.num_heads // self.num_key_value_heads
            key_states = self._repeat_kv(key_states, n_rep)
            value_states = self._repeat_kv(value_states, n_rep)

        past_key_value = (key_states, value_states) if use_cache else None

        # Attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Add mask
        if attention_mask is not None:
            if cache_position is not None:
                mask_q_len = attention_mask.shape[-2]
                if mask_q_len != q_len:
                    start_row = mask_q_len - q_len
                    attention_mask = attention_mask[:, :, start_row:, :]
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)  # (B, H, q_len, hd)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"Unexpected attn_output shape {attn_output.shape}")

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ====================== 替换 Decoder Layer ======================

class Qwen2DecoderLayerWithCTRoPE(Qwen2_5DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = CTRoPEQwen2Attention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        timestamps: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            timestamps=timestamps,
            cache_position=cache_position,
        )

        hidden_states = residual + attn_outputs

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ====================== 基础模型包装 ======================

class PLM4TS_Base(Qwen2_5Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayerWithCTRoPE(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = None  # 禁用 token embedding

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        timestamps: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            raise ValueError("Use inputs_embeds instead of input_ids for PLM4TS_Base.")
        if inputs_embeds is None:
            raise ValueError("inputs_embeds is required.")

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                timestamps=timestamps,
                cache_position=cache_position,
            )
            hidden_states = layer_out[0]

            if use_cache:
                next_decoder_cache += (layer_out[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_out[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ====================== 注入 CT-RoPE + LoRA ======================

def inject_CTRoPE_to_model(model: PreTrainedModel):
    for name, module in model.named_children():
        if isinstance(module, Qwen2_5Attention):
            config = module.config
            layer_idx = module.layer_idx
            new_attn = CTRoPEQwen2Attention(config, layer_idx)

            # 权重复制
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                old_layer = getattr(module, proj, None)
                new_layer = getattr(new_attn, proj, None)
                if old_layer is not None and new_layer is not None:
                    new_layer.weight.data.copy_(old_layer.weight.data)
                    if old_layer.bias is not None and new_layer.bias is not None:
                        new_layer.bias.data.copy_(old_layer.bias.data)

            setattr(model, name, new_attn)
        else:
            inject_CTRoPE_to_model(module)
    return model


def configure_lora(args, model: PLM4TS_Base):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(model, lora_config)
    for name, param in lora_model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    return lora_model


# ====================== 主任务模型 ======================

class istsplm_forecast(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.use_lora = args.use_lora
        self.plm_name = args.plm_path
        self.num_types = args.num_types
        self.n_plm_layers = 2

        # 配置
        self.config = AutoConfig.from_pretrained(self.plm_name, trust_remote_code=True)
        self.config.use_cache = True

        # Embeddings
        self.value_embedding = ValueEmbedding_Qwen(1, self.d_model)
        self.var_embedding = nn.Embedding(self.num_types, self.d_model)

        # 两个 PLM 实例
        gpts = []
        for _ in range(self.n_plm_layers):
            model = PLM4TS_Base.from_pretrained(
                self.plm_name,
                config=self.config,
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
            )
            model = inject_CTRoPE_to_model(model)
            if self.use_lora:
                model = configure_lora(args, model)
            gpts.append(model)
        self.gpts = nn.ModuleList(gpts)

        # 投影与预测头
        self.ln_proj = nn.LayerNorm(self.d_model)
        self.forecasting_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LeakyReLU(),
            nn.Linear(self.d_model // 2, self.num_types),
        )

        # Prompt
        self.patch_len = args.patch_len
        self.prompt_len = args.prompt_len
        self.prompt_embed = nn.Parameter(torch.randn(1, self.prompt_len, self.d_model))

    def forecasting(self, batch_dict, n_vars_to_predict=None):
        observed_data = batch_dict["observed_data"].to(self.device)      # (B, L_obs, D)
        observed_mask = batch_dict["observed_mask"].to(self.device)      # (B, L_obs, D)
        observed_tp_raw = batch_dict["observed_tp"].to(self.device)      # (B, L_obs) or (B, 1, L_obs)
        tp_to_predict = batch_dict["tp_to_predict"].to(self.device)      # (B, L_pred)

        B, L_obs, D = observed_data.shape
        L_pred = tp_to_predict.shape[1]

        # Value embedding
        x_flat = observed_data.permute(0, 2, 1).reshape(B * D, L_obs, 1)
        value_embed = self.value_embedding(x_flat)  # (B*D, L_val, d_model)
        L_val = value_embed.size(1)

        # 时间戳对齐
        observed_tp = observed_tp_raw.reshape(B, -1)
        L_time = observed_tp.size(1)
        if L_time != L_val:
            if L_time >= L_val:
                observed_tp_aligned = observed_tp[:, -L_val:]
            else:
                pad = torch.zeros(B, L_val - L_time, device=self.device, dtype=observed_tp.dtype)
                observed_tp_aligned = torch.cat([observed_tp, pad], dim=1)
        else:
            observed_tp_aligned = observed_tp

        tt = observed_tp_aligned.unsqueeze(1).expand(B, D, L_val).reshape(B * D, L_val)

        # Prompt + embeddings
        prompt_embed = self.prompt_embed.expand(B * D, -1, -1)
        inputs_embeds = torch.cat([prompt_embed, value_embed], dim=1)  # (B*D, L_prompt+L_val, d_model)
        tt_prompt = torch.zeros(B * D, self.prompt_len, device=self.device)
        tt_with_prompt = torch.cat([tt_prompt, tt], dim=1)  # (B*D, L_prompt+L_val)

        # 时间注意力 mask
        time_attn_mask_data_flat = observed_mask.permute(0, 2, 1).reshape(B * D, L_obs)
        if L_obs != L_val:
            if L_obs >= L_val:
                time_attn_mask_data_flat = time_attn_mask_data_flat[:, -L_val:]
            else:
                pad_m = torch.zeros(B * D, L_val - L_obs, device=self.device, dtype=time_attn_mask_data_flat.dtype)
                time_attn_mask_data_flat = torch.cat([time_attn_mask_data_flat, pad_m], dim=1)

        time_attn_mask_prompt = torch.ones(B * D, self.prompt_len, device=self.device)
        time_attn_mask = torch.cat([time_attn_mask_prompt, time_attn_mask_data_flat], dim=1)
        time_attn_mask = (1.0 - time_attn_mask) * torch.finfo(torch.float32).min
        time_attn_mask = time_attn_mask.unsqueeze(1).unsqueeze(1)  # (B*D,1,1,L_total)

        # 时间 PLM
        outputs_time = self.gpts[0](
            inputs_embeds=inputs_embeds,
            attention_mask=time_attn_mask,
            timestamps=tt_with_prompt,
        ).last_hidden_state  # (B*D, L_prompt+L_val, d_model)

        # 池化（去掉 prompt）
        observed_mask_pooled = observed_mask.permute(0, 2, 1).reshape(B * D, L_obs, 1)
        if L_obs != L_val:
            if L_obs >= L_val:
                observed_mask_pooled = observed_mask_pooled[:, -L_val:, :]
            else:
                pad_mm = torch.zeros(B * D, L_val - L_obs, 1, device=self.device, dtype=observed_mask_pooled.dtype)
                observed_mask_pooled = torch.cat([observed_mask_pooled, pad_mm], dim=1)
        outputs_val = outputs_time[:, self.prompt_len:]  # (B*D, L_val, d_model)
        denom = observed_mask_pooled.sum(dim=1, keepdim=True) + 1e-8
        pooled = (outputs_val * observed_mask_pooled).sum(dim=1) / denom.squeeze(1)  # (B*D, d_model)
        pooled = self.ln_proj(pooled).view(B, D, -1)  # (B,D,d_model)

        # 变量 PLM输入
        var_emb = self.var_embedding.weight.unsqueeze(0).expand(B, D, -1)
        var_inputs = pooled + var_emb  # (B,D,d_model)
        var_position_ids = torch.arange(D, device=self.device).unsqueeze(0).expand(B, D)
        var_attn_mask = torch.zeros(B, D, D, device=self.device).unsqueeze(1)  # (B,1,D,D)

        outputs_var = self.gpts[1](
            inputs_embeds=var_inputs,
            attention_mask=var_attn_mask,
            position_ids=var_position_ids,
        ).last_hidden_state  # (B,D,d_model)

        # 预测
        forecast_vars = self.forecasting_head(outputs_var)  # (B,D,D)
        regression_value = forecast_vars[:, :, 0]  # (B,D)
        forecast = regression_value.unsqueeze(1).expand(-1, L_pred, -1)  # (B,L_pred,D)

        if n_vars_to_predict is not None:
            forecast = forecast[:, :, :n_vars_to_predict]
        return forecast

    def forward(self, batch_dict, n_vars_to_predict=None):
        return self.forecasting(batch_dict, n_vars_to_predict)
