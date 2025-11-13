import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List

# ==================================================================================
# 第 1 部分：新模型 (Qwen / LoRA) 的依赖项
# ==================================================================================
from transformers import AutoConfig, PreTrainedModel

# 兼容 Qwen2.5 与 Qwen2：优先尝试 qwen2_5，失败则退回 qwen2
try:
    from transformers.models.qwen2_5.modeling_qwen2_5 import (
        Qwen2_5Attention as QwenAttentionBase,
        Qwen2_5DecoderLayer as QwenDecoderLayerBase,
        Qwen2_5Model as QwenModelBase,
        apply_rotary_pos_emb as apply_rotary_pos_emb_fn,
        Qwen2RMSNorm,
    )
    QWEN_VERSION = "2_5"
except Exception:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention as QwenAttentionBase,
        Qwen2DecoderLayer as QwenDecoderLayerBase,
        Qwen2Model as QwenModelBase,
        apply_rotary_pos_emb as apply_rotary_pos_emb_fn,
        Qwen2RMSNorm,
    )
    QWEN_VERSION = "2"

from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import get_peft_model, LoraConfig, TaskType

# ==================================================================================
# 第 2 部分：基础 Embedding 模块 (Qwen 需要)
# ==================================================================================

class ValueEmbedding_Qwen(nn.Module):
    """将 (值, 掩码) 对投影到 d_model"""
    def __init__(self, c_in, d_model):
        super(ValueEmbedding_Qwen, self).__init__()
        self.projection = nn.Linear(c_in, d_model)
    def forward(self, x):
        return self.projection(x)


class VariableEmbedding_Qwen(nn.Module):
    """为 D 个时间序列变量创建 embedding"""
    def __init__(self, c_in, d_model):
        super(VariableEmbedding_Qwen, self).__init__()
        self.var_emb = nn.Embedding(c_in, d_model)
        self.c_in = c_in

    def forward(self, x):
        B, L, D = x.shape
        var_indices = torch.arange(D, device=x.device, dtype=torch.long).unsqueeze(0).unsqueeze(0).expand(B, L, D)
        return self.var_emb(var_indices) # (B, L, D, H)


# ==================================================================================
# 第 3 部分：CT-RoPE (Continuous Time Rotary Position Embedding) 模块
# ==================================================================================

class CTRoPEQwen2Attention(QwenAttentionBase):
    def __init__(self, config, layer_idx: int):
        # 必须调用父类初始化，确保 num_heads / num_key_value_heads / head_dim / hidden_size 等属性就绪
        super().__init__(config, layer_idx)

    def _repeat_kv(self, kv: torch.Tensor, n_rep: int) -> torch.Tensor:
        # kv: (B, H_kv, L, hd) -> repeat heads to (B, H, L, hd)
        if n_rep == 1:
            return kv
        bsz, n_kv, seqlen, hd = kv.shape
        return kv.unsqueeze(2).expand(bsz, n_kv, n_rep, seqlen, hd).reshape(bsz, n_kv * n_rep, seqlen, hd)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        timestamps: Optional[torch.Tensor] = None, # 连续时间戳 (B, L)
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Qwen 使用 GQA: q 用 num_heads，kv 用 num_key_value_heads
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)             # (B, H, L, hd)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)       # (B, H_kv, L, hd)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)   # (B, H_kv, L, hd)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # 应用 CT-RoPE：使用连续时间戳替代离散 position_ids
        if timestamps is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            timestamps_exp = timestamps.unsqueeze(1).unsqueeze(-1).to(key_states.dtype)  # (B, 1, L, 1)
            query_states, key_states = apply_rotary_pos_emb_fn(query_states, key_states, cos, sin, timestamps_exp)
        elif position_ids is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb_fn(query_states, key_states, cos, sin, position_ids)

        # 追加缓存
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # 将 KV 头重复到与 Q 头一致
        if self.num_key_value_heads != self.num_heads:
            n_rep = self.num_heads // self.num_key_value_heads
            key_states = self._repeat_kv(key_states, n_rep)
            value_states = self._repeat_kv(value_states, n_rep)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if cache_position is not None:
                mask_q_len = attention_mask.shape[-2]
                if q_len != mask_q_len:
                    start_row = mask_q_len - q_len
                    attention_mask = attention_mask[:, :, start_row:, :]
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)  # (B, H, q_len, hd)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()          # (B, q_len, H, hd)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size) # (B, q_len, hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2DecoderLayerWithCTRoPE(QwenDecoderLayerBase):
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
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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


class PLM4TS_Base(QwenModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([Qwen2DecoderLayerWithCTRoPE(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.embed_tokens = None
        self.gradient_checkpointing = False

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
            raise ValueError("PLM4TS_Base expects `inputs_embeds`, not `input_ids`.")
        if inputs_embeds is None:
            raise ValueError("PLM4TS_Base requires `inputs_embeds`.")

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                timestamps=timestamps,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

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


# ==================================================================================
# 第 4 部分：CT-RoPE 注入和 LoRA 配置函数
# ==================================================================================

def inject_CTRoPE_to_model(model: PreTrainedModel):
    """替换模型中的 QwenAttention 为 CTRoPEQwen2Attention 并复制权重"""
    for name, module in model.named_children():
        if isinstance(module, QwenAttentionBase):
            config = module.config
            layer_idx = module.layer_idx
            new_attn = CTRoPEQwen2Attention(config, layer_idx)

            new_attn.q_proj.weight.data.copy_(module.q_proj.weight.data)
            new_attn.k_proj.weight.data.copy_(module.k_proj.weight.data)
            new_attn.v_proj.weight.data.copy_(module.v_proj.weight.data)
            new_attn.o_proj.weight.data.copy_(module.o_proj.weight.data)

            if getattr(module.q_proj, "bias", None) is not None and getattr(new_attn.q_proj, "bias", None) is not None:
                new_attn.q_proj.bias.data.copy_(module.q_proj.bias.data)
            if getattr(module.k_proj, "bias", None) is not None and getattr(new_attn.k_proj, "bias", None) is not None:
                new_attn.k_proj.bias.data.copy_(module.k_proj.bias.data)
            if getattr(module.v_proj, "bias", None) is not None and getattr(new_attn.v_proj, "bias", None) is not None:
                new_attn.v_proj.bias.data.copy_(module.v_proj.bias.data)
            if getattr(module.o_proj, "bias", None) is not None and getattr(new_attn.o_proj, "bias", None) is not None:
                new_attn.o_proj.bias.data.copy_(module.o_proj.bias.data)

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
        if 'lora' not in name:
            param.requires_grad = False
    return lora_model


# ==================================================================================
# 第 5 部分：ISTS_PLM 主模型 (istsplm_forecast)
# ==================================================================================

class istsplm_forecast(nn.Module):
    def __init__(self, args):
        super(istsplm_forecast, self).__init__()

        self.args = args
        self.device = args.device
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.use_lora = args.use_lora

        self.plm_name = args.plm_path
        self.num_types = args.num_types
        self.n_plm_layers = 2

        # 1. 配置
        self.config = AutoConfig.from_pretrained(self.plm_name, trust_remote_code=True)
        self.config.use_cache = True

        # 2. Embedding 层
        self.value_embedding = ValueEmbedding_Qwen(1, self.d_model)
        self.var_embedding = nn.Embedding(self.num_types, self.d_model)

        # 3. PLM 模块 (两个独立的 PLM 实例)
        gpts = []
        for _ in range(self.n_plm_layers):
            model = PLM4TS_Base.from_pretrained(
                self.plm_name,
                config=self.config,
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
            )
            model = inject_CTRoPE_to_model(model)
            if args.use_lora:
                model = configure_lora(args, model)
            gpts.append(model)
        self.gpts = nn.ModuleList(gpts)  # [时间PLM, 变量PLM]

        # 4. Projection Layer
        self.ln_proj = nn.LayerNorm(self.d_model)

        # 5. Forecasting Head
        self.forecasting_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LeakyReLU(),
            nn.Linear(self.d_model // 2, self.num_types)
        )

        # 6. Prompt
        self.patch_len = args.patch_len
        self.prompt_len = args.prompt_len
        self.prompt_embed = nn.Parameter(torch.randn(1, self.prompt_len, self.d_model))

    def forecasting(self, batch_dict, n_vars_to_predict=None):
        observed_data = batch_dict["observed_data"].to(self.device)  # (B, L_obs, D)
        observed_mask = batch_dict["observed_mask"].to(self.device)  # (B, L_obs, D)
        observed_tp_raw = batch_dict["observed_tp"].to(self.device)  # (B, L_obs) 或 (B, 1, L_obs)
        tp_to_predict = batch_dict["tp_to_predict"].to(self.device)  # (B, L_pred)

        B, L_obs, D = observed_data.shape
        L_pred = tp_to_predict.shape[1]

        # 1a. Value Embedding
        x_flat = observed_data.permute(0, 2, 1).reshape(B * D, L_obs, 1)
        value_embed = self.value_embedding(x_flat)          # (B*D, L_val, d_model)
        L_val = value_embed.size(1)

        # 1b. Times Embedding (CT-RoPE) — 对齐时间长度
        observed_tp = observed_tp_raw.reshape(B, -1)        # (B, L_time_raw)
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

        # 1c. Prompt 拼接
        prompt_embed = self.prompt_embed.expand(B * D, -1, -1)
        inputs_embeds = torch.cat([prompt_embed, value_embed], dim=1)  # (B*D, L_prompt + L_val, d_model)

        tt_prompt = torch.zeros(B * D, self.prompt_len, device=self.device)
        tt_with_prompt = torch.cat([tt_prompt, tt], dim=1)            # (B*D, L_prompt + L_val)

        # 1d. Attention Mask
        time_attn_mask_data_flat = observed_mask.permute(0, 2, 1).reshape(B * D, L_obs)  # (B*D, L_obs)
        if L_obs != L_val:
            if L_obs >= L_val:
                time_attn_mask_data_flat = time_attn_mask_data_flat[:, -L_val:]
            else:
                pad_m = torch.zeros(B * D, L_val - L_obs, device=self.device, dtype=time_attn_mask_data_flat.dtype)
                time_attn_mask_data_flat = torch.cat([time_attn_mask_data_flat, pad_m], dim=1)

        time_attn_mask_prompt = torch.ones(B * D, self.prompt_len, device=self.device)
        time_attn_mask = torch.cat([time_attn_mask_prompt, time_attn_mask_data_flat], dim=1)  # (B*D, L_total)
        time_attn_mask = (1.0 - time_attn_mask) * torch.finfo(torch.float32).min
        time_attn_mask = time_attn_mask.unsqueeze(1).unsqueeze(1)  # (B*D, 1, 1, L_total)

        # 2. 时间 PLM 前向
        outputs = self.gpts[0](
            inputs_embeds=inputs_embeds,
            attention_mask=time_attn_mask,
            timestamps=tt_with_prompt,
        ).last_hidden_state

        # 3. 掩码池化（去掉 prompt）
        observed_mask_pooled = observed_mask.permute(0, 2, 1).reshape(B * D, L_obs, 1)  # (B*D, L_obs, 1)
        if L_obs != L_val:
            if L_obs >= L_val:
                observed_mask_pooled = observed_mask_pooled[:, -L_val:, :]
            else:
                pad_m3d = torch.zeros(B * D, L_val - L_obs, 1, device=self.device, dtype=observed_mask_pooled.dtype)
                observed_mask_pooled = torch.cat([observed_mask_pooled, pad_m3d], dim=1)

        outputs_data = outputs[:, self.prompt_len:]   # (B*D, L_val, d_model)
        n_nonmask = (observed_mask_pooled.sum(dim=1) + 1e-8)  # (B*D, 1)
        outputs = (outputs_data * observed_mask_pooled).sum(dim=1) / n_nonmask  # (B*D, d_model)
        outputs = self.ln_proj(outputs).view(B, D, -1)  # (B, D, d_model)

        # 4. 变量 PLM
        var_embedding = self.var_embedding.weight.unsqueeze(0).expand(B, D, -1)
        outputs = outputs + var_embedding

        var_position_ids = torch.arange(D, device=self.device).unsqueeze(0).expand(B, D)
        var_attn_mask = torch.zeros(B, D, D, device=self.device, dtype=torch.float32).unsqueeze(1)  # (B, 1, D, D)

        outputs = self.gpts[1](
            inputs_embeds=outputs,
            attention_mask=var_attn_mask,
            position_ids=var_position_ids
        ).last_hidden_state  # (B, D, d_model)

        # 5. 预测
        forecast_vars = self.forecasting_head(outputs)  # (B, D, D)
        regression_value = forecast_vars[:, :, 0]       # (B, D)
        forecast = regression_value.unsqueeze(1).expand(-1, L_pred, -1)  # (B, L_pred, D)

        if n_vars_to_predict is not None:
            forecast = forecast[:, :, :n_vars_to_predict]
        return forecast

    def forward(self, batch_dict, n_vars_to_predict=None):
        return self.forecasting(batch_dict, n_vars_to_predict)
