import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List

# ==================================================================================
# 节 1: 核心依赖
# (Transformers, PEFT for LoRA, BitsAndBytes for INT4)
# ==================================================================================
from transformers import (
    AutoConfig, 
    PreTrainedModel, 
    Qwen2Model, 
    Qwen2Config, 
    BitsAndBytesConfig
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, 
    Qwen2DecoderLayer, 
    Qwen2Model, 
    Qwen2PreTrainedModel,
    apply_rotary_pos_emb,
    Qwen2MLP
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import get_peft_model, LoraConfig, TaskType

# ==================================================================================
# 节 2: 基础嵌入模块
# (ValueEmbedding, VariableEmbedding)
# ==================================================================================

class ValueEmbedding(nn.Module):
    """
    值嵌入: 将 (value, mask) 对投影到 d_model 维度
    c_in 通常是 2
    """
    def __init__(self, c_in, d_model):
        super(ValueEmbedding, self).__init__()
        self.projection = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.projection(x)

class VariableEmbedding(nn.Module):
    """
    变量嵌入: 为 D 个不同的时间序列变量创建嵌入
    n_var 是变量的数量 (例如 opt.num_types)
    """
    def __init__(self, n_var, d_model):
        super(VariableEmbedding, self).__init__()
        self.varible_emb = nn.Embedding(n_var, d_model)

    def forward(self, x):
        x = self.varible_emb(x.long())
        return x

# ==================================================================================
# 节 3: CT-RoPE (连续时间旋转位置编码)
# ==================================================================================

class CTRoPERotaryEmbedding(nn.Module):
    """
    实现连续时间旋转位置编码 (CT-RoPE)。
    它根据 *连续的* `timestamps` 动态生成 cos 和 sin 矩阵。
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        # CTRoPE 的核心: 角频率 omega_i = 10000^(-2i/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, timestamps):
        """
        Args:
            x: (torch.Tensor) 输入张量 (仅用于获取 dtype 和 device)
            timestamps: (torch.Tensor) 形状为 [batch_size, seq_len] 的连续时间戳
        """
        # t 形状: [batch_size, seq_len]
        # inv_freq 形状: [dim // 2]
        
        # 计算角度 theta(t) = t * omega_i
        # freqs 形状: [batch_size, seq_len, dim // 2]
        with torch.autocast(device_type=x.device.type, enabled=False):
            t = timestamps.float()
            freqs = torch.einsum("b s, d -> b s d", t, self.inv_freq.to(t.device))
        
        # emb 形状: [batch_size, seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# ==================================================================================
# 节 4: 自定义 Qwen 架构 (以支持 CT-RoPE)
# ==================================================================================

class CustomQwen2Attention(Qwen2Attention):
    """
    修改 Qwen2Attention 以使用 CTRoPERotaryEmbedding。
    """
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        # --- 关键替换 ---
        # 用 CTRoPERotaryEmbedding 替换标准的 Qwen2RotaryEmbedding
        self.rotary_emb = CTRoPERotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        # ------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # 此参数将被忽略
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # --- 关键修改 ---
        # 从 kwargs 中提取 'timestamps'
        if "timestamps" not in kwargs:
            raise ValueError("CustomQwen2Attention 必须接收 'timestamps' 参数")
        timestamps = kwargs["timestamps"]
        # ------------------

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # --- 关键调用 ---
        # 使用 timestamps 调用 CTRoPE，而不是 position_ids
        cos, sin = self.rotary_emb(value_states, timestamps=timestamps)
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids=None) # 传递 position_ids=None

        # (剩余的注意力逻辑与 Qwen2Attention 相同)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        if self._use_flash_attention_2:
             attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=self.attention_dropout)
        else:
             attn_output = self._sdpa_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=self.attention_dropout)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

class CustomQwen2DecoderLayer(Qwen2DecoderLayer):
    """
    修改 Qwen2DecoderLayer 以使用 CustomQwen2Attention
    并传递 'timestamps'
    """
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super(Qwen2DecoderLayer, self).__init__() # 绕过 Qwen2DecoderLayer 的 __init__
        self.hidden_size = config.hidden_size
        
        # --- 关键替换 ---
        self.self_attn = CustomQwen2Attention(config, layer_idx)
        # ------------------
        
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        
        # --- 关键修改 ---
        # 提取 'timestamps' 以传递给注意力层
        timestamps = kwargs.get("timestamps")
        if timestamps is None:
            raise ValueError("CustomQwen2DecoderLayer 必须接收 'timestamps' 参数")
        # ------------------

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=None,
            past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
            timestamps=timestamps, # <-- 传递 timestamps
        )
        hidden_states = residual + attn_outputs
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

class CustomQwen2Model(Qwen2Model):
    """
    修改 Qwen2Model 以使用 CustomQwen2DecoderLayer
    并接受 'timestamps' 作为 forward 参数
    """
    def __init__(self, config: Qwen2Config):
        super(Qwen2Model, self).__init__(config) # 绕过 Qwen2Model 的 __init__
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # --- 关键替换 ---
        # 使用我们的自定义 Decoder 层
        self.layers = nn.ModuleList([CustomQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # ------------------
        
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, timestamps=None):
        
        # --- 关键新增 ---
        if timestamps is None:
             raise ValueError("CustomQwen2Model 必须接收 'timestamps' 参数")
        # ------------------

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("必须且只能提供 'input_ids' 或 'inputs_embeds' 中的一个")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx]
            
            # --- 关键修改 ---
            layer_outputs = decoder_layer(
                hidden_states, attention_mask=attention_mask, position_ids=None,
                past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                timestamps=timestamps, # <-- 传递 timestamps
            )
            # ------------------
            
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
            last_hidden_state=hidden_states, past_key_values=next_cache,
            hidden_states=all_hidden_states, attentions=all_self_attns
        )

# ==================================================================================
# 节 5: 数据嵌入层 (CT-RoPE 专用)
# ==================================================================================
class DataEmbedding_ITS_for_CTRoPE(nn.Module):
    """
    修改后的数据嵌入层，用于 CTRoPE。
    
    1. 它 *不* 将时间嵌入添加到值嵌入中 (因为 CT-RoPE 会在 Transformer 内部处理)。
    2. 它返回对齐好的时间戳张量 `tt_with_prompt`。
    """
    def __init__(self, c_in, d_model, n_var, device=None, dropout=0.1):
        super(DataEmbedding_ITS_for_CTRoPE, self).__init__()
        self.d_model = d_model
        
        # 依赖于 节 2 中定义的基础模块
        self.value_embedding = ValueEmbedding(c_in=c_in, d_model=d_model).to(device)
        self.variable_embedding = VariableEmbedding(n_var=n_var, d_model=d_model).to(device)
        
        self.vars = torch.arange(n_var).to(device)
        self.device = device
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tt, x, x_mark=None):
        """
        tt: (B, L, D) - 连续时间戳
        x: (B, L, D) - 观测值
        x_mark: (B, L, D) - 掩码 (1=观测到, 0=未观测)
        """
        B, L, D = x.shape
        
        # --- 准备时间戳 ---
        # 为变量 Prompt 创建一个 "0" 时间戳 (B, 1, D)
        prompt_tt = torch.zeros((B, 1, D), device=self.device, dtype=tt.dtype)
        # (B, L+1, D)
        tt_with_prompt = torch.cat([prompt_tt, tt], dim=1)
        # (B*D, L+1)
        tt_with_prompt_flat = tt_with_prompt.permute(0, 2, 1).reshape(B * D, L + 1)
        
        # --- 准备值嵌入 ---
        x_val = x.unsqueeze(dim=-1) # (B, L, D, 1)
        x_mask_val = x_mark.unsqueeze(dim=-1) # (B, L, D, 1)
        # c_in=2, 输入是 (值, 掩码)
        x_int = torch.cat([x_val, x_mask_val], dim=-1) # (B, L, D, 2)
        value_emb = self.value_embedding(x_int) # (B, L, D, d_model)

        # --- 关键点 ---
        # 不添加位置编码。位置信息将由 CT-RoPE 稍后注入。
        x_emb = value_emb 
        
        # --- 变量 Prompt 逻辑 ---
        # (B, 1, D, d_model)
        vars_prompt = self.variable_embedding(self.vars.view(1, 1, -1).repeat(B, 1, 1))
        
        # (B, L+1, D, d_model)
        x_emb_with_prompt = torch.cat([vars_prompt, x_emb], dim=1)
        
        # (B*D, L+1, d_model)
        x_flat = x_emb_with_prompt
