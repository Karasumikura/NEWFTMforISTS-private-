import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List

# ==================================================================================
# 第 1 部分：新模型 (Qwen / LoRA) 的依赖项
# (!!!) 不再导入 GPT2Model_wope 或 BertModel_wope (!!!)
# ==================================================================================
from transformers import (
    AutoConfig, 
    PreTrainedModel, 
    Qwen2Model, 
    Qwen2Config, 
    # BitsAndBytesConfig  <-- (!!!) 移除：不再需要 INT4
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, 
    Qwen2DecoderLayer, 
    Qwen2Model, 
    Qwen2PreTrainedModel,
    apply_rotary_pos_emb,
    Qwen2MLP,
    Qwen2RMSNorm  # <--- (!!!) 修复点 1: 导入 Qwen2RMSNorm
)
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
    def __init__(self, n_var, d_model):
        super(VariableEmbedding_Qwen, self).__init__()
        self.varible_emb = nn.Embedding(n_var, d_model)
    def forward(self, x):
        x = self.varible_emb(x.long())
        return x

# ==================================================================================
# 第 3 部分：CT-RoPE (连续时间旋转位置编码)
# ==================================================================================

class CTRoPERotaryEmbedding(nn.Module):
    """实现 CT-RoPE"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    def forward(self, x, timestamps):
        with torch.autocast(device_type=x.device.type, enabled=False):
            t = timestamps.float()
            freqs = torch.einsum("b s, d -> b s d", t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# ==================================================================================
# 第 4 部分：自定义 Qwen 架构 (以支持 CT-RoPE)
# ==================================================================================

class CustomQwen2Attention(Qwen2Attention):
    """修改后的 Qwen2Attention，用于使用 CTRoPERotaryEmbedding"""
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.rotary_emb = CTRoPERotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta,
        )
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        if "timestamps" not in kwargs:
            raise ValueError("CustomQwen2Attention 必须接收 'timestamps' 参数")
        timestamps = kwargs["timestamps"]
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
        cos, sin = self.rotary_emb(value_states, timestamps=timestamps)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids=None)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # 注意力逻辑 (Flash 或 SDPA)
        if hasattr(self, '_use_flash_attention_2') and self._use_flash_attention_2:
            attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=self.attention_dropout)
        elif hasattr(self, '_sdpa_attention_forward'):
            attn_output = self._sdpa_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=self.attention_dropout)
        else:
            # 兼容旧版 transformers
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

class CustomQwen2DecoderLayer(Qwen2DecoderLayer):
    """修改后的 Qwen2DecoderLayer，用于传递 'timestamps'"""
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super(Qwen2DecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = CustomQwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        # <--- (!!!) 修复点 2: 将 nn.LayerNorm 替换为 Qwen2RMSNorm
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        timestamps = kwargs.get("timestamps")
        if timestamps is None:
            raise ValueError("CustomQwen2DecoderLayer 必须接收 'timestamps' 参数")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=None,
            past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
            timestamps=timestamps,
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
    """修改后的 Qwen2Model，用于接受 'timestamps'"""
    def __init__(self, config: Qwen2Config):
        super(Qwen2Model, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([CustomQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # <--- (!!!) 修复点 3: 将 nn.LayerNorm 替换为 Qwen2RMSNorm
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, timestamps=None):
        if timestamps is None:
            raise ValueError("CustomQwen2Model 必须接收 'timestamps'
