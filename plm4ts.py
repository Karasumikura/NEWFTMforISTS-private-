import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List

from transformers import AutoConfig, PreTrainedModel, Qwen2Model, Qwen2Config
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

# 从我们修复的 embed.py 中导入
from models.embed import DataEmbedding_ITS_for_CTRoPE, VariableEmbedding, ValueEmbedding, TimeEmbedding

# ==================================================================================
# 1. 新的 CTRoPE (连续时间旋转位置编码) 模块 (来自参考文件)
# ==================================================================================

class CTRoPERotaryEmbedding(nn.Module):
    """
    实现连续时间旋转位置编码 (CTRoPE) (来自参考文件)
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
# 2. 自定义 Qwen 架构以使用 CTRoPE (来自参考文件, 适配 Qwen2)
# ==================================================================================

class CustomQwen2Attention(Qwen2Attention):
    """
    修改 Qwen2Attention 以使用 CTRoPE。
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
        # ------------------
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids=None) # 传递 position_ids=None

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # (剩余的注意力逻辑与 Qwen2Attention 相同)
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # 使用 Qwen2 的 _flash_attention_forward 或 _sdpa_attention_forward
        if self._use_flash_attention_2:
             attn_output = self._flash_attention_forward(
                 query_states, key_states, value_states, attention_mask, q_len, dropout=self.attention_dropout
             )
        else:
             attn_output = self._sdpa_attention_forward(
                 query_states, key_states, value_states, attention_mask, q_len, dropout=self.attention_dropout
             )

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # 将被忽略
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # --- 关键修改 ---
        # 提取 'timestamps' 以传递给注意力层
        timestamps = kwargs.get("timestamps")
        if timestamps is None:
            raise ValueError("CustomQwen2DecoderLayer 必须接收 'timestamps' 参数")
        # ------------------

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # (self-attention)
        attn_outputs, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=None, # 传递 None
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            timestamps=timestamps, # <-- 传递 timestamps
        )
        hidden_states = residual + attn_outputs

        # (Feed Forward)
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
        self.layers = nn.ModuleList(
            [CustomQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # ------------------
        
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # 将被忽略
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
        # --- 关键新增 ---
        timestamps: Optional[torch.Tensor] = None,
        # ------------------
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        if timestamps is None:
             raise ValueError("CustomQwen2Model 必须接收 'timestamps' 参数")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=None, # 传递 None
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                timestamps=timestamps, # <-- 传递 timestamps
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
# 3. 最终的 ists_plm (Qwen + CTRoPE + LoRA) 模型
# !! 注意：我将这个新模型重命名为 'ists_plm'，以匹配 classification.py 的调用 !!
# ==================================================================================

class ists_plm(nn.Module):
    
    def __init__(self, opt):
        super(ists_plm, self).__init__()
        self.feat_dim = opt.input_dim # 应该是 2 (value, mask)
        self.d_model = opt.d_model
        self.device = opt.device
        self.n_var = opt.num_types # 从 classification.py 获取变量数

        # 1. 使用来自 embed.py 的新嵌入层
        self.enc_embedding = DataEmbedding_ITS_for_CTRoPE(
            self.feat_dim, self.d_model, self.n_var, 
            device=opt.device, dropout=opt.dropout
        )

        # 2. 加载 Qwen 配置
        # plm_path 来自 classification.py 的参数 (例如 "Qwen/Qwen2-1.5B-Instruct")
        plm_name = opt.plm_path
        
        config = AutoConfig.from_pretrained(
            plm_name,
            hidden_size=self.d_model,
            num_hidden_layers=opt.n_te_plmlayer, # 时间 PLM 的层数
            # 自动推断
            # num_attention_heads=opt.d_model // 64, 
            # num_key_value_heads=opt.d_model // 128,
            # intermediate_size=self.d_model * 4,
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
        )

        # 3. 实例化两个 PLM
        self.gpts = nn.ModuleList()
        
        # PLM 1: 时间感知模型 (使用我们的 CustomQwen2Model 和 CTRoPE)
        print(f"Loading Time-Aware PLM (CustomQwen2Model) with {opt.n_te_plmlayer} layers...")
        self.time_aware_plm = CustomQwen2Model(config)
        
        # PLM 2: 变量感知模型 (使用 *标准* Qwen2Model 和 *标准* RoPE)
        config_var = AutoConfig.from_pretrained(
            plm_name,
            hidden_size=self.d_model,
            num_hidden_layers=opt.n_st_plmlayer, # 变量 PLM 的层数
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
        )
        print(f"Loading Variable-Aware PLM (Standard Qwen2Model) with {opt.n_st_plmlayer} layers...")
        self.variable_aware_plm = Qwen2Model(config_var)
        
        # 4. 应用 LoRA
        lora_config = LoraConfig(
            r=opt.lora_r,
            lora_alpha=opt.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 适用于 Qwen2
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM, 
        )
        
        print("Applying LoRA to Time-Aware PLM...")
        self.time_aware_plm = get_peft_model(self.time_aware_plm, lora_config)
        self.time_aware_plm.print_trainable_parameters()
        
        print("Applying LoRA to Variable-Aware PLM...")
        self.variable_aware_plm = get_peft_model(self.variable_aware_plm, lora_config)
        self.variable_aware_plm.print_trainable_parameters()
        
        # 将两个 PLM 添加到 ModuleList
        self.gpts.append(self.time_aware_plm)
        self.gpts.append(self.variable_aware_plm)

        self.ln_proj = nn.LayerNorm(self.d_model)
    
    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
        """
        observed_tp: (B, L, D) - 连续时间戳
        observed_data: (B, L, D) - 观测值
        observed_mask: (B, L, D) - 掩码
        """
        B, L, D = observed_data.shape
        
        # 1. 嵌入
        # inputs_embeds: (B*D, L+1, d_model)
        # tt_with_prompt: (B*D, L+1)
        # var_embedding: (B, 1, D, d_model)
        inputs_embeds, tt_with_prompt, var_embedding = self.enc_embedding(
            observed_tp, observed_data, observed_mask
        )
        
        # 2. 时间感知 PLM (CTRoPE)
        # (创建注意力掩码)
        # (B, D, L)
        time_attn_mask_data = observed_mask.permute(0, 2, 1)
        # (B*D, L)
        time_attn_mask_data_flat = time_attn_mask_data.reshape(B * D, L)
        # (B*D, 1) - 为 prompt 添加 '1'
        time_attn_mask_prompt = torch.ones((B * D, 1), device=self.device, dtype=torch.long)
        # (B*D, L+1)
        time_attn_mask = torch.cat([time_attn_mask_prompt, time_attn_mask_data_flat], dim=1)
        
        outputs = self.gpts[0](
            inputs_embeds=inputs_embeds,
            attention_mask=time_attn_mask,
            timestamps=tt_with_prompt
        ).last_hidden_state # (B*D, L+1, d_model)

        # 3. 掩码池化 (与原始代码相同)
        observed_mask_flat = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1) # (B*D, L, 1)
        observed_mask_flat = torch.cat([torch.ones_like(observed_mask_flat[:,:1]), observed_mask_flat], dim=1) # (B*D, L+1, 1)
        
        n_nonmask = observed_mask_flat.sum(dim=1)  # (B*D, 1)
        outputs = (outputs * observed_mask_flat).sum(dim=1) / (n_nonmask + 1e-8) # (B*D, d_model)
        
        outputs = self.ln_proj(outputs.view(B, D, -1))  # (B, D, d_model)
        
        # 4. 变量感知 PLM (标准 RoPE)
        outputs = outputs + var_embedding.squeeze(1) # (B, D, d_model)
        
        # (为变量PLM创建离散的 position_ids 和 attention_mask)
        var_position_ids = torch.arange(D, device=self.device).unsqueeze(0).expand(B, D)
        var_attn_mask = torch.ones(B, D, device=self.device, dtype=torch.long) # 假设所有变量都存在
        
        outputs = self.gpts[1](
            inputs_embeds=outputs,
            attention_mask=var_attn_mask,
            position_ids=var_position_ids
            # 这里不传递 timestamps，使用 Qwen2 内部的标准 RoPE
        ).last_hidden_state # (B, D, d_model)
        
        outputs = outputs.view(B, -1)
        return outputs # (B, D*d_model)

# ==================================================================================
# 4. 保留其他模型 (来自你的 plm4ts.py 文件)
# 注意：这些模型 (vector, set) 使用的是旧的/不同的嵌入逻辑，
# 你需要确保它们与 embed.py 中的相应嵌入类 (如果存在) 配合使用。
# 为了完整性，我从你的原始文件中复制了它们，但未做修改。
# ==================================================================================

# class istsplm_vector(nn.Module):
# ... (你原始的 istsplm_vector 代码) ...
    
# class istsplm_set(nn.Module):
# ... (你原始的 istsplm_set 代码) ...

# class istsplm_forecast(nn.Module):
# ... (你原始的 istsplm_forecast 代码) ...

# class istsplm_vector_forecast(nn.Module):
# ... (你原始的 istsplm_vector_forecast 代码) ...

# class istsplm_set_forecast(nn.Module):
# ... (你原始的 istsplm_set_forecast 代码) ...
