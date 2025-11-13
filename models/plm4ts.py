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
            raise ValueError("CustomQwen2Model 必须接收 'timestamps' 参数")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("必须提供 'input_ids' 或 'inputs_embeds'，但不能两者都提供")
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
                hidden_states, attention_mask=attention_mask, position_ids=None,
                past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                timestamps=timestamps,
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
            last_hidden_state=hidden_states, past_key_values=next_cache,
            hidden_states=all_hidden_states, attentions=all_self_attns
        )

# ==================================================================================
# 第 5 部分：数据 Embedding 层 (专用于 CT-RoPE)
# ==================================================================================
class DataEmbedding_ITS_for_CTRoPE(nn.Module):
    """为 CTRoPE 修改的数据 Embedding 层"""
    def __init__(self, c_in, d_model, n_var, device=None, dropout=0.1):
        super(DataEmbedding_ITS_for_CTRoPE, self).__init__()
        self.d_model = d_model
        # 使用第 2 部分的 embedding 模块
        self.value_embedding = ValueEmbedding_Qwen(c_in=c_in, d_model=d_model).to(device)
        self.variable_embedding = VariableEmbedding_Qwen(n_var=n_var, d_model=d_model).to(device)
        self.vars = torch.arange(n_var).to(device)
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, tt, x, x_mark=None):
        B, L, D = x.shape
        prompt_tt = torch.zeros((B, 1, D), device=self.device, dtype=tt.dtype)
        tt_with_prompt = torch.cat([prompt_tt, tt], dim=1)
        tt_with_prompt_flat = tt_with_prompt.permute(0, 2, 1).reshape(B * D, L + 1)
        x_val = x.unsqueeze(dim=-1) 
        x_mask_val = x_mark.unsqueeze(dim=-1) 
        x_int = torch.cat([x_val, x_mask_val], dim=-1)
        value_emb = self.value_embedding(x_int)
        x_emb = value_emb 
        vars_prompt = self.variable_embedding(self.vars.view(1, 1, -1).repeat(B, 1, 1))
        x_emb_with_prompt = torch.cat([vars_prompt, x_emb], dim=1)
        x_flat = x_emb_with_prompt.permute(0, 2, 1, 3).reshape(B * D, L + 1, self.d_model)
        return self.dropout(x_flat), tt_with_prompt_flat, vars_prompt

# ==================================================================================
# 第 6 部分：主模型 (istsplm_forecast)
# (!!!) 这是唯一剩下的模型！(!!!)
# ==================================================================================
class istsplm_forecast(nn.Module):
    
    def __init__(self, opt):
        super(istsplm_forecast, self).__init__()
        
        self.pred_len = opt.pred_len
        self.input_len = opt.input_len
        self.d_model = opt.d_model
        self.device = opt.device
        self.n_var = opt.num_types # 从 opt.num_types 获取变量数量
        
        # --- 1. Embedding 层 (来自第 5 部分) ---
        self.enc_embedding = DataEmbedding_ITS_for_CTRoPE(
            c_in=2, 
            d_model=self.d_model, 
            n_var=self.n_var, 
            device=opt.device, 
            dropout=opt.dropout
        )

        self.gpts = nn.ModuleList()
        
        # --- 2. 加载模型 (预训练权重) ---
        plm_name = opt.plm_path
        
        # (!!!) 移除：删除 BitsAndBytesConfig
        # quantization_config = BitsAndBytesConfig(...)
        
        print(f"正在从 {plm_name} 加载 Time-Aware PLM (CustomQwen2Model) [完整模型]...")
        time_aware_plm = CustomQwen2Model.from_pretrained(
            plm_name,
            # quantization_config=quantization_config, # (!!!) 移除
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
            # (!!!) 注意: 加载完整模型可能需要指定 torch_dtype=torch.bfloat16 (如果内存不足)
            # torch_dtype=torch.bfloat16 
        )
        
        print(f"正在从 {plm_name} 加载 Variable-Aware PLM (Standard Qwen2Model) [完整模型]...")
        variable_aware_plm = Qwen2Model.from_pretrained(
            plm_name,
            # quantization_config=quantization_config, # (!!!) 移除
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
            # torch_dtype=torch.bfloat16
        )
            
        # --- 3. 应用 LoRA ---
        lora_config = LoraConfig(
            r=opt.lora_r,
            lora_alpha=opt.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            # <--- (!!!) 修复点 4: 修复 `prepare_inputs_for_generation` 错误
            task_type=TaskType.FEATURE_EXTRACTION, 
        )
        
        print("应用 LoRA 到 Time-Aware PLM...")
        time_aware_plm = get_peft_model(time_aware_plm, lora_config)
        time_aware_plm.print_trainable_parameters()
        
        print("应用 LoRA 到 Variable-Aware PLM...")
        variable_aware_plm = get_peft_model(variable_aware_plm, lora_config)
        variable_aware_plm.print_trainable_parameters()
        
        self.gpts.append(time_aware_plm)
        self.gpts.append(variable_aware_plm)
            
        # --- 4. 预测头 (Forecasting Head) ---
        self.ln_proj = nn.LayerNorm(self.d_model) # <--- 注意: 这个 LayerNorm 没报错，因为它处理的是 375 行的 'outputs'
                                                    # 它是在 PLM 之外的，并且输入类型可能是 float32
        
        # Qwen 的输出将是 (B, D * d_model)
        qwen_output_dim = self.d_model * self.n_var 
        
        # 要求的预测输出是 (B, pred_len * D)
        prediction_dim = self.pred_len * self.n_var
        
        # 最终的预测头
        self.forecasting_head = nn.Linear(qwen_output_dim, prediction_dim)
        
        # (!!!) 保留 'predict_decoder' 以防 'evaluation.py' 使用它
        # 尽管我们的主 'forecasting' 不会用它。
        self.predict_decoder = nn.Sequential(
            nn.Linear(opt.d_model+1, opt.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(opt.d_model, opt.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(opt.d_model, 1)
        ).to(opt.device)
        

    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        """
        用于预测的主前向传播函数。
        """
        B, L, D = observed_data.shape
        
        # 1. Embedding
        inputs_embeds, tt_with_prompt, var_embedding = self.enc_embedding(
            observed_tp, observed_data, observed_mask
        )

        # 2. PLM 1 (时间感知)
        time_attn_mask_data = observed_mask.permute(0, 2, 1)
        time_attn_mask_data_flat = time_attn_mask_data.reshape(B * D, L)
        time_attn_mask_prompt = torch.ones((B * D, 1), device=self.device, dtype=torch.long)
        time_attn_mask = torch.cat([time_attn_mask_prompt, time_attn_mask_data_flat], dim=1)
        
        outputs = self.gpts[0](
            inputs_embeds=inputs_embeds,
            attention_mask=time_attn_mask,
            timestamps=tt_with_prompt # <-- 传递连续时间
        ).last_hidden_state

        # 3. Masked Pooling (掩码池化)
        observed_mask_pooled = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1)
        observed_mask_pooled = torch.cat([torch.ones_like(observed_mask_pooled[:,:1]), observed_mask_pooled], dim=1)
        n_nonmask = (observed_mask_pooled.sum(dim=1) + 1e-8)
        outputs = (outputs * observed_mask_pooled).sum(dim=1) / n_nonmask
        outputs = self.ln_proj(outputs.view(B, D, -1))
        
        # 4. PLM 2 (变量感知)
        outputs = outputs + var_embedding.squeeze(1)
        var_position_ids = torch.arange(D, device=self.device).unsqueeze(0).expand(B, D)
        var_attn_mask = torch.ones(B, D, device=self.device, dtype=torch.long)
        
        outputs = self.gpts[1](
            inputs_embeds=outputs,
            attention_mask=var_attn_mask,
            position_ids=var_position_ids # <-- 传递离散索引
        ).last_hidden_state # (B, D, d_model)

        # 5. 预测头
        outputs = outputs.reshape(B, -1) # (B, D * d_model)
        outputs = self.forecasting_head(outputs) # (B, pred_len * D)
        outputs = outputs.reshape(B, self.pred_len, -1) # (B, pred_len, D)
        
        # 添加一个维度以兼容评估脚本
        return outputs.unsqueeze(0) # (1, B, pred_len, D)
