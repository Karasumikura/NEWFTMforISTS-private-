import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List

# ==================================================================================
# 第 1 部分：新模型 (Qwen / LoRA) 的依赖项
# ==================================================================================
from transformers import (
    AutoConfig, 
    PreTrainedModel, 
    Qwen2Model, 
    Qwen2Config, 
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, 
    Qwen2DecoderLayer, 
    Qwen2Model, 
    Qwen2PreTrainedModel,
    apply_rotary_pos_emb,
    Qwen2MLP,
    Qwen2RMSNorm
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
        # c_in 在这里是 1，因为每个变量的值都被单独处理
        self.projection = nn.Linear(c_in, d_model) 
    def forward(self, x):
        return self.projection(x)


class VariableEmbedding_Qwen(nn.Module):
    """为 D 个时间序列变量创建 embedding"""
    def __init__(self, c_in, d_model):
        super(VariableEmbedding_Qwen, self).__init__()
        # c_in: 变量维度 (D)
        # d_model: 隐藏层维度 (H)
        self.var_emb = nn.Embedding(c_in, d_model)
        self.c_in = c_in

    def forward(self, x):
        # x: (B, L, D) -> (B, L, D) 变量索引
        B, L, D = x.shape
        var_indices = torch.arange(D, device=x.device, dtype=torch.long).unsqueeze(0).unsqueeze(0).expand(B, L, D)
        return self.var_emb(var_indices) # (B, L, D, H)


# ==================================================================================
# 第 3 部分：CT-RoPE (Continuous Time Rotary Position Embedding) 模块
# ==================================================================================

# 沿用了 Qwen2Attention 的结构，但修改了 RoPE 的应用方式
class CTRoPEQwen2Attention(Qwen2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        timestamps: Optional[torch.Tensor] = None, # <-- 接收连续时间戳 (B, L)
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # CT-RoPE 应用: 使用连续时间戳 (timestamps) 而不是 position_ids
        if timestamps is not None:
            # timestamps: (B, L)
            # 使用 timestamps 作为位置索引进行旋转
            
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            
            # 扩展 timestamps 以匹配 RoPE 接口所需的形状 (B, 1, L, 1)
            timestamps_exp = timestamps.unsqueeze(1).unsqueeze(-1).to(key_states.dtype)
            
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, timestamps_exp
            )
        elif position_ids is not None:
            # Fallback to standard RoPE
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # [FIX 2: RuntimeError] 修复 use_cache=True (预测阶段) 时的 attention mask 切片问题
        if attention_mask is not None:
            if cache_position is not None:
                # attention_mask shape is typically (B, 1, L_kv, L_kv). 
                # We need to slice the L_kv rows corresponding to q_len (q_len is small in generation).
                attention_mask_kv_len = attention_mask.shape[-2]
                if q_len != attention_mask_kv_len:
                    # 获取需要保留的行的起始索引
                    start_row = attention_mask_kv_len - q_len
                    # 仅保留 attention_mask 中对应 q_len 的行
                    attention_mask = attention_mask[:, :, start_row:, :]

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is of size"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# 沿用了 Qwen2DecoderLayer 的结构，但替换了 Attention 模块
class Qwen2DecoderLayerWithCTRoPE(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 用我们的 CTRoPE 版本替换标准 Qwen2Attention
        self.self_attn = CTRoPEQwen2Attention(config, layer_idx)


    # [FIX 1: SyntaxError] 修复了 Qwen2DecoderLayerWithCTRoPE.forward 中的语法错误
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        timestamps: Optional[torch.Tensor] = None, # <-- 接收连续时间戳
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states

        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        
        # 将 timestamps 传递给自定义的 Attention 模块
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            timestamps=timestamps, # <-- 传递 timestamps
            cache_position=cache_position,
        )

        hidden_states = residual + attn_outputs

        # MLP
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


# 沿用了 Qwen2Model 的结构，但替换了 DecoderLayer
class PLM4TS_Base(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        # 替换所有的 Qwen2DecoderLayer 为我们的 CTRoPE 版本
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayerWithCTRoPE(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.embed_tokens = None # 禁用原始的 token embedding
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # <-- 使用这个作为主要输入
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        timestamps: Optional[torch.Tensor] = None, # <-- 接收连续时间戳
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            raise ValueError("PLM4TS_Base expects `inputs_embeds` to be passed, not `input_ids`.")

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
                timestamps=timestamps, # <-- 传递连续时间戳
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
    """递归地替换模型中的 Qwen2Attention 为 CTRoPEQwen2Attention 并安全地复制权重"""
    
    # 遍历所有子模块
    for name, module in model.named_children():
        
        # 如果找到 Qwen2Attention，则替换它
        if isinstance(module, Qwen2Attention):
            config = module.config
            layer_idx = module.layer_idx
            
            # 创建新的 CTRoPE Attention 模块
            new_attn = CTRoPEQwen2Attention(config, layer_idx)
            
            # 从旧模块复制权重到新模块
            new_attn.q_proj.weight.data.copy_(module.q_proj.weight.data)
            new_attn.k_proj.weight.data.copy_(module.k_proj.weight.data)
            new_attn.v_proj.weight.data.copy_(module.v_proj.weight.data)
            new_attn.o_proj.weight.data.copy_(module.o_proj.weight.data)
            
            # [FIX 7: NoneType bias] 修复了 bias 复制错误
            # 必须为每个 proj 单独检查 bias 是否存在
            if module.q_proj.bias is not None:
                new_attn.q_proj.bias.data.copy_(module.q_proj.bias.data)
            if module.k_proj.bias is not None:
                new_attn.k_proj.bias.data.copy_(module.k_proj.bias.data)
            if module.v_proj.bias is not None:
                new_attn.v_proj.bias.data.copy_(module.v_proj.bias.data)
            if module.o_proj.bias is not None:
                new_attn.o_proj.bias.data.copy_(module.o_proj.bias.data)
            
            # 替换模块
            setattr(model, name, new_attn)
            
        # 递归地检查子模块
        else:
            inject_CTRoPE_to_model(module)
            
    return model


def configure_lora(args, model: PLM4TS_Base):
    """为模型配置和应用 LoRA"""
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 应用 LoRA
    lora_model = get_peft_model(model, lora_config)
    
    # 冻结其他参数
    for name, param in lora_model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
            
    return lora_model


# ==================================================================================
# 第 5 部分：ISTS_PLM 主模型 (istsplm_forecast)
# ==================================================================================

class istsplm_forecast(nn.Module):
    # [FIX 3] 匹配 regression.py 中的类名
    def __init__(self, args):
        super(istsplm_forecast, self).__init__() 

        self.args = args
        self.device = args.device
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.use_lora = args.use_lora
        
        # [FIX 4 - ArgError] 修复参数不匹配
        self.plm_name = args.plm_path 
        self.num_types = args.num_types # D 维度 (变量数)
        self.n_plm_layers = 2 # 明确指定两个PLM：一个用于时间，一个用于变量

        # 1. 配置
        self.config = AutoConfig.from_pretrained(self.plm_name, trust_remote_code=True)
        self.config.use_cache = True 
        # 在这里不覆盖 num_hidden_layers，因为我们只加载两个模型实例，但它们会继承原始配置
        
        # 2. Embedding 层
        # Value Embedding (将 1 维值投影到 d_model)
        self.value_embedding = ValueEmbedding_Qwen(1, self.d_model)
        # Variable Embedding (为 D 个变量创建 embedding)
        self.var_embedding = nn.Embedding(self.num_types, self.d_model)

        # 3. PLM 模块 (两个独立的 PLM 实例)
        gpts = []
        for i in range(self.n_plm_layers):
            # 3a. 使用 .from_pretrained() 加载预训练模型 (FIX 6: load_weights)
            model = PLM4TS_Base.from_pretrained(
                self.plm_name,
                config=self.config,
                ignore_mismatched_sizes=True,
                trust_remote_code=True
            )
            
            # 3b. 注入 CT-RoPE
            model = inject_CTRoPE_to_model(model)
            
            # 3c. 应用 LoRA
            if args.use_lora:
                model = configure_lora(args, model)
                
            gpts.append(model)
        
        self.gpts = nn.ModuleList(gpts) # gpts[0] 是时间PLM，gpts[1] 是变量PLM
        
        # 4. Projection Layer (池化后的输出投影层)
        self.ln_proj = nn.LayerNorm(self.d_model)
        
        # 5. Forecasting Head (预测头)
        self.forecasting_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LeakyReLU(),
            nn.Linear(self.d_model // 2, self.num_types) # 输出维度 D
        )
        
        # 6. Prompt/Patch Embedding 
        self.patch_len = args.patch_len
        self.prompt_len = args.prompt_len
        # (1, L_prompt, d_model)
        self.prompt_embed = nn.Parameter(torch.randn(1, self.prompt_len, self.d_model))
        

    def forecasting(self, batch_dict, n_vars_to_predict=None):
        # [FIX 5: KeyError & FIX 7: evaluation.py 接口]
        # 现在接收 batch_dict 和 n_vars_to_predict
        
        observed_data = batch_dict["observed_data"].to(self.device) # (B, L_obs, D)
        observed_mask = batch_dict["observed_mask"].to(self.device) # (B, L_obs, D)
        observed_tp = batch_dict["observed_tp"].to(self.device)     # (B, L_obs)
        tp_to_predict = batch_dict["tp_to_predict"].to(self.device) # (B, L_pred)

        B, L_obs, D = observed_data.shape
        L_pred = tp_to_predict.shape[1]

        # ----------------------------------------------------
        # 步骤 1: 准备输入 (时间 PLM)
        # ----------------------------------------------------
        
        # 1a. Value Embedding
        # 将 (B, L_obs, D) 转换为 (B*D, L_obs, 1) 以进行时间 PLM (每个变量独立)
        x_flat = observed_data.permute(0, 2, 1).reshape(B * D, L_obs, 1) 
        # 投影到隐藏层维度 (B*D, L_obs, d_model)
        value_embed = self.value_embedding(x_flat) 

        # 1b. Times Embedding (CT-RoPE)
        # 复制 observed_tp，使其与 (B*D) 批次大小匹配
        tt = observed_tp.unsqueeze(1).expand(B, D, L_obs).reshape(B * D, L_obs)
        
        # 1c. 添加 Prompt/Prefix
        prompt_embed = self.prompt_embed.expand(B * D, -1, -1)
        inputs_embeds = torch.cat([prompt_embed, value_embed], dim=1) # (B*D, L_prompt + L_obs, d_model)
        
        # 拼接时间戳：Prompt 的时间戳为 0
        tt_prompt = torch.zeros(B * D, self.prompt_len, device=self.device)
        tt_with_prompt = torch.cat([tt_prompt, tt], dim=1) # (B*D, L_prompt + L_obs)
        
        # 1d. Attention Mask (忽略 Prompt)
        time_attn_mask_data_flat = observed_mask[:, :, 0].flatten().unsqueeze(1) # (B*D, L_obs)
        time_attn_mask_prompt = torch.ones(B * D, self.prompt_len, device=self.device) # Prompt 完全可见
        time_attn_mask = torch.cat([time_attn_mask_prompt, time_attn_mask_data_flat], dim=1)
        # 转换为 Qwen 所需的格式 (B, 1, 1, L_total)
        # Qwen attention mask 处理复杂，这里我们使用 (B, 1, L_total, L_total) 的对角下三角掩码
        # 为了简单，我们只使用 padding mask (B, 1, 1, L_total)
        # 注意：Qwen内部会自动生成 causal mask
        time_attn_mask = (1.0 - time_attn_mask) * torch.finfo(torch.float32).min # (B*D, L_total) -> 转换为 additive mask
        time_attn_mask = time_attn_mask.unsqueeze(1).unsqueeze(1) # (B*D, 1, 1, L_total)
        time_attn_mask = time_attn_mask.to(self.device)

        # ----------------------------------------------------
        # 步骤 2: PLM 1 (时间感知)
        # ----------------------------------------------------
        
        outputs = self.gpts[0](
            inputs_embeds=inputs_embeds,
            attention_mask=time_attn_mask,
            timestamps=tt_with_prompt 
        ).last_hidden_state

        # ----------------------------------------------------
        # 步骤 3: Masked Pooling (掩码池化)
        # ----------------------------------------------------
        
        # 重新引入 observed_mask
        observed_mask_pooled = observed_mask.permute(0, 2, 1).reshape(B * D, L_obs, 1) # (B*D, L_obs, 1)
        
        # outputs shape: (B*D, L_prompt + L_obs, d_model)
        # 排除 Prompt 后的输出
        outputs_data = outputs[:, self.prompt_len:] # (B*D, L_obs, d_model)
        
        # 应用掩码和求平均（跳过 Prompt 部分）
        n_nonmask = (observed_mask_pooled.sum(dim=1) + 1e-8) # (B*D, 1)
        outputs = (outputs_data * observed_mask_pooled).sum(dim=1) / n_nonmask # (B*D, d_model)
        
        # 投影和重塑回 (B, D, d_model)
        outputs = self.ln_proj(outputs).view(B, D, -1) # (B, D, d_model)
        
        # ----------------------------------------------------
        # 步骤 4: PLM 2 (变量感知)
        # ----------------------------------------------------

        # 4a. Variable Embedding 
        var_embedding = self.var_embedding.weight.unsqueeze(0).expand(B, D, -1)
        outputs = outputs + var_embedding
        
        # 4b. Position IDs (离散位置索引)
        var_position_ids = torch.arange(D, device=self.device).unsqueeze(0).expand(B, D)
        
        # 4c. Variable Attention Mask (全连接)
        # (B, 1, D, D)
        var_attn_mask = torch.zeros(B, D, D, device=self.device, dtype=torch.float32)
        var_attn_mask = var_attn_mask.masked_fill(var_attn_mask == 0, 0.0) # 确保无负无穷
        var_attn_mask = var_attn_mask.unsqueeze(1) # (B, 1, D, D)

        # 4d. PLM 2 执行 (不传递 timestamps)
        outputs = self.gpts[1](
            inputs_embeds=outputs,
            attention_mask=var_attn_mask,
            position_ids=var_position_ids 
        ).last_hidden_state # (B, D, d_model)
        
        # ----------------------------------------------------
        # 步骤 5: 预测
        # ----------------------------------------------------
        
        # 预测头返回 D 维度的变量值 (B, D, D)
        forecast_vars = self.forecasting_head(outputs) 

        # 假设预测值是 D 维度中的第一个 (或平均值)
        # 这里使用第一个维度作为回归预测值
        regression_value = forecast_vars[:, :, 0] # (B, D)
        
        # 扩展到 L_pred 时间步 (ISTS-PLM 原有策略)
        forecast = regression_value.unsqueeze(1).expand(-1, L_pred, -1) # (B, L_pred, D)

        # 如果需要预测的变量维度少于 D
        if n_vars_to_predict is not None:
            forecast = forecast[:, :, :n_vars_to_predict]

        return forecast


    def forward(self, batch_dict, n_vars_to_predict=None):
        # 允许 forward 和 forecasting 使用相同的签名
        return self.forecasting(batch_dict, n_vars_to_predict)
