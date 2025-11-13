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
from transformers.cache_utils import Cache
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
    def __init__(self, d_var, d_model, max_len=5000):
        super(VariableEmbedding_Qwen, self).__init__()
        # 使用与 Qwen 位置编码兼容的 embedding
        self.var_embedding = torch.nn.Embedding(d_var, d_model)
    
    def forward(self, var_indices):
        # var_indices: (B, D)
        # return: (B, D, d_model)
        return self.var_embedding(var_indices)

# ==================================================================================
# 第 3 部分：CT-RoPE (来自 MIRA)
# ==================================================================================

class CTRoPE(nn.Module):
    """连续时间旋转位置编码 (CT-RoPE)"""
    def __init__(self, dim, ratio=1.0):
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x, seq_len, timestamps):
        # x: (B, H, L, D)
        # timestamps: (B, L)
        
        self.inv_freq = self.inv_freq.to(x.device)
        
        # (B, L) -> (B, L, D/2)
        t = timestamps.unsqueeze(-1) * self.inv_freq
        
        # (B, L, D)
        freqs = torch.cat((t, t), dim=-1)
        
        # (B, 1, L, D)
        freqs = freqs.unsqueeze(1)
        
        # (B, 1, L, D) -> (B, 1, L, D)
        cos_cached = freqs.cos()
        sin_cached = freqs.sin()
        
        # Qwen-style 旋转
        # x: (B, H, L, D)
        # x_embed: (B, H, L, D/2, 2)
        x_embed = x.float().reshape(x.shape[:-1] + (self.dim // 2, 2))
        
        # (B, H, L, D/2, 2)
        x_cos = x_embed[..., 0]
        x_sin = x_embed[..., 1]
        
        # (B, 1, L, D) -> (B, 1, L, D/2, 2)
        freqs_cos_cached = cos_cached.reshape(cos_cached.shape[:-1] + (self.dim // 2, 2))
        freqs_sin_cached = sin_cached.reshape(sin_cached.shape[:-1] + (self.dim // 2, 2))
        
        # (B, 1, L, D/2, 2)
        cos = freqs_cos_cached[..., 0]
        sin = freqs_sin_cached[..., 0]
        
        # 应用旋转
        # (B, H, L, D/2)
        x_rot_cos = x_cos * cos - x_sin * sin
        x_rot_sin = x_cos * sin + x_sin * cos
        
        # (B, H, L, D/2, 2)
        x_rotated = torch.stack((x_rot_cos, x_rot_sin), dim=-1)
        
        # (B, H, L, D)
        x_rotated = x_rotated.reshape(x.shape)
        
        return x_rotated.type_as(x)

# ==================================================================================
# 第 4 部分：注入 CT-RoPE 的 Qwen2 模块
# ==================================================================================

class CTRoPEQwen2Attention(Qwen2Attention):
    """
    Qwen2Attention 的子类，用 CT-RoPE 替换标准的 RoPE。
    """
    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: Optional[int] = None
    ):
        # 调用父类 (Qwen2Attention) 的 __init__
        super().__init__(config, layer_idx)

        # 移除标准的 RoPE (如果它在父类中被定义为 nn.Module)
        # Qwen2 的 RoPE 是在 forward 中即时计算的，所以我们只需要替换
        # apply_rotary_pos_emb 调用
        
        self.rope_ratio = 1.0 # MIRA 默认 ratio
        
        # 实例化我们自己的 CT-RoPE 模块
        self.ct_rope_emb = CTRoPE(
            self.head_dim, 
            ratio=self.rope_ratio
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestamps: torch.Tensor, # <-- CT-RoPE 需要的额外参数
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # <-- 这个将被忽略
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        # Q/K/V 投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # (B, L, H*D) -> (B, L, H, D) -> (B, H, L, D)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        # [MODIFIED] 应用 CT-RoPE
        # 我们需要 'timestamps'，它应该通过 Qwen2DecoderLayerWithCTRoPE 传递下来
        # 注意：不再使用 position_ids
        
        # (B, H, L, D) , (B, L) -> (B, H, L, D)
        query_states = self.ct_rope_emb(query_states, seq_len=q_len, timestamps=timestamps)
        key_states = self.ct_rope_emb(key_states, seq_len=q_len, timestamps=timestamps)

        # K/V 缓存
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # Qwen2 GQA (重复 K/V)
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # MHA (多头注意力)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            # [FIX 2 - RuntimeError]
            # 修复了 RuntimeError：为 use_cache=True (解码/预测) 添加掩码切片逻辑
            # cache_position 在 use_cache=True 的生成期间被传递
            if cache_position is not None:
                # 我们处于解码模式，需要对掩码进行切片
                # cache_position[0] 提供了 *新* 标记的起始索引
                mask_slicing_from = cache_position[0]
                mask_slicing_to = cache_position[0] + q_len
                
                # 在查询维度 (dim 2) 上对掩码进行切片
                # 并确保键维度 (dim 3) 与 (可能已缓存的) key_states 匹配
                sliced_attention_mask = attention_mask[:, :, mask_slicing_from:mask_slicing_to, :kv_seq_len]
                
                # 应用切片后的掩码
                attn_weights = attn_weights + sliced_attention_mask
            else:
                # 这是训练模式或提示处理 (cache_position 为 None)
                # 掩码和 q_len 应该匹配
                attn_weights = attn_weights + attention_mask
            # [FIX 2 END]

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2DecoderLayerWithCTRoPE(Qwen2DecoderLayer):
    """
    Qwen2DecoderLayer 的子类，以确保 'timestamps' 参数
    可以被传递给 CTRoPEQwen2Attention 模块。
    """
    def __init__(self, config: Qwen2Config, layer_idx: int):
        # 正常初始化 Qwen2DecoderLayer
        super().__init__(config, layer_idx)
        
        # 这里的 self.self_attn 仍然是 *原始的* Qwen2Attention
        # 它将在 inject_CTRoPE_to_model 中被 *替换*
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestamps: torch.Tensor, # <-- CT-RoPE 需要的额外参数
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # <-- 将被忽略
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力 (Self Attention)
        attn_outputs, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            timestamps=timestamps, # <-- 将 timestamps 传递给 CT-RoPE Attention
            attention_mask=attention_mask,
            position_ids=position_ids, # <-- 传递，但 CT-RoPE 会忽略它
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        
        # [FIX 1 - SyntaxError]
        # 修复了 SyntaxError：删除了下面这行无效的代码
        # past_key_value: Optional] = None, 
        # [FIX 1 END]

        hidden_states = residual + attn_outputs

        # 全连接层 (MLP)
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

# ==================================================================================
# 第 5 部分：PLM4TS 基础模型 (现在使用 Qwen + CT-RoPE + LoRA)
# ==================================================================================

class PLM4TS_Base(Qwen2Model):
    """
    修改 Qwen2Model (继承自 Qwen2PreTrainedModel) 以：
    1. 接受 'timestamps' (时间戳) 作为 forward 参数。
    2. 将 'timestamps' 传递给每个 Qwen2DecoderLayerWithCTRoPE。
    3. (可选) 修改 'inputs_embeds' 的构建方式。
    """
    def __init__(self, config: Qwen2Config):
        # 以 Qwen2Model 的方式初始化
        super(Qwen2Model, self).__init__(config) # 调用 Qwen2PreTrainedModel
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # [MODIFIED] 使用我们的 Qwen2DecoderLayerWithCTRoPE 替换
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayerWithCTRoPE(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        timestamps: torch.Tensor = None, # <-- CT-RoPE 需要的额外参数
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # <-- 将被忽略
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # [MODIFIED] 确保我们可以处理 timestamps
        if timestamps is None:
            # 如果不提供 timestamps，我们必须
            # 1. 检查是否提供了 position_ids (用于标准 RoPE)
            # 2. 或者报错
            # 为了简单起见，我们假设 CT-RoPE 模块 *总是* 需要 timestamps
            raise ValueError("CT-RoPE enabled model requires 'timestamps' to be passed in the forward method.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # [STANDARD] Qwen2Model.forward() 的标准设置
        
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        # [MODIFIED] 获取 cache_position (为掩码切片做准备)
        cache_position = self._get_cache_position(attention_mask, inputs_embeds.shape[1])
        
        if attention_mask is not None:
             # [MODIFIED] 准备注意力掩码 (Qwen2的标准流程)
             attention_mask = self._prepare_causal_attention_mask(
                attention_mask,
                inputs_embeds.shape[1],
                inputs_embeds.shape[0],
                cache_position,
                past_key_values[0].get_usable_length(inputs_embeds.shape[1], 0) if past_key_values[0] else 0
            )

        hidden_states = inputs_embeds
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # [MODIFIED] 循环遍历 self.layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx]

            if self.gradient_checkpointing and self.training:
                # ... (省略 checkpointing 逻辑)
                pass
            
            # [MODIFIED] 将 'timestamps' 和 'cache_position' 传递给 decoder_layer
            layer_outputs = decoder_layer(
                hidden_states,
                timestamps=timestamps, # <-- 在这里传递
                attention_mask=attention_mask,
                position_ids=position_ids, # <-- 传递，但会被忽略
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position, # <-- 在这里传递
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


def inject_CTRoPE_to_model(model: Qwen2Model):
    """
    辅助函数：遍历 Qwen2Model (或 PLM4TS_Base)
    并将 CTRoPEQwen2Attention 注入到每个 Qwen2DecoderLayerWithCTRoPE 中。
    """
    for i, layer in enumerate(model.layers):
        if not isinstance(layer, Qwen2DecoderLayerWithCTRoPE):
            raise TypeError(f"Layer {i} is not Qwen2DecoderLayerWithCTRoPE. Model setup is incorrect.")

        # 1. 创建新的 CT-RoPE 注意力模块
        new_attn = CTRoPEQwen2Attention(
            model.config,
            layer_idx=i,
        )
        
        # 2. [关键] 将权重从 *旧的* (原始的) 注意力模块复制过来
        # 假设 LoRA 尚未应用
        old_attn_module = layer.self_attn
        
        new_attn.q_proj.weight = old_attn_module.q_proj.weight
        new_attn.k_proj.weight = old_attn_module.k_proj.weight
        new_attn.v_proj.weight = old_attn_module.v_proj.weight
        new_attn.o_proj.weight = old_attn_module.o_proj.weight
        
        if old_attn_module.q_proj.bias is not None:
            new_attn.q_proj.bias = old_attn_module.q_proj.bias
            new_attn.k_proj.bias = old_attn_module.k_proj.bias
            new_attn.v_proj.bias = old_attn_module.v_proj.bias
            new_attn.o_proj.bias = old_attn_module.o_proj.bias

        # 3. 用新模块覆盖旧模块
        model.layers[i].self_attn = new_attn
        
    return model

# ==================================================================================
# 第 6 部分：模型集成 (主 ISTS-PLM 模型)
# ==================================================================================

# [FIX 3 - NameError] 重命名类 (原为 ISTS_PLM)
class istsplm_forecast(nn.Module):
    def __init__(self, args):
        super(istsplm_forecast, self).__init__() # [FIX 3] 匹配类名
        self.args = args
        self.device = args.device

        # 1. 加载 PLM (Qwen2)
        # [FIX 4 - ArgError] 使用 plm_path (原为 args.plm)
        self.plm_name = args.plm_path 
        self.config = AutoConfig.from_pretrained(self.plm_name, trust_remote_code=True)
        
        # [MODIFIED] 设置我们的 PLM4TS_Base 作为基础
        self.config.use_cache = True 
        
        # [FIX 6 - AttributeError]
        # 修复了 AttributeError: 'PLM4TS_Base' object has no attribute 'load_weights'
        # 我们必须使用 .from_pretrained() 来加载权重，然后再注入 CT-RoPE。
        
        # [FIX 4 - ArgError] 硬编码 n_plm_layers=2 (一个用于时间，一个用于变量)
        gpts = []
        for _ in range(2):
            # 2a. 使用 .from_pretrained() 加载预训练模型
            # 这会使用我们的 PLM4TS_Base 类，并加载预训练权重
            model = PLM4TS_Base.from_pretrained(
                self.plm_name,
                config=self.config,
                trust_remote_code=True # 确保 Qwen 代码可以运行
            )
            
            # 2b. 注入 CT-RoPE
            # 这会用 CTRoPEQwen2Attention 替换 Qwen2Attention
            # 并将 .from_pretrained() 加载的权重复制过去
            model = inject_CTRoPE_to_model(model)
            
            gpts.append(model)
        
        # 2. 加载预训练权重 (在注入 CT-RoPE 之后)
        # for gpt in gpts:
        #     gpt.load_weights(self.plm_name) # <-- [FIX 6] 移除此错误行
        
        # 3. (可选) 应用 LoRA
        if args.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, # 尽管我们用于特征提取，但这通常是必需的
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 针对 Qwen2 的标准目标
            )
            gpts = [get_peft_model(gpt, peft_config) for gpt in gpts]
            for gpt in gpts:
                gpt.print_trainable_parameters()

        self.gpts = nn.ModuleList(gpts)
        self.gpts.to(self.device)

        # 4. ISTS-PLM 的特定层
        self.d_model = self.config.hidden_size
        self.patch_len = args.patch_len
        self.prompt_len = args.prompt_len
        
        # Embedding 层
        self.value_embedding = ValueEmbedding_Qwen(self.patch_len, self.d_model)
        # [FIX 4 - ArgError] 使用 num_types (原为 args.n_vars)
        self.variable_embedding = VariableEmbedding_Qwen(args.num_types, self.d_model)
        
        # 投影层
        self.ln_proj = nn.LayerNorm(self.d_model)
        self.out_projection = nn.Linear(self.d_model, args.pred_len)

    def forecasting(self, batch_dict, n_vars_to_predict):
        # 1. 准备输入
        x = batch_dict["x"].to(self.device)                 # (B, L_total, D)
        t = batch_dict["t"].to(self.device)                 # (B, L_total)
        t_pos_idx = batch_dict["t_pos_idx"].to(self.device) # (B, L_total) [离散索引]
        var_idx = batch_dict["var_idx"].to(self.device)     # (B, D)
        observed_mask = batch_dict["observed_mask"].to(self.device) # (B, L, D)

        B, L, D = observed_mask.shape
        L_total = x.shape[1] # L + prompt_len

        # 2. PLM 1 (时间感知)
        
        # 假设 `x` 已经是 `inputs_embeds` (B, D, L_p, d_model)
        # `t` 已经是 (B, D, L_p)
        # `t_pos_idx` (B, D, L_p)
        
        # 假设 inputs_embeds 和 timestamps 已经准备好
        inputs_embeds = x # (B, D, L_total, d_model)
        timestamps = t    # (B, D, L_total)
        position_ids = t_pos_idx # (B, D, L_total)
        
        # (B, D, L_total, d_model) -> (B*D, L_total, d_model)
        inputs_embeds = inputs_embeds.view(-1, L_total, self.d_model)
        # (B, D, L_total) -> (B*D, L_total)
        tt_with_prompt = timestamps.view(-1, L_total)
        pos_with_prompt = position_ids.view(-1, L_total)

        # 准备变量 Embedding (B, D, d_model) -> (B, D, 1, d_model)
        var_embedding = self.variable_embedding(var_idx).unsqueeze(2)

        # (B*D, L_total, d_model) + (B*D, 1, d_model)
        inputs_embeds = inputs_embeds + var_embedding.view(-1, 1, self.d_model)
        
        # 准备掩码
        time_attn_mask_data = observed_mask.permute(0, 2, 1) # (B, D, L)
        
        # (B, D, L) -> (B*D, L_total)
        time_attn_mask_data_flat = torch.cat([
            torch.ones(B*D, self.prompt_len, device=self.device, dtype=torch.long), 
            time_attn_mask_data.reshape(B*D, -1)
        ], dim=1)
        
        # (B*D, L_total) -> (B*D, 1, L_total, L_total)
        # 注意: Qwen2 的 _prepare_causal_attention_mask 是在 PLM4TS_Base.forward 中调用的
        # 我们在这里只需要 (B*D, L_total)
        time_attn_mask = time_attn_mask_data_flat
        
        outputs = self.gpts[0](
            inputs_embeds=inputs_embeds,
            attention_mask=time_attn_mask,
            timestamps=tt_with_prompt # <-- 传递连续时间
            # position_ids=pos_with_prompt # <-- CT-RoPE 不需要这个
        ).last_hidden_state

        # 3. Masked Pooling (掩码池化)
        observed_mask_pooled = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1)
        # 确保 prompt 部分被包含在池化中
        observed_mask_pooled_with_prompt = torch.cat([
            torch.ones(B*D, self.prompt_len, 1, device=self.device, dtype=observed_mask_pooled.dtype), 
            observed_mask_pooled
        ], dim=1)
        
        n_nonmask = (observed_mask_pooled_with_prompt.sum(dim=1) + 1e-8)
        outputs = (outputs * observed_mask_pooled_with_prompt).sum(dim=1) / n_nonmask
        outputs = self.ln_proj(outputs.view(B, D, -1))
        
        # 4. PLM 2 (变量感知)
        outputs = outputs + var_embedding.squeeze(1)
        
        # (B, D)
        var_attn_mask = torch.ones(B, D, device=self.device, dtype=torch.long)
        
        # [FIX 5 - 新的Bug修复] 
        # 第二个 PLM (gpts[1]) 也是一个 CT-RoPE 模型，它需要 'timestamps' 而不是 'position_ids'
        # 我们为 D 个变量创建一个简单的连续时间戳 [0, 1, ..., D-1]
        var_timestamps = torch.arange(D, device=self.device).float().unsqueeze(0).expand(B, D)

        outputs = self.gpts[1](
            inputs_embeds=outputs,
            attention_mask=var_attn_mask,
            timestamps=var_timestamps # <-- [FIX 5] 传递连续时间
            # position_ids=var_position_ids # <-- 不再使用这个
        ).last_hidden_state

        # 5. 最终投影
        # (B, D, d_model) -> (B, D, pred_len)
        forecast = self.out_projection(outputs)
        
        # (B, D, pred_len) -> (B, pred_len, D)
        forecast = forecast.permute(0, 2, 1)

        # 返回需要预测的变量
        if n_vars_to_predict is not None:
            forecast = forecast[:, :, :n_vars_to_predict]

        return forecast

    def forward(self, batch_dict, n_vars_to_predict=None):
        return self.forecasting(batch_dict, n_vars_to_predict)
