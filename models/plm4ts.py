"""
plm4ts.py - 完整实现

该文件包含将 QWEN 迁移到使用连续时间旋转位置编码 (CT-RoPE)
所需的自定义 PyTorch 模块，用于不规则采样时间序列 (ISTS) [1, 2] 分析。

包含的模块：
1. ContinuousTimeRotaryEmbedding: 实现 MIRA 论文中的 CT-RoPE [7, 8]。
2. CustomQwen2Attention: 继承自 Qwen2Attention，但：
    a) 移除了离散 RoPE，替换为 CT-RoPE。
    b) 修复了在自回归解码中因果掩码的维度不匹配错误 (RuntimeError) [6, 9]。
3. CustomQwen2DecoderLayer: 继承自 Qwen2DecoderLayer，以使用 CustomQwen2Attention
   并传递 'timestamps' 参数。
4. CustomQwen2Model: 继承自 Qwen2Model，以使用 CustomQwen2DecoderLayer。
5. CustomQwen2ForCausalLM: 继承自 Qwen2ForCausalLM，以使用 CustomQwen2Model
   并处理 'timestamps' 在.forward() 和.generate() 中的传递。

LoRA 兼容性 [10]：
所有线性层（q_proj, k_proj, v_proj, o_proj, mlp）的名称
均被保留，以确保 PEFT LoRA [11] 适配器可以正确附加。
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Union

# 从 transformers 导入 QWEN 的基础模块
try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention, 
        Qwen2DecoderLayer,
        Qwen2Model,
        Qwen2ForCausalLM
    )
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.modeling_outputs import (
        BaseModelOutputWithPast,
        CausalLMOutputWithPast
    )
    from torch.nn import CrossEntropyLoss
    import torch.utils.checkpoint
    from transformers.utils import logging
    
    logger = logging.get_logger(__name__)

except ImportError:
    print("="*80)
    print("错误：无法导入 'transformers' 库或 'Qwen2' 模型。")
    print("请确保已安装 transformers 库：pip install transformers")
    print("="*80)
    raise

# --- 1. CT-RoPE 辅助函数 ---

def apply_ct_rotation(
    x: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> torch.Tensor:
    """
    应用 CT-RoPE 旋转。
    这是高效的 "rotate_half" 实现 [7]。
    
    Args:
        x (torch.Tensor): 形状为 [b, h, s, d_h]
        cos (torch.Tensor): 形状为 [b, 1, s, d_h/2]
        sin (torch.Tensor): 形状为 [b, 1, s, d_h/2]
    """
    # 将 x 分为两半 (x1, x2)
    # x1/x2 形状: [b, h, s, d_h/2]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    
    # 扩展 cos 和 sin 以匹配 (x1, x2) 的维度
    # [b, 1, s, d_h/2] -> [b, 1, s, d_h]
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    
    # 高效旋转:
    # x_rotated = (x * cos) + (rotate_half(x) * sin)
    # rotate_half(x) = torch.cat((-x2, x1), dim=-1)
    
    x_rotated = torch.cat(
        (-x2, x1), dim=-1
    ) * sin + torch.cat(
        (x1, x2), dim=-1
    ) * cos
    
    return x_rotated.to(x.dtype)

# --- 2. CT-RoPE 模块 ---

class ContinuousTimeRotaryEmbedding(nn.Module):
    """
    连续时间旋转位置编码 (CT-RoPE) 的实现
    基于 MIRA 论文 (arxiv.org/pdf/2506.07584 [7, 8])。
    
    该模块根据连续的浮点时间戳动态计算旋转嵌入,
    以取代标准QWEN模型中基于离散索引的RoPE。
    """
    def __init__(self, dim: int, base: float = 10000.0, device: Optional[str] = None):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # 计算逆频率 (即 MIRA 论文 [7] 中的 \omega_i)
        # 形状: [dim / 2]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        timestamps_q: torch.Tensor, 
        timestamps_k: torch.Tensor
    ) -> Tuple:
        """
        将 CT-RoPE 应用于查询 (Q) 和键 (K) 张量。
        
        Args:
            q (torch.Tensor): 查询张量, 形状 [b, h, q_len, d_h]
            k (torch.Tensor): 键张量, 形状 [b, h, k_len, d_h]
            timestamps_q (torch.Tensor): 查询的时间戳, 形状 [b, q_len]
            timestamps_k (torch.Tensor): 键的时间戳, 形状 [b, k_len]
            
        Returns:
            Tuple: 旋转后的 q 和 k 张量。
        """
        
        # 1. 计算时间相关的角度 (theta_i(t) = omega_i * t) [7]
        #    timestamps_q: [b, q_len] -> [b, q_len, 1]
        #    self.inv_freq: [d/2] -> [1, 1, d/2]
        #    angles_q: [b, q_len, d/2]
        angles_q = torch.einsum(
            "bi,d->bid", timestamps_q.float(), self.inv_freq
        ).to(q.device)
        
        #    angles_k: [b, k_len, d/2]
        angles_k = torch.einsum(
            "bi,d->bid", timestamps_k.float(), self.inv_freq
        ).to(k.device)

        # 2. 从角度计算 cos 和 sin
        #    重塑形状以便广播到多头注意力:
        #    [b, q_len, d/2] -> [b, q_len, 1, d/2]
        cos_q = angles_q.cos()[:, :, None, :]
        sin_q = angles_q.sin()[:, :, None, :]
        
        #    [b, k_len, d/2] -> [b, k_len, 1, d/2]
        cos_k = angles_k.cos()[:, :, None, :]
        sin_k = angles_k.sin()[:, :, None, :]
        
        # 3. 应用旋转
        #    q/k 形状为 [b, h, s, d_h], 
        #    cos/sin 形状为 [b, s, 1, d_h/2], 需要 transpose -> [b, 1, s, d_h/2]
        q_rotated = apply_ct_rotation(
            q, cos_q.transpose(1, 2), sin_q.transpose(1, 2)
        )
        k_rotated = apply_ct_rotation(
            k, cos_k.transpose(1, 2), sin_k.transpose(1, 2)
        )
        
        return q_rotated, k_rotated

# --- 3. 自定义 Qwen2 注意力层 (包含Bug修复) ---

class CustomQwen2Attention(Qwen2Attention):
    """
    一个自定义的 Qwen2Attention 层，它用 MIRA [7, 8] 的
    连续时间 (CT-RoPE) 机制取代了默认的离散索引 RoPE。
    
    该模块被设计为 plm4ts.py 中标准 Qwen2Attention 层的直接替代品，
    并正确处理：
    1. 连续时间的不规则时间戳 (通过 CT-RoPE)。
    2. 自回归掩码切片 (修复 RuntimeError) [6, 9]。
    3. 通过保留模块名称 (q_proj, k_proj...) 来保证 PEFT/LoRA 的兼容性 [10]。
    """
    
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        # 1. 初始化基类
        #    这将创建 self.q_proj, self.k_proj, self.v_proj, self.o_proj
        #    以及 self.rotary_emb
        super().__init__(config, layer_idx=layer_idx)
        
        # 2. 移除基类的离散 RoPE
        #    我们不再需要它，删除以避免混淆
        del self.rotary_emb
        
        # 3. 初始化我们新的 CT-RoPE 模块
        self.ct_rope = ContinuousTimeRotaryEmbedding(
            self.head_dim,
            base=config.rope_theta,  # 重用QWEN的base参数
            device=config.torch_dtype # 尝试预设设备/类型
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional = None,
        position_ids: Optional = None, # 不再使用, 但保留签名
        past_key_value: Optional] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        
        #!!! 关键的新增参数!!!
        timestamps: Optional = None,
        
        **kwargs,
    ) -> Tuple, Optional]]:
        
        if timestamps is None:
            raise ValueError(
                "CustomQwen2Attention 需要一个 'timestamps' 张量 "
                "来进行 CT-RoPE 计算。"
            )

        bsz, q_len, _ = hidden_states.size()
        
        # 1. 标准投影 (LoRA 将附加到这里)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 2. 重塑以进行多头注意力
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # 3. 获取 KV 序列总长度
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.shape[-2] # index 0 is key_states
        
        # --- CT-ROPE 替换块 开始 ---
        
        # 4. 准备 Q 和 K 的时间戳
        #    timestamps 张量必须是 [b, full_seq_len]
        if past_key_value is not None:
            # 这是自回归 (forecasting) 步骤
            past_k_len = past_key_value.shape[-2]
            
            # Q 的时间戳 (新 token)
            # q_len 将是 1 或 2 (如您的错误所示)
            timestamps_q = timestamps[:, -q_len:] # 形状 [b, q_len]
            
            # K 的时间戳 (所有 token)
            # 我们假设 'timestamps' 包含 *完整* 的历史
            timestamps_k = timestamps[:, -kv_seq_len:] # 形状 [b, kv_seq_len]
        
        else:
            # 这是预填充 (prefill) 步骤 (q_len == k_len)
            timestamps_q = timestamps
            timestamps_k = timestamps

        # 5. 应用 CT-RoPE [7]
        #    这完全取代了原始的 rotary_emb 逻辑
        query_states, key_states = self.ct_rope(
            query_states, key_states, timestamps_q, timestamps_k
        )
        
        # --- CT-ROPE 替换块 结束 ---

        # 6. KV 缓存处理
        if past_key_value is not None:
            key_states = torch.cat([past_key_value, key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 7. GQA (Grouped Query Attention) 处理
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # 8. 计算注意力权重
        #    形状: [b, h, q_len, kv_seq_len]
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # --- 注意力掩码 (BUG修复) 开始 ---
        
        # attn_weights 形状 (例如): [b, h, 2, 192]

        if attention_mask is not None:
            # attention_mask 形状 (例如): [b, 1, 192, 192]
            
            # 这是修复: 切片掩码以匹配 q_len [6, 9]
            current_q_len = query_states.shape[2] # 这是 2 (在您的错误中)
            
            # 切片掩码为 [b, 1, 2, 192]
            sliced_mask = attention_mask[:, :, -current_q_len:, :]
            
            # 应用正确切片的掩码
            # [b, h, 2, 192] + [b, 1, 2, 192] (广播成功)
            attn_weights = attn_weights + sliced_mask
            
        # --- 注意力掩码 (BUG修复) 结束 ---
        
        # 9. 归一化和输出
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # 10. 恢复形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # 11. 最终投影
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
            
        return attn_output, attn_weights, past_key_value

# --- 4. 自定义 Qwen2 解码器层 ---

class CustomQwen2DecoderLayer(Qwen2DecoderLayer):
    """
    一个自定义的 Qwen2DecoderLayer，它使用 CustomQwen2Attention
    并正确地将 'timestamps' 参数传递给注意力层。
    """
    def __init__(self, config: Qwen2Config, layer_idx: int):
        # 1. 初始化基类
        #    基类会创建 self.self_attn = Qwen2Attention(...)
        super().__init__(config, layer_idx)
        
        # 2. 用我们的自定义注意力层替换它
        self.self_attn = CustomQwen2Attention(config, layer_idx)
        
        # self.mlp, self.input_layernorm, self.post_attention_layernorm
        # 均被继承且无需更改

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional = None,
        position_ids: Optional = None,
        past_key_value: Optional] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        
        #!!! 关键的新增参数!!!
        timestamps: Optional = None,
        
        **kwargs,
    ) -> Tuple]]:
        
        if timestamps is None:
            raise ValueError(
                "CustomQwen2DecoderLayer 需要一个 'timestamps' 张量。"
            )
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # *** 关键的修改 ***
        # 将 timestamps 传递给 self_attn
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            timestamps=timestamps,  # <-- 在此传递
        )
        # *** 修改结束 ***
        
        hidden_states = residual + attn_outputs
        
        # MLP block (unchanged)
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

# --- 5. 自定义 Qwen2 模型 (主干) ---

class CustomQwen2Model(Qwen2Model):
    """
    自定义 Qwen2Model，它：
    1. 在 'self.layers' 中使用 CustomQwen2DecoderLayer。
    2. 接受 'timestamps' 作为 forward 参数。
    3. 将 'timestamps' 传递给其所有层。
    """
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        
        # 用我们的自定义层覆盖 self.layers
        self.layers = nn.ModuleList(
           
        )
        # self.embed_tokens, self.norm 被继承
        self.post_init() # 完成模型设置

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional = None,
        position_ids: Optional = None,
        past_key_values: Optional] = None,
        inputs_embeds: Optional = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
        #!!! 关键的新增参数!!!
        timestamps: Optional = None,
        
    ) -> Union:
        
        if timestamps is None:
            raise ValueError(
                "CustomQwen2Model 需要一个 'timestamps' 张量。"
            )
        
        # --- 标准 Qwen2Model.forward() 逻辑 ---
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values.shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # 4D 因果掩码准备 (继承自基类)
        # 这是创建4D掩码的地方。
        # 错误不在于 *创建*，而在于 *应用* (已在 CustomQwen2Attention 中修复)。
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), dtype=torch.bool, device=inputs_embeds.device
            )
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # --- 关键修改: 将 'timestamps' 传递给所有层 ---
        for decoder_layer, layer_past in zip(self.layers, past_key_values or [None] * len(self.layers)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                # 梯度检查点逻辑
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # inputs 包含 (hidden_states, attention_mask, position_ids,
                        #             layer_past, output_attentions, use_cache, timestamps)
                        # 我们必须明确地将 'timestamps' 作为关键字参数传递
                        return module(
                            inputs,
                            attention_mask=inputs[1],
                            position_ids=inputs[2],
                            past_key_value=inputs[3],
                            output_attentions=inputs[4],
                            use_cache=inputs[5],
                            timestamps=inputs[6] # 我们新的参数
                        )
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    layer_past,
                    output_attentions,
                    use_cache,
                    timestamps, # 将 timestamps 作为检查点函数的第7个输入
                    use_reentrant=False, # 推荐使用非重入式
                )
            else:
                # 标准前向传播
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    timestamps=timestamps, # <-- 在此传递
                )
            
            hidden_states = layer_outputs
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        # --- 修改结束 ---

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

# --- 6. 自定义 Qwen2 CausalLM (顶层) ---

class CustomQwen2ForCausalLM(Qwen2ForCausalLM):
    """
    用于因果语言建模的顶层模型。
    1. 使用 CustomQwen2Model 作为其主干 (self.model)。
    2. 接受 'timestamps' 并在 forward() 中传递它。
    3. 修改 'prepare_inputs_for_generation' 以便在.generate() 中
       正确处理 'timestamps' 和 KV 缓存。
    """
    _auto_class = "AutoModelForCausalLM"
    
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        
        # 用我们的自定义模型覆盖主干
        self.model = CustomQwen2Model(config)
        
        # self.vocab_size 和 self.lm_head 被继承且无需更改
        
        self.post_init() # 完成设置

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional = None,
        position_ids: Optional = None,
        past_key_values: Optional] = None,
        inputs_embeds: Optional = None,
        labels: Optional = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
        #!!! 关键的新增参数!!!
        timestamps: Optional = None,
        
    ) -> Union:
        
        # 仅在训练时强制要求
        if timestamps is None and self.training:
            raise ValueError(
                "CustomQwen2ForCausalLM 在训练期间需要一个 'timestamps' 张量。"
            )
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # *** 关键修改: 将 'timestamps' 传递给 self.model ***
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            timestamps=timestamps, # <-- 在此传递
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float() # 转换为 float

        loss = None
        if labels is not None:
            # 损失计算 (不变)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None, 
        inputs_embeds=None, 
        **kwargs
    ):
        """
        覆盖此方法以处理.generate() 调用期间的 'timestamps'。
        """
        
        # 从 kwargs 中提取 'timestamps'
        timestamps = kwargs.pop("timestamps", None)
        if timestamps is None:
             raise ValueError(
                "`model.generate()` 必须使用一个 'timestamps' 浮点张量 "
                "作为关键字参数调用。"
            )
        
        # 调用基类的 prepare_inputs_for_generation
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values, 
            attention_mask=attention_mask, 
            inputs_embeds=inputs_embeds, 
            **kwargs
        )
        
        # --- KV 缓存的时间戳处理 ---
        # 基类方法会处理 input_ids, attention_mask 和 position_ids
        # 的切片 (当 past_key_values 存在时)。
        # 我们必须对 timestamps 执行 *我们自己* 的逻辑，
        # 因为基类不知道它。
        
        if past_key_values is not None:
            # 我们处于解码步骤 (q_len=1)
            # model_inputs['input_ids'] 已经被切片为 [b, 1]
            # 我们 *假设* 传入的 'timestamps' 张量是 *完整* 的
            # 形状为 [b, full_seq_len] 的张量。
            # CustomQwen2Attention 中的逻辑将从这个
            # 完整张量中切片出它需要的部分。
            model_inputs["timestamps"] = timestamps
        else:
            # 我们处于预填充 (prefill) 步骤 (第一次调用)
            # model_inputs['input_ids'] 是 [b, seq_len]
            # timestamps 也应该是 [b, seq_len]
            model_inputs["timestamps"] = timestamps
        
        return model_inputs
