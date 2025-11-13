import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.Dataset_MM as Dataset_MM # (!!!) 恢复了旧的 import

# ==================================================================================
# 节 0: 旧模型的依赖项 (GPT-2 / BERT)
# (!!!) 恢复了旧的 import，以确保 ists_plm, istsplm_vector 等能被定义
# ==================================================================================
from transformers.models.gpt2.modeling_gpt2_wope import GPT2Model_wope
from transformers.models.bert.modeling_bert_wope import BertModel_wope
from models.embed import *

# ==================================================================================
# 节 1: 新模型的依赖项 (Qwen / LoRA / INT4)
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
# 节 2: 新的基础嵌入模块 (Qwen 需要)
# (这些是旧 'models/embed.py' 的 Qwen 兼容版本)
# ==================================================================================

class ValueEmbedding_Qwen(nn.Module):
    def __init__(self, c_in, d_model):
        super(ValueEmbedding_Qwen, self).__init__()
        self.projection = nn.Linear(c_in, d_model)
    def forward(self, x):
        return self.projection(x)

class VariableEmbedding_Qwen(nn.Module):
    def __init__(self, n_var, d_model):
        super(VariableEmbedding_Qwen, self).__init__()
        self.varible_emb = nn.Embedding(n_var, d_model)
    def forward(self, x):
        x = self.varible_emb(x.long())
        return x

# ==================================================================================
# 节 3: CT-RoPE (连续时间旋转位置编码)
# ==================================================================================

class CTRoPERotaryEmbedding(nn.Module):
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
# 节 4: 自定义 Qwen 架构 (以支持 CT-RoPE)
# ==================================================================================

class CustomQwen2Attention(Qwen2Attention):
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
        if hasattr(self, '_use_flash_attention_2') and self._use_flash_attention_2:
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
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super(Qwen2DecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = CustomQwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
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
    def __init__(self, config: Qwen2Config):
        super(Qwen2Model, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([CustomQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, timestamps=None):
        if timestamps is None:
             raise ValueError("CustomQwen2Model 必须接收 'timestamps' 参数")
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
# 节 5: 数据嵌入层 (CT-RoPE 专用)
# ==================================================================================
class DataEmbedding_ITS_for_CTRoPE(nn.Module):
    def __init__(self, c_in, d_model, n_var, device=None, dropout=0.1):
        super(DataEmbedding_ITS_for_CTRoPE, self).__init__()
        self.d_model = d_model
        # (!!!) 使用 节 2 中定义的 Qwen 兼容嵌入层
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
# 节 6: 恢复的旧模型 (ists_plm, istsplm_vector, istsplm_set)
# (!!!) 恢复了你提供的旧代码，以确保所有 'NameError' 都消失
# ==================================================================================

class ists_plm(nn.Module):
    # (!!!) 这是你提供的旧 ists_plm 代码，保持不变
    def __init__(self, opt):
        super(ists_plm, self).__init__()
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        self.enc_embedding = DataEmbedding_ITS_Ind_VarPrompt(self.feat_dim, self.d_model, self.feat_dim, device=opt.device, dropout=opt.dropout)
        self.gpts = nn.ModuleList()
        for i in range(2):
            gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
            bert = BertModel_wope.from_pretrained('./PLMs/bert-base-uncased', output_attentions=True, output_hidden_states=True)
            print(opt.te_model, opt.st_model)
            if(i==0):
                if opt.te_model == 'gpt':
                    gpt2.h = gpt2.h[:opt.n_te_plmlayer]
                    self.gpts.append(gpt2)
                elif opt.te_model == 'bert':
                    bert.encoder.layer = bert.encoder.layer[:opt.n_te_plmlayer]
                    self.gpts.append(bert)
            elif(i==1):
                if opt.st_model == 'gpt':
                    gpt2.h = gpt2.h[:opt.n_st_plmlayer]
                    self.gpts.append(gpt2)
                elif opt.st_model == 'bert':
                    bert.encoder.layer = bert.encoder.layer[:opt.n_st_plmlayer]
                    self.gpts.append(bert)
        if(opt.semi_freeze):
            print("Semi-freeze gpt")
            for i in range(len(self.gpts)):
                for _, (name, param) in enumerate(self.gpts[i].named_parameters()):
                    if 'ln' in name or 'LayerNorm' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        else:
            print("Fully-freeze gpt")
            for i in range(len(self.gpts)):
                for _, (name, param) in enumerate(self.gpts[i].named_parameters()):
                        param.requires_grad = False
        self.act = F.gelu
        self.dropout = nn.Dropout(p=opt.dropout)
        self.ln_proj = nn.LayerNorm(self.d_model)
    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
        B, L, D = observed_data.shape
        outputs, var_embedding = self.enc_embedding(observed_tp, observed_data, observed_mask)
        outputs = self.gpts[0](inputs_embeds=outputs).last_hidden_state
        observed_mask = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1)
        observed_mask = torch.cat([torch.ones_like(observed_mask[:,:1]), observed_mask], dim=1)
        n_nonmask = observed_mask.sum(dim=1) 
        outputs = (outputs * observed_mask).sum(dim=1) / n_nonmask
        outputs = self.ln_proj(outputs.view(B, D, -1)) 
        outputs = outputs + var_embedding.squeeze()
        outputs = self.gpts[1](inputs_embeds=outputs).last_hidden_state
        outputs = outputs.view(B, -1)
        return outputs

class istsplm_vector(nn.Module):
    # (!!!) 这是你提供的旧 istsplm_vector 代码，保持不变
    def __init__(self, opt):
        super(istsplm_vector, self).__init__()
        self.seq_len = opt.max_len
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        self.enc_embedding = DataEmbedding_ITS_Vector(self.feat_dim, self.d_model, opt.dropout)
        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
        observed_mask_agg = observed_mask.sum(dim=-1, keepdim=True).bool()
        outputs = self.enc_embedding(observed_tp, observed_data, observed_mask, observed_mask_agg)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        n_nonmask = observed_mask_agg.sum(dim=1) 
        outputs = (outputs * observed_mask_agg).sum(dim=1) / (n_nonmask + 1e-8)
        return outputs

class istsplm_set(nn.Module):
    # (!!!) 这是你提供的旧 istsplm_set 代码，保持不变
    def __init__(self, opt):
        super(istsplm_set, self).__init__()
        self.seq_len = opt.max_len
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        self.plm_maxlen = 1024
        self.enc_embedding = DataEmbedding_ITS_Set(self.feat_dim, self.d_model, opt.dropout)
        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
        observed_tp = observed_tp.squeeze(dim=-1)
        observed_triple, observed_mask = Dataset_MM.collate_fn_triple(observed_tp, observed_data, observed_mask)
        observed_mask = observed_mask.unsqueeze(dim=-1)
        outputs = self.enc_embedding(observed_triple, observed_mask)
        outputs = outputs[:,:self.plm_maxlen]
        observed_mask = observed_mask[:,:self.plm_maxlen]
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        n_nonmask = observed_mask.sum(dim=1) 
        outputs = (outputs * observed_mask).sum(dim=1) / (n_nonmask + 1e-8)
        return outputs

# ==================================================================================
# 节 7: (!!!) 新的核心模型 (istsplm_forecast)
# (已更新为 Qwen + CT-RoPE + LoRA + INT4 并添加了预测头)
# ==================================================================================
class istsplm_forecast(nn.Module):
    
    def __init__(self, opt):
        super(istsplm_forecast, self).__init__()
        
        # (!!!) 确保从 opt 中获取了所有必要参数
        self.pred_len = opt.pred_len
        self.input_len = opt.input_len
        self.d_model = opt.d_model
        self.device = opt.device
        self.n_var = opt.num_types # 从 opt.num_types 获取变量数
        
        # --- 1. 嵌入层 (使用 节 5 的新嵌入层) ---
        self.enc_embedding = DataEmbedding_ITS_for_CTRoPE(
            c_in=2, 
            d_model=self.d_model, 
            n_var=self.n_var, 
            device=opt.device, 
            dropout=opt.dropout
        )

        self.gpts = nn.ModuleList()
        
        # --- 2. 加载模型 (INT4 + 预训练权重) ---
        plm_name = opt.plm_path
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        print(f"Loading Time-Aware PLM (CustomQwen2Model) from {plm_name} WITH INT4...")
        time_aware_plm = CustomQwen2Model.from_pretrained(
            plm_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
        )
        
        print(f"Loading Variable-Aware PLM (Standard Qwen2Model) from {plm_name} WITH INT4...")
        variable_aware_plm = Qwen2Model.from_pretrained(
            plm_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
        )
            
        # --- 3. 应用 LoRA ---
        lora_config = LoraConfig(
            r=opt.lora_r,
            lora_alpha=opt.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM, 
        )
        
        print("Applying LoRA to INT4 Time-Aware PLM...")
        time_aware_plm = get_peft_model(time_aware_plm, lora_config)
        time_aware_plm.print_trainable_parameters()
        
        print("Applying LoRA to INT4 Variable-Aware PLM...")
        variable_aware_plm = get_peft_model(variable_aware_plm, lora_config)
        variable_aware_plm.print_trainable_parameters()
        
        self.gpts.append(time_aware_plm)
        self.gpts.append(variable_aware_plm)
            
        # --- 4. 预测头 (Forecasting Head) ---
        # (!!!) 移植了旧代码的预测头逻辑
        self.ln_proj = nn.LayerNorm(self.d_model)
        
        # (!!!) 这是我们新的 Qwen 模型的输出维度
        # (B, D, d_model) -> (B, D * d_model)
        qwen_output_dim = self.d_model * self.n_var 
        
        # (!!!) 这是我们需要的预测输出维度
        # (B, pred_len, D) -> (B, pred_len * D)
        prediction_dim = self.pred_len * self.n_var
        
        # (!!!) 新的预测头
        self.forecasting_head = nn.Linear(qwen_output_dim, prediction_dim)
        
        # (!!!) 移植了旧的 predict_decoder (以防 evaluation.py 需要它)
        # (注意：新的 forecasting 方法不会使用这个，但保留它以防万一)
        self.predict_decoder = nn.Sequential(
			nn.Linear(opt.d_model+1, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, 1)
		).to(opt.device)
        

    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        """
        这是模型的主前向传播函数，用于预测。
        """
        B, L, D = observed_data.shape
        
        # 1. 嵌入 (New)
        inputs_embeds, tt_with_prompt, var_embedding = self.enc_embedding(
            observed_tp, observed_data, observed_mask
        )

        # 2. PLM 1 (Time-Aware)
        time_attn_mask_data = observed_mask.permute(0, 2, 1)
        time_attn_mask_data_flat = time_attn_mask_data.reshape(B * D, L)
        time_attn_mask_prompt = torch.ones((B * D, 1), device=self.device, dtype=torch.long)
        time_attn_mask = torch.cat([time_attn_mask_prompt, time_attn_mask_data_flat], dim=1)
        
        outputs = self.gpts[0](
            inputs_embeds=inputs_embeds,
            attention_mask=time_attn_mask,
            timestamps=tt_with_prompt
        ).last_hidden_state

        # 3. 掩码池化 (Masked Pooling)
        observed_mask_pooled = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1)
        observed_mask_pooled = torch.cat([torch.ones_like(observed_mask_pooled[:,:1]), observed_mask_pooled], dim=1)
        n_nonmask = (observed_mask_pooled.sum(dim=1) + 1e-8)
        outputs = (outputs * observed_mask_pooled).sum(dim=1) / n_nonmask
        outputs = self.ln_proj(outputs.view(B, D, -1))
        
        # 4. PLM 2 (Variable-Aware)
        outputs = outputs + var_embedding.squeeze(1)
        var_position_ids = torch.arange(D, device=self.device).unsqueeze(0).expand(B, D)
        var_attn_mask = torch.ones(B, D, device=self.device, dtype=torch.long)
        
        outputs = self.gpts[1](
            inputs_embeds=outputs,
            attention_mask=var_attn_mask,
            position_ids=var_position_ids
        ).last_hidden_state # (B, D, d_model)

        # 5. (!!!) 新的预测头逻辑
        # (B, D * d_model)
        outputs = outputs.reshape(B, -1)
        # (B, pred_len * D)
        outputs = self.forecasting_head(outputs)
        # (B, pred_len, D)
        outputs = outputs.reshape(B, self.pred_len, -1)
        
        # (!!!) 确保输出形状与 evaluation.py 兼容
        # 原始代码返回 (1, B, L, D)
        # 我们的返回 (B, pred_len, D)
        # evaluation.py 里的 compute_all_losses 需要处理这个
        # 我们在这里添加一个维度以匹配旧代码
        return outputs.unsqueeze(0) # (1, B, pred_len, D)

# ==================================================================================
# 节 8: 恢复的旧 forecast 模型 (vector, set)
# (!!!) 恢复了你提供的旧代码，以确保所有 'NameError' 都消失
# ==================================================================================

class istsplm_vector_forecast(nn.Module):
    # (!!!) 这是你提供的旧 istsplm_vector_forecast 代码，保持不变
    def __init__(self, opt):
        super(istsplm_vector_forecast, self).__init__()
        self.seq_len = opt.max_len
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        self.enc_embedding = DataEmbedding_ITS_Vector(self.feat_dim, self.d_model, opt.dropout)
        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.predict_decoder = nn.Sequential(
			nn.Linear(opt.d_model+1, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, opt.input_dim)
			).to(opt.device)
    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        observed_tp = observed_tp.unsqueeze(dim=-1)
        observed_mask_agg = observed_mask.sum(dim=-1, keepdim=True).bool()
        outputs = self.enc_embedding(observed_tp, observed_data, observed_mask, observed_mask_agg)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        n_nonmask = observed_mask_agg.sum(dim=1) 
        outputs = (outputs * observed_mask_agg).sum(dim=1) / (n_nonmask + 1e-8)
        B, Lp = time_steps_to_predict.size()
        time_pred = time_steps_to_predict.unsqueeze(dim = -1)
        h = outputs.unsqueeze(dim=1).repeat(1, Lp, 1)
        h = torch.cat([h, time_pred], dim=-1)
        output = self.predict_decoder(h).unsqueeze(dim=0)
        return output

class istsplm_set_forecast(nn.Module):
    # (!!!) 这是你提供的旧 istsplm_set_forecast 代码，保持不变
    def __init__(self, opt):
        super(istsplm_set_forecast, self).__init__()
        self.seq_len = opt.max_len
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        self.plm_maxlen = 1024
        self.enc_embedding = DataEmbedding_ITS_Set(self.feat_dim, self.d_model, opt.dropout)
        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.predict_decoder = nn.Sequential(
			nn.Linear(opt.d_model+1, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, opt.input_dim)
			).to(opt.device)
    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        observed_triple, observed_mask = Dataset_MM.collate_fn_triple(observed_tp, observed_data, observed_mask)
        observed_mask = observed_mask.unsqueeze(dim=-1)
        outputs = self.enc_embedding(observed_triple, observed_mask)
        outputs = outputs[:,:self.plm_maxlen]
        observed_mask = observed_mask[:,:self.plm_maxlen]
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        n_nonmask = observed_mask.sum(dim=1) 
        outputs = (outputs * observed_mask).sum(dim=1) / (n_nonmask + 1e-8)
        B, Lp = time_steps_to_predict.size()
        time_pred = time_steps_to_predict.unsqueeze(dim = -1)
        h = outputs.unsqueeze(dim=1).repeat(1, Lp, 1)
        h = torch.cat([h, time_pred], dim=-1)
        output = self.predict_decoder(h).unsqueeze(dim=0)
        return output

class Classifier(nn.Module):
    # (!!!) 这是你提供的旧 Classifier 代码，保持不变
    def __init__(self, dim, cls_dim, activate=None):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(dim, cls_dim)
    def forward(self, ENCoutput):
        ENCoutput = self.linear(ENCoutput)
        return ENCoutput
