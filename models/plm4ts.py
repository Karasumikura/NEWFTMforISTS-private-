import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List

# ==================================================================================
# Sección 1: Dependencias del NUEVO modelo (Qwen / LoRA / INT4)
# (!!!) NO MÁS importaciones de GPT2Model_wope o BertModel_wope (!!!)
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
# Sección 2: Módulos de Embedding de base (Qwen necesita esto)
# ==================================================================================

class ValueEmbedding_Qwen(nn.Module):
    """Proyecta pares (valor, máscara) a d_model"""
    def __init__(self, c_in, d_model):
        super(ValueEmbedding_Qwen, self).__init__()
        self.projection = nn.Linear(c_in, d_model)
    def forward(self, x):
        return self.projection(x)

class VariableEmbedding_Qwen(nn.Module):
    """Crea embeddings para D variables de series temporales"""
    def __init__(self, n_var, d_model):
        super(VariableEmbedding_Qwen, self).__init__()
        self.varible_emb = nn.Embedding(n_var, d_model)
    def forward(self, x):
        x = self.varible_emb(x.long())
        return x

# ==================================================================================
# Sección 3: CT-RoPE (Rotary Position Embedding de Tiempo Continuo)
# ==================================================================================

class CTRoPERotaryEmbedding(nn.Module):
    """Implementa CT-RoPE"""
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
# Sección 4: Arquitectura Qwen personalizada (para soportar CT-RoPE)
# ==================================================================================

class CustomQwen2Attention(Qwen2Attention):
    """Atención Qwen2 modificada para usar CTRoPERotaryEmbedding"""
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.rotary_emb = CTRoPERotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta,
        )
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        if "timestamps" not in kwargs:
            raise ValueError("CustomQwen2Attention debe recibir el argumento 'timestamps'")
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
        
        # Lógica de atención (Flash o SDPA)
        if hasattr(self, '_use_flash_attention_2') and self._use_flash_attention_2:
             attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=self.attention_dropout)
        elif hasattr(self, '_sdpa_attention_forward'):
             attn_output = self._sdpa_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=self.attention_dropout)
        else:
            # Fallback para versiones antiguas de transformers
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
    """Capa de decodificador Qwen2 modificada para pasar 'timestamps'"""
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
            raise ValueError("CustomQwen2DecoderLayer debe recibir el argumento 'timestamps'")
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
    """Modelo Qwen2 modificado para aceptar 'timestamps'"""
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
             raise ValueError("CustomQwen2Model debe recibir el argumento 'timestamps'")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Debes proporcionar 'input_ids' o 'inputs_embeds', pero no ambos")
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
# Sección 5: Capa de Embedding de Datos (Específica para CT-RoPE)
# ==================================================================================
class DataEmbedding_ITS_for_CTRoPE(nn.Module):
    """Capa de embedding de datos modificada para CTRoPE"""
    def __init__(self, c_in, d_model, n_var, device=None, dropout=0.1):
        super(DataEmbedding_ITS_for_CTRoPE, self).__init__()
        self.d_model = d_model
        # Usa los módulos de embedding de la Sección 2
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
# Sección 6: El Modelo Principal (istsplm_forecast)
# (!!!) ¡Este es el único modelo que queda! (!!!)
# ==================================================================================
class istsplm_forecast(nn.Module):
    
    def __init__(self, opt):
        super(istsplm_forecast, self).__init__()
        
        self.pred_len = opt.pred_len
        self.input_len = opt.input_len
        self.d_model = opt.d_model
        self.device = opt.device
        self.n_var = opt.num_types # Obtiene el número de variables de opt.num_types
        
        # --- 1. Capa de Embedding (de la Sección 5) ---
        self.enc_embedding = DataEmbedding_ITS_for_CTRoPE(
            c_in=2, 
            d_model=self.d_model, 
            n_var=self.n_var, 
            device=opt.device, 
            dropout=opt.dropout
        )

        self.gpts = nn.ModuleList()
        
        # --- 2. Carga del Modelo (INT4 + Pesos Preentrenados) ---
        plm_name = opt.plm_path
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        print(f"Cargando PLM Time-Aware (CustomQwen2Model) desde {plm_name} CON INT4...")
        time_aware_plm = CustomQwen2Model.from_pretrained(
            plm_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
        )
        
        print(f"Cargando PLM Variable-Aware (Standard Qwen2Model) desde {plm_name} CON INT4...")
        variable_aware_plm = Qwen2Model.from_pretrained(
            plm_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
        )
            
        # --- 3. Aplicar LoRA ---
        lora_config = LoraConfig(
            r=opt.lora_r,
            lora_alpha=opt.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION, 
        )
        
        print("Aplicando LoRA al PLM Time-Aware INT4...")
        time_aware_plm = get_peft_model(time_aware_plm, lora_config)
        time_aware_plm.print_trainable_parameters()
        
        print("Aplicando LoRA al PLM Variable-Aware INT4...")
        variable_aware_plm = get_peft_model(variable_aware_plm, lora_config)
        variable_aware_plm.print_trainable_parameters()
        
        self.gpts.append(time_aware_plm)
        self.gpts.append(variable_aware_plm)
            
        # --- 4. Cabezal de Predicción (Forecasting Head) ---
        self.ln_proj = nn.LayerNorm(self.d_model)
        
        # La salida de Qwen será (B, D * d_model)
        qwen_output_dim = self.d_model * self.n_var 
        
        # La salida de predicción requerida es (B, pred_len * D)
        prediction_dim = self.pred_len * self.n_var
        
        # El cabezal de predicción final
        self.forecasting_head = nn.Linear(qwen_output_dim, prediction_dim)
        
        # (!!!) Mantenemos 'predict_decoder' por si acaso 'evaluation.py' lo usa
        # aunque nuestro 'forecasting' principal no lo hará.
        self.predict_decoder = nn.Sequential(
			nn.Linear(opt.d_model+1, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, 1)
		).to(opt.device)
        

    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        """
        Función principal de forward propagation para predicción.
        """
        B, L, D = observed_data.shape
        
        # 1. Embedding
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
            timestamps=tt_with_prompt # <-- Pasa el tiempo continuo
        ).last_hidden_state

        # 3. Masked Pooling
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
            position_ids=var_position_ids # <-- Pasa índices discretos
        ).last_hidden_state # (B, D, d_model)

        # 5. Cabezal de Predicción
        outputs = outputs.reshape(B, -1) # (B, D * d_model)
        outputs = self.forecasting_head(outputs) # (B, pred_len * D)
        outputs = outputs.reshape(B, self.pred_len, -1) # (B, pred_len, D)
        
        # Añade una dimensión para compatibilidad con el script de evaluación
        return outputs.unsqueeze(0) # (1, B, pred_len, D)
