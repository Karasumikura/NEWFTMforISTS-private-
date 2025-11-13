import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


class ValueEmbeddingQwen(nn.Module):
    """
    Projects scalar values to d_model.
    Input shape: (B*D, L, 1)
    Output shape: (B*D, L, d_model)
    """
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


class istsplm_forecast(nn.Module):
    """
    ISTS forecasting model using a two-stage PLM pipeline (time-level + variable-level) based on Qwen2.5.
    - Continuous Time Rotary Position Embeddings (CT-RoPE) injected by wrapping each self_attn.forward.
    - Pure float32 pipeline for simplicity.
    - LoRA optional (applied only on q_proj/k_proj/v_proj/o_proj).
    Batch dict expected keys:
        observed_data: (B, L_obs, D)
        observed_mask: (B, L_obs, D)
        observed_tp:   (B, L_obs) or (B, 1, L_obs)
        tp_to_predict: (B, L_pred)
        data_to_predict: (B, L_pred, D)
        mask_predicted_data: (B, L_pred, D)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.d_model = args.d_model
        self.num_types = args.num_types
        self.prompt_len = args.prompt_len
        self.use_lora = args.use_lora
        self.plm_name = args.plm_path
        self.n_plm_layers = 2  # 0: time-level, 1: variable-level

        # Embeddings
        self.value_emb = ValueEmbeddingQwen(1, self.d_model)
        self.var_emb = nn.Embedding(self.num_types, self.d_model)
        self.prompt_embed = nn.Parameter(torch.randn(1, self.prompt_len, self.d_model))

        # Load Qwen2.5 in float32
        self.config = AutoConfig.from_pretrained(self.plm_name, trust_remote_code=True)
        self.config.use_cache = False  # no generation caching needed for batch forecasting

        gpts = []
        for _ in range(self.n_plm_layers):
            base = AutoModelForCausalLM.from_pretrained(
                self.plm_name,
                trust_remote_code=True,
                torch_dtype=torch.float32  # force float32
            )
            core = getattr(base, "model", base)  # some Qwen variants expose .model
            core = self._inject_ctr_rope(core, verbose=False)
            if self.use_lora:
                core = self._apply_lora(self.args, core)
            gpts.append(core)

        self.gpts = nn.ModuleList(gpts)

        # Projection and forecasting head
        self.ln_proj = nn.LayerNorm(self.d_model)
        self.forecasting_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LeakyReLU(),
            nn.Linear(self.d_model // 2, self.num_types)
        )

    # ---------------- LoRA ----------------
    def _apply_lora(self, args, core_model):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        core_model = get_peft_model(core_model, lora_cfg)
        for n, p in core_model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
        return core_model

    # ---------------- CT-RoPE Injection ----------------
    def _inject_ctr_rope(self, core_model, verbose: bool = True):
        if not hasattr(core_model, "layers"):
            raise ValueError("Qwen2.5 core model has no 'layers' attribute.")
        layers = core_model.layers

        try:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        except Exception:
            raise RuntimeError("Failed to import apply_rotary_pos_emb. Upgrade transformers for Qwen2.5 support.")

        def wrap(old_attn):
            q_proj = getattr(old_attn, "q_proj", None)
            k_proj = getattr(old_attn, "k_proj", None)
            v_proj = getattr(old_attn, "v_proj", None)
            o_proj = getattr(old_attn, "o_proj", None)
            if any(x is None for x in [q_proj, k_proj, v_proj, o_proj]):
                raise ValueError("Attention layer missing q_proj/k_proj/v_proj/o_proj")

            cfg = getattr(old_attn, "config", None)
            num_heads = getattr(old_attn, "num_heads", None) or getattr(cfg, "num_attention_heads", None)
            num_kv = getattr(old_attn, "num_key_value_heads", None) or getattr(cfg, "num_key_value_heads", num_heads)
            hidden_size = getattr(cfg, "hidden_size", q_proj.in_features)
            head_dim = hidden_size // num_heads
            rotary_emb = getattr(old_attn, "rotary_emb", None)
            attn_dropout = getattr(old_attn, "attn_dropout", nn.Dropout(0.0))

            def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
                if n_rep == 1:
                    return kv
                bsz, kv_heads, seqlen, hd = kv.shape
                return kv.unsqueeze(2).expand(bsz, kv_heads, n_rep, seqlen, hd).reshape(
                    bsz, kv_heads * n_rep, seqlen, hd
                )

            def new_forward(
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                timestamps: Optional[torch.Tensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs,
            ):
                hidden_states = hidden_states.float()

                bsz, q_len, _ = hidden_states.shape
                q = q_proj(hidden_states)
                k = k_proj(hidden_states)
                v = v_proj(hidden_states)

                q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                k = k.view(bsz, q_len, num_kv, head_dim).transpose(1, 2)
                v = v.view(bsz, q_len, num_kv, head_dim).transpose(1, 2)

                kv_seq_len = k.shape[-2]
                if past_key_value is not None:
                    kv_seq_len += past_key_value[0].shape[-2]

                # CT-RoPE with continuous timestamps
                if rotary_emb is not None:
                    cos, sin = rotary_emb(v, seq_len=kv_seq_len)
                    if timestamps is not None:
                        ts = timestamps.unsqueeze(1).unsqueeze(-1).to(k.dtype)
                        q, k = apply_rotary_pos_emb(q, k, cos, sin, ts)
                    elif position_ids is not None:
                        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

                if past_key_value is not None:
                    k = torch.cat([past_key_value[0], k], dim=2)
                    v = torch.cat([past_key_value[1], v], dim=2)

                if num_kv != num_heads:
                    k = repeat_kv(k, num_heads // num_kv)
                    v = repeat_kv(v, num_heads // num_kv)

                past = (k, v) if use_cache else None

                attn_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(attn_scores.dtype)
                    if cache_position is not None:
                        mask_q_len = attention_mask.shape[-2]
                        if mask_q_len != q_len:
                            start_row = mask_q_len - q_len
                            attention_mask = attention_mask[:, :, start_row:, :]
                    attn_scores = attn_scores + attention_mask

                attn_weights = nn.functional.softmax(attn_scores, dim=-1)
                attn_weights = attn_dropout(attn_weights)
                attn_output = torch.matmul(attn_weights, v)

                if attn_output.shape != (bsz, num_heads, q_len, head_dim):
                    raise ValueError(f"attn_output shape mismatch {attn_output.shape}")

                attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, num_heads * head_dim)
                attn_output = o_proj(attn_output)

                attn_ret = attn_weights if output_attentions else None
                if use_cache:
                    return attn_output, attn_ret, past
                else:
                    return attn_output, attn_ret

            return new_forward

        for i, layer in enumerate(layers):
            if hasattr(layer, "self_attn"):
                layer.self_attn.forward = wrap(layer.self_attn)
                if verbose:
                    print(f"[CT-RoPE] Patched layer {i} attention: {type(layer.self_attn)}")
        return core_model

    # ---------------- Forecasting ----------------
    def forecasting(self, batch_dict, n_vars_to_predict=None):
        observed_data = batch_dict["observed_data"].to(self.device).float()    # (B, L_obs, D)
        observed_mask = batch_dict["observed_mask"].to(self.device).float()    # (B, L_obs, D)
        observed_tp_raw = batch_dict["observed_tp"].to(self.device).float()    # (B, L_obs) or (B, 1, L_obs)
        tp_to_predict = batch_dict["tp_to_predict"].to(self.device)            # (B, L_pred)

        B, L_obs, D = observed_data.shape
        L_pred = tp_to_predict.shape[1]

        # Value embeddings per variable
        x_flat = observed_data.permute(0, 2, 1).reshape(B * D, L_obs, 1).float()
        value_embed = self.value_emb(x_flat)  # (B*D, L_obs, d_model)
        L_val = value_embed.size(1)

        # Align timestamps length
        observed_tp = observed_tp_raw.reshape(B, -1)
        L_time = observed_tp.size(1)
        if L_time != L_val:
            if L_time >= L_val:
                tp_aligned = observed_tp[:, -L_val:]
            else:
                pad = torch.zeros(B, L_val - L_time, device=self.device)
                tp_aligned = torch.cat([observed_tp, pad], dim=1)
        else:
            tp_aligned = observed_tp

        tt = tp_aligned.unsqueeze(1).expand(B, D, L_val).reshape(B * D, L_val)

        # Prompt + concat
        prompt = self.prompt_embed.expand(B * D, -1, -1).float()
        inputs_embeds = torch.cat([prompt, value_embed], dim=1)  # (B*D, prompt_len + L_val, d_model)

        tt_prompt = torch.zeros(B * D, self.prompt_len, device=self.device)
        tt_with_prompt = torch.cat([tt_prompt, tt], dim=1)  # (B*D, prompt_len + L_val)

        # Time mask
        time_mask_flat = observed_mask.permute(0, 2, 1).reshape(B * D, L_obs)
        if L_obs != L_val:
            if L_obs >= L_val:
                time_mask_flat = time_mask_flat[:, -L_val:]
            else:
                padm = torch.zeros(B * D, L_val - L_obs, device=self.device)
                time_mask_flat = torch.cat([time_mask_flat, padm], dim=1)

        time_mask_prompt = torch.ones(B * D, self.prompt_len, device=self.device)
        time_mask = torch.cat([time_mask_prompt, time_mask_flat], dim=1)  # (B*D, total_len)
        additive_mask = (1.0 - time_mask) * torch.finfo(torch.float32).min
        additive_mask = additive_mask.unsqueeze(1).unsqueeze(1)  # (B*D,1,1,total_len)

        # Time PLM forward (no cache)
        out_time = self.gpts[0](
            inputs_embeds=inputs_embeds,
            attention_mask=additive_mask,
            timestamps=tt_with_prompt,
            use_cache=False,
            output_attentions=False
        ).last_hidden_state  # (B*D, total_len, d_model)

        # Masked pooling (remove prompt portion)
        mask_pool = observed_mask.permute(0, 2, 1).reshape(B * D, L_obs, 1)
        if L_obs != L_val:
            if L_obs >= L_val:
                mask_pool = mask_pool[:, -L_val:, :]
            else:
                padm2 = torch.zeros(B * D, L_val - L_obs, 1, device=self.device)
                mask_pool = torch.cat([mask_pool, padm2], dim=1)

        out_val = out_time[:, self.prompt_len:]  # (B*D, L_val, d_model)
        denom = mask_pool.sum(dim=1, keepdim=True) + 1e-8
        pooled = (out_val * mask_pool).sum(dim=1) / denom.squeeze(1)  # (B*D, d_model)
        pooled = self.ln_proj(pooled).view(B, D, -1)  # (B, D, d_model)

        # Variable-level PLM input
        var_emb = self.var_emb.weight.unsqueeze(0).expand(B, D, -1).float()
        var_inputs = pooled + var_emb
        var_attn_mask = torch.zeros(B, D, D, device=self.device).unsqueeze(1)  # (B,1,D,D)
        position_ids = torch.arange(D, device=self.device).unsqueeze(0).expand(B, D)

        out_var = self.gpts[1](
            inputs_embeds=var_inputs,
            attention_mask=var_attn_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False
        ).last_hidden_state  # (B, D, d_model)

        # Forecast head
        forecast_vars = self.forecasting_head(out_var)  # (B, D, D)
        regression_value = forecast_vars[:, :, 0]       # (B, D)
        forecast = regression_value.unsqueeze(1).expand(-1, L_pred, -1)  # (B, L_pred, D)

        if n_vars_to_predict is not None:
            forecast = forecast[:, :, :n_vars_to_predict]

        return forecast.float()

    def forward(self, batch_dict, n_vars_to_predict=None):
        return self.forecasting(batch_dict, n_vars_to_predict)
