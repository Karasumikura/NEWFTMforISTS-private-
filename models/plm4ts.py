import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
from peft import get_peft_model, LoraConfig, TaskType


class ValueEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


class istsplm_forecast(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.d_model = args.d_model
        self.num_types = args.num_types

        self.use_lora = getattr(args, "use_lora", False)
        self.use_lora_bert = getattr(args, "use_lora_bert", False)
        self.semi_freeze = getattr(args, "semi_freeze", False)
        self.enable_ct_rope = getattr(args, "enable_ct_rope", True)
        self.debug_flag = getattr(args, "debug_flag", False)

        self.qwen_path = args.plm_path
        self.bert_path = getattr(args, "bert_path", "bert-base-uncased")

        self.value_emb = ValueEmbedding(2, self.d_model)
        self.var_emb = nn.Embedding(self.num_types, self.d_model)
        self.time_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.qwen_config = AutoConfig.from_pretrained(self.qwen_path, trust_remote_code=True)
        self.qwen_config.use_cache = False
        qwen_base = AutoModelForCausalLM.from_pretrained(
            self.qwen_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        self.qwen = getattr(qwen_base, "model", qwen_base)
        if self.enable_ct_rope:
            self.qwen = self._inject_ctr_rope(self.qwen, verbose=False)

        self.bert = AutoModel.from_pretrained(self.bert_path, torch_dtype=torch.float32)
        self._disable_bert_position_embeddings(self.bert)
        self.bert_hidden = self.bert.config.hidden_size
        if hasattr(args, "n_st_plmlayer"):
            self.bert.encoder.layer = self.bert.encoder.layer[:args.n_st_plmlayer]

        if self.bert_hidden != self.d_model:
            self.proj_to_bert = nn.Linear(self.d_model, self.bert_hidden)
            self.proj_from_bert = nn.Linear(self.bert_hidden, self.d_model)
        else:
            self.proj_to_bert = nn.Identity()
            self.proj_from_bert = nn.Identity()

        self._apply_freeze_policies()

        self.ln_proj = nn.LayerNorm(self.d_model)
        self.var_time_mlp = nn.Sequential(
            nn.Linear(self.d_model + 1, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 1)
        )

    @staticmethod
    def _disable_bert_position_embeddings(bert_model: AutoModel):
        emb = bert_model.embeddings
        if hasattr(emb, "position_embeddings"):
            with torch.no_grad():
                emb.position_embeddings.weight.zero_()
            emb.position_embeddings.weight.requires_grad = False
        if hasattr(emb, "token_type_embeddings"):
            with torch.no_grad():
                emb.token_type_embeddings.weight.zero_()
            emb.token_type_embeddings.weight.requires_grad = False

    def _apply_lora_qwen(self, args, model):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=getattr(args, "lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        model = get_peft_model(model, lora_cfg)
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
        return model

    def _apply_lora_bert(self, args, model):
        target_modules = ["query", "key", "value", "dense"]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=getattr(args, "lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        model = get_peft_model(model, lora_cfg)
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
        return model

    @staticmethod
    def _freeze_or_semi(model: nn.Module, semi: bool):
        def is_norm(name: str):
            lname = name.lower()
            return ("layernorm" in lname) or ("ln" in lname) or ("norm" in lname)
        if not semi:
            for _, p in model.named_parameters():
                p.requires_grad = False
        else:
            for n, p in model.named_parameters():
                p.requires_grad = is_norm(n)

    def _apply_freeze_policies(self):
        if self.use_lora:
            self.qwen = self._apply_lora_qwen(self.args, self.qwen)
            if self.semi_freeze:
                for n, p in self.qwen.named_parameters():
                    if ("layernorm" in n.lower() or "norm" in n.lower()) and "lora" not in n:
                        p.requires_grad = True
        else:
            self._freeze_or_semi(self.qwen, semi=self.semi_freeze)

        if self.use_lora_bert:
            self.bert = self._apply_lora_bert(self.args, self.bert)
            if self.semi_freeze:
                for n, p in self.bert.named_parameters():
                    if ("layernorm" in n.lower() or "norm" in n.lower()) and "lora" not in n:
                        p.requires_grad = True
        else:
            self._freeze_or_semi(self.bert, semi=self.semi_freeze)

    def _inject_ctr_rope(self, core_model, verbose: bool = True):
        if not hasattr(core_model, "layers"):
            raise ValueError("Qwen backbone缺少 layers 属性。")
        layers = core_model.layers

        # 连续时间版 RoPE：自实现，避免依赖 Qwen 内部的 apply_rotary_pos_emb 签名
        def _apply_rotary_pos_emb_continuous(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            # x: (bsz, n_heads, seq_len, head_dim)
            # cos/sin: (bsz, 1, seq_len, head_dim//2)
            hd = x.size(-1)
            assert hd % 2 == 0, "head_dim 必须为偶数以应用 RoPE"
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            # broadcast cos/sin to (bsz, n_heads, seq_len, head_dim//2)
            cos_b = cos.expand(-1, x.size(1), -1, -1)
            sin_b = sin.expand(-1, x.size(1), -1, -1)
            x1_new = x1 * cos_b - x2 * sin_b
            x2_new = x2 * cos_b + x1 * sin_b
            # interleave even/odd back
            out = torch.empty_like(x)
            out[..., 0::2] = x1_new
            out[..., 1::2] = x2_new
            return out

        def wrap(old_attn):
            original_forward = old_attn.forward
            q_proj = getattr(old_attn, "q_proj", None)
            k_proj = getattr(old_attn, "k_proj", None)
            v_proj = getattr(old_attn, "v_proj", None)
            o_proj = getattr(old_attn, "o_proj", None)
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
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs,
            ):
                if not self.enable_ct_rope or rotary_emb is None:
                    return original_forward(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        **kwargs,
                    )

                # 从 backbone 属性读取 timestamps（由上层在调用前设置）
                timestamps = getattr(core_model, "_ct_timestamps", None)
                if timestamps is None:
                    raise ValueError("[CT-RoPE] timestamps is None (必须提供).")
                bsz, q_len, hidden_size = hidden_states.shape
                if timestamps.shape[-1] != q_len:
                    raise ValueError(f"[CT-RoPE] timestamps length {timestamps.shape[-1]} != q_len {q_len}")
                if q_len < 2:
                    raise ValueError(f"[CT-RoPE] 序列长度过短 (q_len={q_len})，不允许使用 CT-RoPE。")

                hidden_states = hidden_states.float()
                q = q_proj(hidden_states)
                k = k_proj(hidden_states)
                v = v_proj(hidden_states)

                head_dim_attr = getattr(old_attn, "head_dim", None)
                if head_dim_attr is not None:
                    head_dim = head_dim_attr
                    num_query_heads = q.shape[-1] // head_dim
                else:
                    tentative_heads = getattr(old_attn, "num_heads", None)
                    if tentative_heads is None:
                        head_dim = 64
                        num_query_heads = q.shape[-1] // head_dim
                    else:
                        head_dim = q.shape[-1] // tentative_heads
                        num_query_heads = tentative_heads

                num_kv_heads = getattr(old_attn, "num_key_value_heads", None)
                if num_kv_heads is None:
                    num_kv_heads = num_query_heads

                if q.shape[-1] != num_query_heads * head_dim:
                    raise ValueError(f"[CT-RoPE] q_proj输出维度 {q.shape[-1]} 不等于 num_query_heads*head_dim {num_query_heads*head_dim}")

                q = q.view(bsz, q_len, num_query_heads, head_dim).transpose(1, 2)
                k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

                kv_seq_len = k.shape[-2]
                if past_key_value is not None:
                    kv_seq_len += past_key_value[0].shape[-2]

                # 连续时间归一化
                tmin = timestamps.min(dim=-1, keepdim=True).values
                tmax = timestamps.max(dim=-1, keepdim=True).values
                span = torch.clamp(tmax - tmin, min=1e-8)
                t_norm = (timestamps - tmin) / span
                t_scaled = t_norm * self.time_scale  # (bsz, q_len)

                # 使用 attn 的 RotaryEmbedding 参数（inv_freq）按连续时间生成 cos/sin
                if not hasattr(rotary_emb, "inv_freq"):
                    raise ValueError("[CT-RoPE] 找不到 rotary_emb.inv_freq，无法生成连续 RoPE。")
                inv_freq = rotary_emb.inv_freq.to(q.dtype)  # (head_dim//2,)
                head_dim = q.shape[-1]
                if head_dim % 2 != 0:
                    raise ValueError("[CT-RoPE] head_dim 必须为 2 的倍数。")
                half_dim = head_dim // 2
                if inv_freq.numel() != half_dim:
                    # 某些实现里 inv_freq 可能是 (half_dim,) 或 (1, half_dim)
                    inv_freq = inv_freq.view(-1)
                    if inv_freq.numel() != half_dim:
                        raise ValueError(f"[CT-RoPE] inv_freq 长度 {inv_freq.numel()} 与 half_dim {half_dim} 不匹配")

                # 角度： (bsz, q_len, half_dim)
                angles = t_scaled.to(q.dtype).unsqueeze(-1) * inv_freq.view(1, 1, half_dim)
                cos = torch.cos(angles).unsqueeze(1)  # (bsz, 1, q_len, half_dim)
                sin = torch.sin(angles).unsqueeze(1)  # (bsz, 1, q_len, half_dim)

                q = _apply_rotary_pos_emb_continuous(q, cos, sin)
                k = _apply_rotary_pos_emb_continuous(k, cos, sin)

                if past_key_value is not None:
                    k = torch.cat([past_key_value[0], k], dim=2)
                    v = torch.cat([past_key_value[1], v], dim=2)

                if num_kv_heads != num_query_heads:
                    assert num_query_heads % num_kv_heads == 0, "query_heads 必须是 kv_heads 的整数倍"
                    k = repeat_kv(k, num_query_heads // num_kv_heads)
                    v = repeat_kv(v, num_query_heads // num_kv_heads)

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
                attn_output = attn_output.transpose(1, 2).contiguous().view(
                    bsz, q_len, num_query_heads * head_dim
                )
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
                if verbose and self.debug_flag:
                    print(f"[CT-RoPE] patched layer {i}")
        return core_model

    def forecasting(self, batch_dict, n_vars_to_predict=None):
        observed_data = batch_dict["observed_data"].to(self.device).float()
        observed_mask = batch_dict["observed_mask"].to(self.device).float()
        observed_tp_raw = batch_dict["observed_tp"].to(self.device).float()
        tp_to_predict = batch_dict["tp_to_predict"].to(self.device).float()

        B, L_obs, D = observed_data.shape
        if L_obs == 0:
            raise ValueError("Empty observed timeline: L_obs == 0（该样本无历史观测，强制报错以定位数据问题）")

        # 检查时间张量非空
        if observed_tp_raw.dim() == 2 and observed_tp_raw.shape[1] == 0:
            raise ValueError("observed_tp_raw shape[1] == 0 (无时间戳)")
        if observed_tp_raw.dim() == 3 and observed_tp_raw.shape[-1] == 0:
            raise ValueError("observed_tp_raw last dim == 0 (无时间戳)")

        if observed_tp_raw.dim() == 3:
            tt_bd = observed_tp_raw
            if tt_bd.shape[2] not in (D, 1):
                raise ValueError(f"observed_tp_raw 3D 第二个维度 {tt_bd.shape[2]} 既不是 D={D} 也不是 1")
            if tt_bd.shape[2] == 1:
                tt_bd = tt_bd.repeat(1, 1, D)
        else:
            tt_bd = observed_tp_raw.unsqueeze(-1).expand(-1, -1, D)

        var_ids = torch.arange(D, device=self.device)
        var_prompt = self.var_emb(var_ids).unsqueeze(0).unsqueeze(1).expand(B, 1, D, -1)

        x = observed_data.unsqueeze(-1)
        m = observed_mask.unsqueeze(-1)
        val_in = torch.cat([x, m], dim=-1)
        val_emb = self.value_emb(val_in)

        tokens = torch.cat([var_prompt, val_emb], dim=1)  # (B,L_obs+1,D,H)
        seq_len = tokens.shape[1]
        tokens_flat = tokens.permute(0, 2, 1, 3).reshape(B * D, seq_len, self.d_model)

        tp_for_each_var = tt_bd.permute(0, 2, 1)  # (B,D,L_obs)
        t_min = tp_for_each_var.min(dim=-1, keepdim=True).values
        tp_with_prompt = torch.cat([t_min, tp_for_each_var], dim=-1)  # (B,D,L_obs+1)
        timestamps = tp_with_prompt.reshape(B * D, -1)
        # 强制要求 timestamps 长度至少 2（prompt + 至少一个观测）
        if timestamps.shape[1] < 2:
            raise ValueError(f"构造出的 timestamps 长度 {timestamps.shape[1]} < 2，不符合 CT-RoPE 要求。")

        # 通过属性向注意力层传递连续时间戳（避免多层 **kwargs 传递失败）
        if self.enable_ct_rope:
            setattr(self.qwen, "_ct_timestamps", timestamps)
        try:
            out_time = self.qwen(
                inputs_embeds=tokens_flat,
                use_cache=False,
                output_attentions=False
            ).last_hidden_state
        finally:
            if hasattr(self.qwen, "_ct_timestamps"):
                delattr(self.qwen, "_ct_timestamps")

        obs_mask_flat = observed_mask.permute(0, 2, 1).reshape(B * D, L_obs, 1)
        mask_with_prompt = torch.cat(
            [torch.ones(B * D, 1, 1, device=self.device), obs_mask_flat], dim=1
        )
        denom = mask_with_prompt.sum(dim=1) + 1e-8
        pooled = (out_time * mask_with_prompt).sum(dim=1) / denom
        pooled = self.ln_proj(pooled).view(B, D, -1)

        var_in = pooled + self.var_emb.weight.unsqueeze(0).expand(B, D, -1)
        var_in_bert = self.proj_to_bert(var_in)
        bert_out = self.bert(inputs_embeds=var_in_bert).last_hidden_state
        out_var = self.proj_from_bert(bert_out)

        L_pred = tp_to_predict.shape[1]
        t_future = tp_to_predict.unsqueeze(-1)
        var_rep = out_var.unsqueeze(1).expand(-1, L_pred, -1, -1)
        time_rep = t_future.unsqueeze(2).expand(-1, -1, D, -1)
        fused = torch.cat([var_rep, time_rep], dim=-1)
        pred = self.var_time_mlp(fused).squeeze(-1)

        if n_vars_to_predict is not None:
            pred = pred[:, :, :n_vars_to_predict]
        return pred.float()

    def forward(self, batch_dict, n_vars_to_predict=None):
        return self.forecasting(batch_dict, n_vars_to_predict)
