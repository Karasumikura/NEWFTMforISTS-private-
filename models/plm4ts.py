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
        # 基础 d_model（可能在 auto_match_var_plm_dim 下被调整）
        self.d_model = args.d_model
        self.num_types = args.num_types

        self.use_lora = getattr(args, "use_lora", False)
        self.use_lora_bert = getattr(args, "use_lora_bert", False)
        self.semi_freeze = getattr(args, "semi_freeze", False)
        self.enable_ct_rope = getattr(args, "enable_ct_rope", True)
        self.debug_flag = getattr(args, "debug_flag", False)
        # 新增: 时间归一化/扩展控制参数
        # ctrope_norm_mode: minmax | none | center
        self.ctrope_norm_mode = getattr(args, "ctrope_norm_mode", "minmax")
        # 是否使用 0 作为 prompt 时间戳 (与 OLD 原版变量 prompt 不含时间一致)
        self.prompt_zero_timestamp = getattr(args, "prompt_zero_timestamp", True)
        # 可学习线性时间缩放 (代替简单 time_scale)
        self.time_linear_scale = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))  # [a,b] 用于 a*t + b

        self.qwen_path = args.plm_path
        self.bert_path = getattr(args, "bert_path", "bert-base-uncased")
        self.st_plm_type = getattr(args, "st_plm_type", "bert")  # 'bert' | 'qwen'

        self.qwen_config = AutoConfig.from_pretrained(self.qwen_path, trust_remote_code=True)
        self.qwen_config.use_cache = False
        # 若开启自动对齐并使用 qwen 作为第二阶段，强制 d_model 与 Qwen hidden 对齐
        if getattr(args, 'auto_match_var_plm_dim', False) and getattr(args, 'st_plm_type', 'bert') == 'qwen':
            if self.d_model != self.qwen_config.hidden_size:
                self.d_model = self.qwen_config.hidden_size

        # 初始化依赖 d_model 的嵌入（在可能调整维度之后）
        self.value_emb = ValueEmbedding(2, self.d_model)
        self.var_emb = nn.Embedding(self.num_types, self.d_model)
        # 保留旧的 time_scale 作为额外幅度因子
        self.time_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        qwen_base = AutoModelForCausalLM.from_pretrained(
            self.qwen_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        self.qwen = getattr(qwen_base, "model", qwen_base)
        if self.enable_ct_rope:
            self.qwen = self._inject_ctr_rope(self.qwen, verbose=False)

        # ---------------- Second Stage PLM (variable correlation) ----------------
        if self.st_plm_type == 'bert':
            self.bert = AutoModel.from_pretrained(self.bert_path, torch_dtype=torch.float32)
            self._disable_bert_position_embeddings(self.bert)
            self.bert_hidden = self.bert.config.hidden_size
            if hasattr(args, "n_st_plmlayer"):
                self.bert.encoder.layer = self.bert.encoder.layer[:args.n_st_plmlayer]
            if self.bert_hidden != self.d_model:
                self.proj_to_varplm = nn.Linear(self.d_model, self.bert_hidden)
                self.proj_from_varplm = nn.Linear(self.bert_hidden, self.d_model)
            else:
                self.proj_to_varplm = nn.Identity()
                self.proj_from_varplm = nn.Identity()
            self.variable_plm = self.bert
        elif self.st_plm_type == 'qwen':
            # 创建一个仅用于变量自注意力的 Qwen 子模型 (不注入 CT-RoPE, 需要双向)
            var_config = AutoConfig.from_pretrained(self.qwen_path, trust_remote_code=True)
            var_config.use_cache = False
            qwen_var_base = AutoModelForCausalLM.from_pretrained(
                self.qwen_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            var_core = getattr(qwen_var_base, 'model', qwen_var_base)
            # 截断层数
            if hasattr(args, 'n_st_plmlayer'):
                var_core.layers = var_core.layers[:args.n_st_plmlayer]
            # 注入双向注意力（去掉因果掩码）
            self.variable_plm = self._inject_bidirectional_qwen(var_core)
            self.proj_to_varplm = nn.Identity()  # hidden size 与 d_model 应一致
            self.proj_from_varplm = nn.Identity()
        else:
            raise ValueError(f"Unsupported st_plm_type: {self.st_plm_type}")

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

        if self.st_plm_type == 'bert':
            if self.use_lora_bert:
                self.variable_plm = self._apply_lora_bert(self.args, self.variable_plm)
                if self.semi_freeze:
                    for n, p in self.variable_plm.named_parameters():
                        if ("layernorm" in n.lower() or "norm" in n.lower()) and "lora" not in n:
                            p.requires_grad = True
            else:
                self._freeze_or_semi(self.variable_plm, semi=self.semi_freeze)
        else:  # qwen variable stage follow same freeze policy as time qwen
            if not self.use_lora:
                self._freeze_or_semi(self.variable_plm, semi=self.semi_freeze)
            else:
                # Optionally allow LoRA on variable Qwen (same target modules)
                self.variable_plm = self._apply_lora_qwen(self.args, self.variable_plm)
                if self.semi_freeze:
                    for n, p in self.variable_plm.named_parameters():
                        if ("layernorm" in n.lower() or "norm" in n.lower()) and "lora" not in n:
                            p.requires_grad = True

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

                # 连续时间处理
                if self.ctrope_norm_mode == "minmax":
                    tmin = timestamps.min(dim=-1, keepdim=True).values
                    tmax = timestamps.max(dim=-1, keepdim=True).values
                    span = torch.clamp(tmax - tmin, min=1e-8)
                    t_base = (timestamps - tmin) / span
                elif self.ctrope_norm_mode == "none":
                    t_base = timestamps
                elif self.ctrope_norm_mode == "center":
                    t_mean = timestamps.mean(dim=-1, keepdim=True)
                    t_std = timestamps.std(dim=-1, keepdim=True) + 1e-8
                    t_base = (timestamps - t_mean) / t_std
                else:
                    raise ValueError(f"未知的 ctrope_norm_mode: {self.ctrope_norm_mode}")
                # 线性缩放 + 幅度缩放
                a, b = self.time_linear_scale[0], self.time_linear_scale[1]
                t_scaled = (a * t_base + b) * self.time_scale  # (bsz, q_len)

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

    def _inject_bidirectional_qwen(self, core_model):
        """移除 Qwen 因果掩码，构造双向自注意力 forward，供变量相关性建模使用。"""
        if not hasattr(core_model, 'layers'):
            raise ValueError('variable Qwen core missing layers')
        layers = core_model.layers

        def make_forward(old_attn):
            q_proj = getattr(old_attn, 'q_proj', None)
            k_proj = getattr(old_attn, 'k_proj', None)
            v_proj = getattr(old_attn, 'v_proj', None)
            o_proj = getattr(old_attn, 'o_proj', None)
            attn_dropout = getattr(old_attn, 'attn_dropout', nn.Dropout(0.0))

            def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
                if n_rep == 1:
                    return kv
                bsz, kv_heads, seqlen, hd = kv.shape
                return kv.unsqueeze(2).expand(bsz, kv_heads, n_rep, seqlen, hd).reshape(
                    bsz, kv_heads * n_rep, seqlen, hd
                )

            def bi_forward(
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs,
            ):
                bsz, q_len, hidden_size = hidden_states.shape
                hidden_states = hidden_states.float()
                q = q_proj(hidden_states)
                k = k_proj(hidden_states)
                v = v_proj(hidden_states)

                head_dim_attr = getattr(old_attn, 'head_dim', None)
                if head_dim_attr is not None:
                    head_dim = head_dim_attr
                    num_query_heads = q.shape[-1] // head_dim
                else:
                    tentative_heads = getattr(old_attn, 'num_heads', None)
                    if tentative_heads is None:
                        head_dim = 64
                        num_query_heads = q.shape[-1] // head_dim
                    else:
                        head_dim = q.shape[-1] // tentative_heads
                        num_query_heads = tentative_heads
                num_kv_heads = getattr(old_attn, 'num_key_value_heads', None)
                if num_kv_heads is None:
                    num_kv_heads = num_query_heads
                if q.shape[-1] != num_query_heads * head_dim:
                    raise ValueError('q dim mismatch during bidirectional qwen attention')

                q = q.view(bsz, q_len, num_query_heads, head_dim).transpose(1, 2)
                k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

                if past_key_value is not None:
                    k = torch.cat([past_key_value[0], k], dim=2)
                    v = torch.cat([past_key_value[1], v], dim=2)
                if num_kv_heads != num_query_heads:
                    assert num_query_heads % num_kv_heads == 0
                    k = repeat_kv(k, num_query_heads // num_kv_heads)
                    v = repeat_kv(v, num_query_heads // num_kv_heads)

                past = (k, v) if use_cache else None
                # 双向：不添加下三角因果掩码
                attn_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
                if attention_mask is not None:
                    attn_scores = attn_scores + attention_mask.to(attn_scores.dtype)
                attn_weights = nn.functional.softmax(attn_scores, dim=-1)
                attn_weights = attn_dropout(attn_weights)
                attn_output = torch.matmul(attn_weights, v)
                attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, num_query_heads * head_dim)
                attn_output = o_proj(attn_output)
                attn_ret = attn_weights if output_attentions else None
                if use_cache:
                    return attn_output, attn_ret, past
                return attn_output, attn_ret

            return bi_forward

        for lyr in layers:
            if hasattr(lyr, 'self_attn'):
                lyr.self_attn.forward = make_forward(lyr.self_attn)
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

        # ---------------- Value normalization (optional) ----------------
        norm_mode = getattr(self.args, 'value_norm', 'none')
        # compute variable-wise stats over observed points only
        if norm_mode != 'none':
            # mask shape (B,L_obs,D); create mask for reduction
            mask_obs = observed_mask  # (B,L,D)
            # Avoid division by zero
            count = mask_obs.sum(dim=1)  # (B,D)
            safe_count = torch.clamp(count, min=1.0)
            if norm_mode == 'batch_zscore':
                sum_vals = (observed_data * mask_obs).sum(dim=1)
                mean = sum_vals / safe_count
                var = ((observed_data - mean.unsqueeze(1))**2 * mask_obs).sum(dim=1) / safe_count
                std = torch.sqrt(var + 1e-6)
                # center & scale only observed positions, keep unobserved zeros
                observed_data = (observed_data - mean.unsqueeze(1)) / std.unsqueeze(1)
                norm_ctx = {"mode":"zscore","mean":mean, "std":std}
            elif norm_mode == 'batch_minmax':
                observed_min = torch.where(mask_obs>0, observed_data, torch.full_like(observed_data, float('inf'))).min(dim=1).values
                observed_max = torch.where(mask_obs>0, observed_data, torch.full_like(observed_data, float('-inf'))).max(dim=1).values
                # handle all-missing by fallback to zeros
                observed_min = torch.where(count>0, observed_min, torch.zeros_like(observed_min))
                observed_max = torch.where(count>0, observed_max, torch.ones_like(observed_max))
                span = (observed_max - observed_min).clamp(min=1e-6)
                observed_data = (observed_data - observed_min.unsqueeze(1)) / span.unsqueeze(1)
                norm_ctx = {"mode":"minmax","min":observed_min, "span":span}
        else:
            norm_ctx = None

        x = observed_data.unsqueeze(-1)
        m = observed_mask.unsqueeze(-1)
        val_in = torch.cat([x, m], dim=-1)
        val_emb = self.value_emb(val_in)

        tokens = torch.cat([var_prompt, val_emb], dim=1)  # (B,L_obs+1,D,H)
        seq_len = tokens.shape[1]
        tokens_flat = tokens.permute(0, 2, 1, 3).reshape(B * D, seq_len, self.d_model)

        tp_for_each_var = tt_bd.permute(0, 2, 1)  # (B,D,L_obs)
        if self.prompt_zero_timestamp:
            zero_ts = torch.zeros_like(tp_for_each_var[:, :, :1])
            tp_with_prompt = torch.cat([zero_ts, tp_for_each_var], dim=-1)
        else:
            # 保持与之前实现兼容: 使用各变量最小时间作为 prompt 时间戳
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
        if self.st_plm_type == 'bert':
            var_in_emb = self.proj_to_varplm(var_in)
            var_out = self.variable_plm(inputs_embeds=var_in_emb).last_hidden_state
            out_var = self.proj_from_varplm(var_out)
        else:  # qwen variable stage
            out_var = self.variable_plm(inputs_embeds=var_in).last_hidden_state

        L_pred = tp_to_predict.shape[1]
        # -----------------------------------------------
        # 预测阶段时间归一化与训练阶段一致
        # -----------------------------------------------
        # tp_for_each_var: (B,D,L_obs) 之前构造过，用于统计
        # tp_to_predict: (B,L_pred)
        t_future_raw = tp_to_predict  # (B,L_pred)
        # 展开至 (B,L_pred,D)
        t_future = t_future_raw.unsqueeze(-1).expand(-1, L_pred, D)
        if self.ctrope_norm_mode == 'minmax':
            tmin_hist = tp_for_each_var.min(dim=-1).values  # (B,D)
            tmax_hist = tp_for_each_var.max(dim=-1).values  # (B,D)
            span_hist = torch.clamp(tmax_hist - tmin_hist, min=1e-8)  # (B,D)
            tmin_hist = tmin_hist.unsqueeze(1)  # (B,1,D)
            span_hist = span_hist.unsqueeze(1)  # (B,1,D)
            t_future_norm = (t_future - tmin_hist) / span_hist
        elif self.ctrope_norm_mode == 'center':
            t_mean_hist = tp_for_each_var.mean(dim=-1).values  # (B,D)
            t_std_hist = tp_for_each_var.std(dim=-1).values + 1e-8  # (B,D)
            t_mean_hist = t_mean_hist.unsqueeze(1)
            t_std_hist = t_std_hist.unsqueeze(1)
            t_future_norm = (t_future - t_mean_hist) / t_std_hist
        else:  # 'none'
            t_future_norm = t_future
        # 线性缩放 + 幅度同前
        a, b = self.time_linear_scale[0], self.time_linear_scale[1]
        t_future_scaled = (a * t_future_norm + b) * self.time_scale  # (B,L_pred,D)
        # 若不旋转 prompt，只影响历史的第一 token；未来时间不含 prompt 可直接使用 scaled 值
        var_rep = out_var.unsqueeze(1).expand(-1, L_pred, -1, -1)  # (B,L_pred,D,H)
        time_rep = t_future_scaled.unsqueeze(-1)  # (B,L_pred,D,1)
        fused = torch.cat([var_rep, time_rep], dim=-1)  # (B,L_pred,D,H+1)
        pred = self.var_time_mlp(fused).squeeze(-1)

        # ---------------- Undo normalization for prediction ----------------
        if norm_ctx is not None:
            if norm_ctx["mode"] == "zscore":
                pred = pred * norm_ctx["std"].unsqueeze(1) + norm_ctx["mean"].unsqueeze(1)
            elif norm_ctx["mode"] == "minmax":
                pred = pred * norm_ctx["span"].unsqueeze(1) + norm_ctx["min"].unsqueeze(1)

        if n_vars_to_predict is not None:
            pred = pred[:, :, :n_vars_to_predict]
        return pred.float()

    def forward(self, batch_dict, n_vars_to_predict=None):
        return self.forecasting(batch_dict, n_vars_to_predict)
