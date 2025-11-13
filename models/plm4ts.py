import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
from peft import get_peft_model, LoraConfig, TaskType


class ValueEmbedding(nn.Module):
    """
    简化版：只把 [value, mask] 两通道投影到 d_model。
    输入形状 (B, L, D, 2) 输出 (B, L, D, d_model)
    """
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


class istsplm_forecast(nn.Module):
    """
    时间分支：Qwen2.5 + CT-RoPE（连续时间旋转位置编码）
      - 不再使用原公式 x_mark * TimeEmbedding(tt) + ValueEmbedding([x, x_mark])，只保留 ValueEmbedding([value, mask]) 内容；
      - 时间信息完全通过 CT-RoPE 的 timestamps 张量进入注意力。

    变量分支：BERT (位置嵌入置零模拟 w/o positional embedding)

    预测头：与原版 forecast 模块一致 (concat(hidden_var, raw_future_time) -> MLP -> 标量)

    冻结策略：
      - 默认：全部冻结 PLM
      - --semi_freeze：只放开 LayerNorm
      - --use_lora：Qwen 注入 LoRA (FEATURE_EXTRACTION)，以后可加 --use_lora_bert（此处预留接口）
      - LoRA+semi_freeze：可选放开 Qwen 中 LayerNorm（已实现）

    CT-RoPE 归一化：
      - 在 attention patch 内对每个 (B*D) 序列做 min-max 归一化后乘以 time_scale（可学习）。
      - 若无历史 (L_obs == 0) 则只剩 variable prompt，一个退化场景：timestamps 全 0，不会崩溃。
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.d_model = args.d_model
        self.num_types = args.num_types

        # Flags
        self.use_lora = getattr(args, "use_lora", False)
        self.use_lora_bert = getattr(args, "use_lora_bert", False)
        self.semi_freeze = getattr(args, "semi_freeze", False)

        self.qwen_path = args.plm_path
        self.bert_path = getattr(args, "bert_path", "bert-base-uncased")

        # Embeddings
        self.value_emb = ValueEmbedding(2, self.d_model)   # [value, mask]
        self.var_emb = nn.Embedding(self.num_types, self.d_model)

        # 可学习时间尺度（影响 CT-RoPE 中归一化后放大/缩小角度）
        self.time_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # 加载 Qwen2.5 Backbone
        self.qwen_config = AutoConfig.from_pretrained(self.qwen_path, trust_remote_code=True)
        self.qwen_config.use_cache = False
        qwen_base = AutoModelForCausalLM.from_pretrained(
            self.qwen_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        self.qwen = getattr(qwen_base, "model", qwen_base)
        # 注入 CT-RoPE
        self.qwen = self._inject_ctr_rope(self.qwen, verbose=False)

        # 加载 BERT 变量分支
        self.bert = AutoModel.from_pretrained(self.bert_path, torch_dtype=torch.float32)
        self._disable_bert_position_embeddings(self.bert)
        self.bert_hidden = self.bert.config.hidden_size

        # 层数裁剪（与原版可配置 n_st_plmlayer 对齐）
        if hasattr(args, "n_st_plmlayer"):
            self.bert.encoder.layer = self.bert.encoder.layer[:args.n_st_plmlayer]

        # 维度桥接
        if self.bert_hidden != self.d_model:
            self.proj_to_bert = nn.Linear(self.d_model, self.bert_hidden)
            self.proj_from_bert = nn.Linear(self.bert_hidden, self.d_model)
        else:
            self.proj_to_bert = nn.Identity()
            self.proj_from_bert = nn.Identity()

        # 冻结策略 + LoRA
        self._apply_freeze_policies()

        self.ln_proj = nn.LayerNorm(self.d_model)

        # 预测头 (与原版 forecast 相同结构)
        self.var_time_mlp = nn.Sequential(
            nn.Linear(self.d_model + 1, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 1)
        )

    # ------------------- BERT 去位置嵌入 -------------------
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

    # ------------------- LoRA 注入 -------------------
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
        # Qwen
        if self.use_lora:
            self.qwen = self._apply_lora_qwen(self.args, self.qwen)
            # LoRA + semi_freeze：可选择开放 LN
            if self.semi_freeze:
                for n, p in self.qwen.named_parameters():
                    if ("layernorm" in n.lower() or "norm" in n.lower()) and "lora" not in n:
                        p.requires_grad = True
        else:
            self._freeze_or_semi(self.qwen, semi=self.semi_freeze)

        # BERT
        if self.use_lora_bert:
            self.bert = self._apply_lora_bert(self.args, self.bert)
            if self.semi_freeze:
                for n, p in self.bert.named_parameters():
                    if ("layernorm" in n.lower() or "norm" in n.lower()) and "lora" not in n:
                        p.requires_grad = True
        else:
            self._freeze_or_semi(self.bert, semi=self.semi_freeze)

    # ------------------- CT-RoPE 注入 -------------------
    def _inject_ctr_rope(self, core_model, verbose: bool = True):
        if not hasattr(core_model, "layers"):
            raise ValueError("Qwen2.5 backbone缺少 layers 属性。")
        layers = core_model.layers

        try:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        except Exception:
            raise RuntimeError("无法导入 apply_rotary_pos_emb，请升级 transformers 以支持 Qwen2.5。")

        def wrap(old_attn):
            q_proj = getattr(old_attn, "q_proj", None)
            k_proj = getattr(old_attn, "k_proj", None)
            v_proj = getattr(old_attn, "v_proj", None)
            o_proj = getattr(old_attn, "o_proj", None)
            if any(x is None for x in [q_proj, k_proj, v_proj, o_proj]):
                raise ValueError("注意力层缺少 q/k/v/o 投影。")

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

                # 连续时间 RoPE
                if rotary_emb is not None and timestamps is not None and timestamps.shape[-1] > 0:
                    tmin = timestamps.min(dim=-1, keepdim=True).values
                    tmax = timestamps.max(dim=-1, keepdim=True).values
                    span = torch.clamp(tmax - tmin, min=1e-8)
                    t_norm = (timestamps - tmin) / span
                    t_scaled = t_norm * self.time_scale  # (B*D, seq_len)
                    cos, sin = rotary_emb(v, seq_len=kv_seq_len)  # 原实现需要 cos/sin 按最大长度预取
                    ts = t_scaled.unsqueeze(1).unsqueeze(-1).to(k.dtype)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, ts)
                elif rotary_emb is not None and position_ids is not None:
                    # 回退到离散位置（不推荐，但防御）
                    cos, sin = rotary_emb(v, seq_len=kv_seq_len)
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
                    print(f"[CT-RoPE] patched layer {i}")
        return core_model

    # ------------------- 前向预测 -------------------
    def forecasting(self, batch_dict, n_vars_to_predict=None):
        observed_data = batch_dict["observed_data"].to(self.device).float()     # (B, L_obs, D)
        observed_mask = batch_dict["observed_mask"].to(self.device).float()     # (B, L_obs, D)
        observed_tp_raw = batch_dict["observed_tp"].to(self.device).float()     # (B, L_obs) 或 (B, 1, L_obs)
        tp_to_predict = batch_dict["tp_to_predict"].to(self.device).float()     # (B, L_pred)

        B, L_obs, D = observed_data.shape
        L_pred = tp_to_predict.shape[1]

        # ---- 时间分支 token 构造 ----
        # 扩展时间 (B, L_obs, D)
        if observed_tp_raw.dim() == 3:
            tt_bd = observed_tp_raw
        else:
            tt_bd = observed_tp_raw.unsqueeze(-1).expand(-1, -1, D)

        # variable prompt (B, 1, D, H)
        var_ids = torch.arange(D, device=self.device)
        var_prompt = self.var_emb(var_ids).unsqueeze(0).unsqueeze(1).expand(B, 1, D, -1)

        # value + mask (B, L_obs, D, 2) -> embed
        x = observed_data.unsqueeze(-1)
        m = observed_mask.unsqueeze(-1)
        val_in = torch.cat([x, m], dim=-1)
        val_emb = self.value_emb(val_in)                                      # (B, L_obs, D, H)

        tokens = torch.cat([var_prompt, val_emb], dim=1)                       # (B, L_obs+1, D, H)
        seq_len = tokens.shape[1]

        # 展平到 (B*D, seq_len, H)
        tokens_flat = tokens.permute(0, 2, 1, 3).reshape(B * D, seq_len, self.d_model)

        # 构造 timestamps：(B*D, seq_len)
        # prompt 时间设为最小时间（或 0），这里设为最小值；若 L_obs=0 则全 0
        if L_obs > 0:
            base_tp = tt_bd                                                   # (B, L_obs, D)
            # 对齐维度 (B, D, L_obs)
            tp_for_each_var = base_tp.permute(0, 2, 1)                        # (B, D, L_obs)
            # 插入 prompt 时间（取该变量序列的最小时间）
            t_min = tp_for_each_var.min(dim=-1, keepdim=True).values          # (B, D, 1)
            tp_with_prompt = torch.cat([t_min, tp_for_each_var], dim=-1)      # (B, D, L_obs+1)
        else:
            tp_with_prompt = torch.zeros(B, D, 1, device=self.device)

        timestamps = tp_with_prompt.reshape(B * D, -1)                        # (B*D, seq_len)

        # Attention mask：不屏蔽未观测（保持与原版理念一致），仅可选保留 causal mask (Qwen 内部已有)
        out_time = self.qwen(
            inputs_embeds=tokens_flat,
            timestamps=timestamps,              # CT-RoPE 用
            use_cache=False,
            output_attentions=False
        ).last_hidden_state                     # (B*D, seq_len, H)

        # 池化：按 (prompt + 观测mask) 加权平均
        obs_mask_flat = observed_mask.permute(0, 2, 1).reshape(B * D, L_obs, 1)   # (B*D,L_obs,1)
        mask_with_prompt = torch.cat(
            [torch.ones(B * D, 1, 1, device=self.device), obs_mask_flat], dim=1
        )                                                                          # (B*D,L_obs+1,1)
        denom = mask_with_prompt.sum(dim=1) + 1e-8
        pooled = (out_time * mask_with_prompt).sum(dim=1) / denom                  # (B*D,H)
        pooled = self.ln_proj(pooled).view(B, D, -1)                               # (B,D,H)

        # ---- 变量分支 BERT ----
        var_in = pooled + self.var_emb.weight.unsqueeze(0).expand(B, D, -1)
        var_in_bert = self.proj_to_bert(var_in)
        bert_out = self.bert(inputs_embeds=var_in_bert).last_hidden_state          # (B,D,bert_hidden)
        out_var = self.proj_from_bert(bert_out)                                    # (B,D,H)

        # ---- 预测头 ----
        # 原始未来时间标量 (B,L_pred,1)
        t_future = tp_to_predict.unsqueeze(-1)
        var_rep = out_var.unsqueeze(1).expand(-1, L_pred, -1, -1)                  # (B,L_pred,D,H)
        time_rep = t_future.unsqueeze(2).expand(-1, -1, D, -1)                     # (B,L_pred,D,1)
        fused = torch.cat([var_rep, time_rep], dim=-1)                             # (B,L_pred,D,H+1)
        pred = self.var_time_mlp(fused).squeeze(-1)                                # (B,L_pred,D)

        if n_vars_to_predict is not None:
            pred = pred[:, :, :n_vars_to_predict]
        return pred.float()

    def forward(self, batch_dict, n_vars_to_predict=None):
        return self.forecasting(batch_dict, n_vars_to_predict)
