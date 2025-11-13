import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
from peft import get_peft_model, LoraConfig, TaskType


class TimeEmbedding(nn.Module):
    # 与原仓库 models/embed.py 一致：线性 + 周期（sin）
    def __init__(self, d_model: int):
        super().__init__()
        self.periodic = nn.Linear(1, d_model - 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, tt: torch.Tensor) -> torch.Tensor:
        # tt: (..., 1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], dim=-1)


class ValueEmbedding(nn.Module):
    # 与原仓库一致：将 [value, mask] 两通道投影到 d_model
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


class istsplm_forecast(nn.Module):
    """
    Hybrid: 时间分支 Qwen2.5（decoder-only），变量分支 BERT（双向，且禁用位置编码），
    时间分支输入与原 ISTS-PLM 对齐：
      token = x_mark * TimeEmbedding(tt) + ValueEmbedding([x, x_mark])
      并在每个变量序列前插入 variable prompt token。

    训练策略：
      - 默认完全冻结两大 PLM，仅训练小头/嵌入
      - --semi_freeze 时仅开放 PLM 中 LayerNorm
      - --use_lora 只对 Qwen 注入 LoRA（FEATURE_EXTRACTION）
      - --use_lora_bert 可选对 BERT 注入 LoRA（默认关）
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.d_model = args.d_model
        self.num_types = args.num_types
        self.use_lora = getattr(args, "use_lora", False)
        self.use_lora_bert = getattr(args, "use_lora_bert", False)
        self.semi_freeze = getattr(args, "semi_freeze", False)

        self.qwen_path = args.plm_path
        self.bert_path = getattr(args, "bert_path", "bert-base-uncased")

        # 与原版一致的时间/数值/变量嵌入
        self.time_emb = TimeEmbedding(self.d_model)
        self.value_emb = ValueEmbedding(2, self.d_model)
        self.var_emb = nn.Embedding(self.num_types, self.d_model)

        # Qwen2.5 (时间分支)
        self.qwen_config = AutoConfig.from_pretrained(self.qwen_path, trust_remote_code=True)
        self.qwen_config.use_cache = False
        qwen_base = AutoModelForCausalLM.from_pretrained(
            self.qwen_path, trust_remote_code=True, torch_dtype=torch.float32
        )
        # 取 backbone（不需要 LM 头）
        self.qwen = getattr(qwen_base, "model", qwen_base)

        # BERT (变量分支)
        self.bert = AutoModel.from_pretrained(self.bert_path, torch_dtype=torch.float32)
        self.bert_hidden = self.bert.config.hidden_size

        # 将 BERT 的位置/segment 嵌入置零并冻结（近似 _wope）
        self._disable_bert_positional(self.bert)

        # 维度桥接
        if self.bert_hidden != self.d_model:
            self.proj_to_bert = nn.Linear(self.d_model, self.bert_hidden)
            self.proj_from_bert = nn.Linear(self.bert_hidden, self.d_model)
        else:
            self.proj_to_bert = nn.Identity()
            self.proj_from_bert = nn.Identity()

        # 冻结/LoRA 策略
        self._apply_freeze_policies()

        # 时间分支输出的池化后再做 LN
        self.ln_proj = nn.LayerNorm(self.d_model)

        # 预测头：与原版相同，用原始未来时间标量拼接
        self.var_time_mlp = nn.Sequential(
            nn.Linear(self.d_model + 1, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 1)
        )

    # --------- BERT 置零位置/segment 嵌入，近似 w/o positional embedding ----------
    @staticmethod
    def _disable_bert_positional(bert_model: AutoModel):
        emb = bert_model.embeddings
        if hasattr(emb, "position_embeddings"):
            with torch.no_grad():
                emb.position_embeddings.weight.zero_()
            emb.position_embeddings.weight.requires_grad = False
        if hasattr(emb, "token_type_embeddings"):
            with torch.no_grad():
                emb.token_type_embeddings.weight.zero_()
            emb.token_type_embeddings.weight.requires_grad = False

    # ---------------- Freeze / LoRA ----------------
    def _apply_lora_qwen(self, args, model):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=target_modules, lora_dropout=args.lora_dropout,
            bias="none", task_type=TaskType.FEATURE_EXTRACTION  # 非生成任务
        )
        model = get_peft_model(model, lora_cfg)
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
        return model

    def _apply_lora_bert(self, args, model):
        target_modules = ["query", "key", "value", "dense"]
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=target_modules, lora_dropout=args.lora_dropout,
            bias="none", task_type=TaskType.FEATURE_EXTRACTION
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
        else:
            self._freeze_or_semi(self.qwen, semi=self.semi_freeze)

        if self.use_lora_bert:
            self.bert = self._apply_lora_bert(self.args, self.bert)
        else:
            self._freeze_or_semi(self.bert, semi=self.semi_freeze)

    # ---------------- Forecasting ----------------
    def forecasting(self, batch_dict, n_vars_to_predict=None):
        # 输入
        observed_data = batch_dict["observed_data"].to(self.device).float()   # (B, L_obs, D)
        observed_mask = batch_dict["observed_mask"].to(self.device).float()   # (B, L_obs, D)
        observed_tp_raw = batch_dict["observed_tp"].to(self.device).float()   # (B, L_obs) 或 (B, 1, L_obs)
        tp_to_predict = batch_dict["tp_to_predict"].to(self.device).float()   # (B, L_pred)

        B, L_obs, D = observed_data.shape
        L_pred = tp_to_predict.shape[1]

        # 1) 构造时间分支 token（严格对齐原版）
        # 扩展时间到 (B, L_obs, D)
        if observed_tp_raw.dim() == 3:
            tt_bd = observed_tp_raw
        else:
            tt_bd = observed_tp_raw.unsqueeze(-1).expand(-1, -1, D)  # (B, L_obs, D)

        # TimeEmbedding(tt) 与 ValueEmbedding([x, x_mark])
        time_tok = self.time_emb(tt_bd.unsqueeze(-1))                     # (B, L_obs, D, H)
        x = observed_data.unsqueeze(-1)                                   # (B, L_obs, D, 1)
        x_mark = observed_mask.unsqueeze(-1)                               # (B, L_obs, D, 1)
        val_tok = self.value_emb(torch.cat([x, x_mark], dim=-1))           # (B, L_obs, D, H)
        tokens = x_mark * time_tok + val_tok                               # (B, L_obs, D, H)

        # 变量提示（variable prompt）作为首 token
        var_ids = torch.arange(D, device=self.device)                      # (D,)
        var_prompt = self.var_emb(var_ids).unsqueeze(0).unsqueeze(1)       # (1, 1, D, H)
        var_prompt = var_prompt.expand(B, 1, D, -1)                        # (B, 1, D, H)

        tokens = torch.cat([var_prompt, tokens], dim=1)                    # (B, L_obs+1, D, H)
        tokens_flat = tokens.permute(0, 2, 1, 3).reshape(B * D, L_obs + 1, self.d_model)

        # Qwen 前向（不再用基于观测的 attention mask；也不传 timestamps，避免“双重时间编码”冲突）
        out_time = self.qwen(
            inputs_embeds=tokens_flat,
            use_cache=False,
            output_attentions=False
        ).last_hidden_state                                               # (B*D, L_obs+1, H)

        # 仅用于池化的 mask：首 token（prompt）置 1，其余用 observed_mask
        obs_mask_flat = observed_mask.permute(0, 2, 1).reshape(B * D, L_obs, 1)  # (B*D, L_obs, 1)
        obs_mask_with_prompt = torch.cat(
            [torch.ones(B * D, 1, 1, device=self.device), obs_mask_flat], dim=1
        )                                                                  # (B*D, L_obs+1, 1)

        n_nonmask = obs_mask_with_prompt.sum(dim=1) + 1e-8                 # (B*D, 1)
        pooled = (out_time * obs_mask_with_prompt).sum(dim=1) / n_nonmask  # (B*D, H)
        pooled = self.ln_proj(pooled).view(B, D, -1)                       # (B, D, H)

        # 2) 变量分支（BERT，近似 w/o pos emb）
        var_in = pooled + self.var_emb.weight.unsqueeze(0).expand(B, D, -1).float()
        var_in_bert = self.proj_to_bert(var_in)                            # (B, D, bert_hidden)

        # BERT 的 attention_mask 全 1，且我们已将 position/token_type 嵌入置零
        bert_out = self.bert(inputs_embeds=var_in_bert).last_hidden_state  # (B, D, bert_hidden)
        out_var = self.proj_from_bert(bert_out)                            # (B, D, H)

        # 3) 预测头：用原始未来时间标量（与原版一致）
        tpred_scalar = tp_to_predict.unsqueeze(-1)                         # (B, L_pred, 1)
        var_rep = out_var.unsqueeze(1).expand(-1, L_pred, -1, -1)         # (B, L_pred, D, H)
        time_rep = tpred_scalar.unsqueeze(2).expand(-1, -1, D, -1)        # (B, L_pred, D, 1)
        fused = torch.cat([var_rep, time_rep], dim=-1)                    # (B, L_pred, D, H+1)

        pred = self.var_time_mlp(fused).squeeze(-1)                       # (B, L_pred, D)

        if n_vars_to_predict is not None:
            pred = pred[:, :, :n_vars_to_predict]
        return pred.float()

    def forward(self, batch_dict, n_vars_to_predict=None):
        return self.forecasting(batch_dict, n_vars_to_predict)
