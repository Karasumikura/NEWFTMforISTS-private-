import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.Dataset_MM as Dataset_MM

# from transformers.models.gpt2.modeling_gpt2_wope import GPT2Model_wope
# from transformers.models.bert.modeling_bert_wope import BertModel_wope
# !!! 关键修改：导入 AutoModel, AutoConfig 和 lora
from transformers import AutoModel, AutoConfig
import lora

from models.embed import *

class ists_plm(nn.Module):
    
    def __init__(self, opt):
        super(ists_plm, self).__init__()
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        
        # !!! 关键修改：确保 DataEmbedding_ITS_Ind_VarPrompt (embed.py) 已被修改为使用 CTRoPE
        self.enc_embedding = DataEmbedding_ITS_Ind_VarPrompt(self.feat_dim, self.d_model, self.feat_dim, device=opt.device, dropout=opt.dropout)

        self.gpts = nn.ModuleList()
        
        # !!! 关键修改：加载 Qwen3 (Qwen2) 并应用 LoRA
        for i in range(2): # 加载两个PLM，用于双PLM架构
            
            print(f"Loading PLM {i} from: {opt.plm_path}")
            config = AutoConfig.from_pretrained(opt.plm_path, output_attentions=True, output_hidden_states=True, trust_remote_code=True)
            
            # 我们假设 embed.py 中的 CTRoPEInspiredTimeEmbedding 提供了时间信息。
            # Qwen 自己的 RoPE (基于序列顺序) 将按原样工作。
            # 模型将接收到两种信号：
            # 1. 旋转的值嵌入（来自CTRoPE）
            # 2. 内部的序列RoPE（来自Qwen）
            # 这是一种混合方法，但保留了原始的双PLM架构和新的组件。
            plm = AutoModel.from_pretrained(opt.plm_path, config=config, trust_remote_code=True)

            print(f"Applying LoRA (r={opt.lora_r}, alpha={opt.lora_alpha}) to PLM {i}")
            # 目标模块 "q_proj", "k_proj", "v_proj", "o_proj" 适用于 Qwen2
            lora.apply_lora_to_model(plm, r=opt.lora_r, lora_alpha=opt.lora_alpha,
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

            # 冻结所有参数
            for name, param in plm.named_parameters():
                param.requires_grad = False
            
            # 只解冻 LoRA 参数
            lora.mark_lora_as_trainable(plm)

            if(opt.semi_freeze):
                # 原始逻辑：也解冻 LayerNorm
                print(f"Semi-freezing: Unfreezing LayerNorm layers for PLM {i}")
                for name, param in plm.named_parameters():
                    if 'ln' in name or 'LayerNorm' in name or 'norm' in name.lower():
                        param.requires_grad = True
            else:
                print(f"LoRA-only tuning: All non-LoRA params are frozen for PLM {i}")

            self.gpts.append(plm)
            
        self.act = F.gelu
        self.dropout = nn.Dropout(p=opt.dropout)
        self.ln_proj = nn.LayerNorm(self.d_model)
        
    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
        """ 
        observed_tp: (B, L, D)
        observed_data: (B, L, D) tensor containing the observed values.
        observed_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
    
        B, L, D = observed_data.shape
        
        # (B*D, L+1, d_model)
        outputs, var_embedding = self.enc_embedding(observed_tp, observed_data, observed_mask) 
        
        # !!! 关键修改：为 PLM 创建 attention_mask
        B_D, L_1, D_M = outputs.shape
        # (B*D, L+1)
        attn_mask_plm_1 = torch.ones(B_D, L_1, device=outputs.device, dtype=torch.long)
        
        # (B*D, L+1, d_model)
        outputs = self.gpts[0](inputs_embeds=outputs, attention_mask=attn_mask_plm_1).last_hidden_state 
        
        # (B*D, L, 1)
        observed_mask_pooled = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1) 
        # (B*D, L+1, 1)
        observed_mask_pooled = torch.cat([torch.ones_like(observed_mask_pooled[:,:1]), observed_mask_pooled], dim=1) 
        n_nonmask = observed_mask_pooled.sum(dim=1)  # (B*D, 1)
        
        # (B*D, d_model)
        outputs = (outputs * observed_mask_pooled).sum(dim=1) / (n_nonmask + 1e-8)
        # (B, D, d_model)
        outputs = self.ln_proj(outputs.view(B, D, -1))  
        
        # (B, D, d_model)
        outputs = outputs + var_embedding.squeeze()
        
        # !!! 关键修改：为第二个 PLM 创建 attention_mask
        B_D_2, D_2, D_M_2 = outputs.shape
        # (B, D)
        attn_mask_plm_2 = torch.ones(B_D_2, D_2, device=outputs.device, dtype=torch.long)
        
        # (B, D, d_model)
        outputs = self.gpts[1](inputs_embeds=outputs, attention_mask=attn_mask_plm_2).last_hidden_state 
        
        outputs = outputs.view(B, -1)
        return outputs #(B, D*d_model)
    
class istsplm_vector(nn.Module):
# ... existing code ...
    def __init__(self, opt):
        super(istsplm_vector, self).__init__()
# ... existing code ...
        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
# ... existing code ...
    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
# ... existing code ...
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state # (B, L, d_model)
# ... existing code ...
        return outputs
    
class istsplm_set(nn.Module):
# ... existing code ...
    def __init__(self, opt):
        super(istsplm_set, self).__init__()
# ... existing code ...
        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
# ... existing code ...
    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
# ... existing code ...
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state # (B, L, d_model)
# ... existing code ...
        return outputs

class istsplm_forecast(nn.Module):
# ... existing code ...
    def __init__(self, opt):
        super(istsplm_forecast, self).__init__()
# ... existing code ...
        self.gpts = nn.ModuleList()
        for i in range(2):
# ... existing code ...
                elif opt.te_model == 'bert':
                    bert.encoder.layer = bert.encoder.layer[:opt.n_te_plmlayer]
# ... existing code ...
                    self.gpts.append(bert)
            elif(i==1):
                if opt.st_model == 'gpt':
# ... existingG...
                    gpt2.h = gpt2.h[:opt.n_st_plmlayer]
                    self.gpts.append(gpt2)
                elif opt.st_model == 'bert':
# ... existing code ...
                    bert.encoder.layer = bert.encoder.layer[:opt.n_st_plmlayer]
                    self.gpts.append(bert)
# ... existing code ...
    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
# ... existing code ...
        outputs = self.gpts[0](inputs_embeds=outputs).last_hidden_state # (B*D, L+1, d_model)

        observed_mask = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1) # (B*D, L, 1)
# ... existing code ...
        outputs = (outputs * observed_mask).sum(dim=1) / n_nonmask # (B*D, d_model)
        outputs = self.ln_proj(outputs.view(B, D, -1))  # (B, D, d_model)
# ... existing code ...
        outputs = self.gpts[1](inputs_embeds=outputs).last_hidden_state # (B, D, d_model)

        # forcasting
# ... existing code ...
class istsplm_vector_forecast(nn.Module):
# ... existing code ...
    def __init__(self, opt):
        super(istsplm_vector_forecast, self).__init__()
# ... existing code ...
        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
# ... existing code ...
    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
# ... existing code ...
        outputs = self.enc_embedding(observed_tp, observed_data, observed_mask, observed_mask_agg) # (B, L, d_model)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state # (B, L, d_model)
# ... existing code ...
class istsplm_set_forecast(nn.Module):
# ... existing code ...
    def __init__(self, opt):
        super(istsplm_set_forecast, self).__init__()
# ... existing code ...
        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
# ... existing code ...
    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
# ... existing code ...
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state # (B, n_patch, d_model)
# ... existing code ...
