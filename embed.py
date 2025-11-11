import torch
import torch.nn as nn
import math

# ==================================================================================
# 1. 基础嵌入模块 (来自你的参考文件)
# 这些是新架构 (plm4ts.py) 所需的辅助类
# ==================================================================================

class TimeEmbedding(nn.Module):
    """
    原始的可学习时间嵌入 (来自参考文件)
    注意：在新的 CTRoPE 架构中，这个模块主要被 DataEmbedding_ITS_for_CTRoPE 内部使用，
    或者可以作为旧模型的备用。
    """
    def __init__(self, d_model, max_len=5000):
        super(TimeEmbedding, self).__init__()
        self.periodic = nn.Linear(1, d_model - 1)
        self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, time):
        return self.learn_time_embedding(time)

class ValueEmbedding(nn.Module):
    """
    值嵌入 (来自参考文件)
    将 (value, mask) 对投影到 d_model 维度
    """
    def __init__(self, c_in, d_model):
        super(ValueEmbedding, self).__init__()
        self.projection = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.projection(x)

class VariableEmbedding(nn.Module):
    """
    变量嵌入 (来自参考文件)
    为 D 个不同的时间序列变量创建嵌入
    """
    def __init__(self, n_var, d_model):
        super(VariableEmbedding, self).__init__()
        self.varible_emb = nn.Embedding(n_var, d_model)

    def forward(self, x):
        x = self.varible_emb(x.long())
        return x

# ==================================================================================
# 2. 新的数据嵌入层 (来自你的参考文件)
# 这是新架构的核心入口
# ==================================================================================

class DataEmbedding_ITS_for_CTRoPE(nn.Module):
    """
    修改自 DataEmbedding_ITS_Ind_VarPrompt，用于 CTRoPE (来自参考文件)。
    
    关键区别:
    1. 它 *不* 将时间嵌入（Time Embedding）添加到值嵌入（Value Embedding）中。
       位置信息将由 CTRoPE 在 Transformer 内部注入。
    2. 它返回对齐好的时间戳张量 `tt_with_prompt`，以便传递给 Transformer。
    """
    def __init__(self, c_in, d_model, n_var, device=None, dropout=0.1):
        super(DataEmbedding_ITS_for_CTRoPE, self).__init__()
        self.d_model = d_model
        
        # 依赖于上面定义的基础模块
        self.value_embedding = ValueEmbedding(c_in=c_in, d_model=d_model).to(device)
        self.variable_embedding = VariableEmbedding(n_var=n_var, d_model=d_model).to(device)
        
        self.vars = torch.arange(n_var).to(device)
        self.device = device
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tt, x, x_mark=None):
        """
        tt: (B, L, D) - 连续时间戳
        x: (B, L, D) - 观测值
        x_mark: (B, L, D) - 掩码 (1=观测到, 0=未观测)
        """
        B, L, D = x.shape
        
        # --- 准备时间戳 ---
        # 为变量 Prompt 创建一个 "0" 时间戳 (B, 1, D)
        prompt_tt = torch.zeros((B, 1, D), device=self.device, dtype=tt.dtype)
        # (B, L+1, D)
        tt_with_prompt = torch.cat([prompt_tt, tt], dim=1)
        # (B*D, L+1)
        tt_with_prompt_flat = tt_with_prompt.permute(0, 2, 1).reshape(B * D, L + 1)
        
        # --- 准备值嵌入 ---
        # (B, L, D, 1)
        x_val = x.unsqueeze(dim=-1) 
        x_mask_val = x_mark.unsqueeze(dim=-1) 
        # (B, L, D, 2)
        # 假设 c_in=2，输入是 (值, 掩码)
        x_int = torch.cat([x_val, x_mask_val], dim=-1) 
        # (B, L, D, d_model)
        value_emb = self.value_embedding(x_int) 

        # --- 关键修改 ---
        # 不添加位置编码。位置信息将由 CTRoPE 稍后注入。
        x_emb = value_emb 
        
        # (变量 Prompt 逻辑)
        # (B, 1, D, d_model)
        vars_prompt = self.variable_embedding(self.vars.view(1, 1, -1).repeat(B, 1, 1))
        
        # (B, L+1, D, d_model)
        x_emb_with_prompt = torch.cat([vars_prompt, x_emb], dim=1)
        
        # (B*D, L+1, d_model)
        x_flat = x_emb_with_prompt.permute(0, 2, 1, 3).reshape(B * D, L + 1, self.d_model)
    
        # 返回: 扁平化的嵌入，扁平化的时间戳 (用于CTRoPE)，变量嵌入 (用于第二阶段)
        return self.dropout(x_flat), tt_with_prompt_flat, vars_prompt
