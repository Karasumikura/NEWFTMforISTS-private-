import torch
import torch.nn as nn
# ... existing code ...
import math
import sys

class PositionalEmbedding(nn.Module):
# ... existing code ...
    def forward(self, x):
        return self.pe[:, :x.size(1)]

# class TimeEmbedding(nn.Module):
# ... (原始 TimeEmbedding 已被CTRoPEInspiredTimeEmbedding取代) ...

class CTRoPEInspiredTimeEmbedding(nn.Module):
    """
    CTRoPE (Continuous-Time Rotary Position Embedding) 风格的时间嵌入。
    
    这个模块实现了 MIRA 论文中 CTRoPE 的核心思想：使用连续时间戳
    来创建一个旋转矩阵 (通过 cos 和 sin)，然后将这个旋转应用到
    值嵌入向量上。这与原始的加性 TimeEmbedding (APE) 不同。
    
    注意：d_model 必须是偶数。
    """
    def __init__(self, d_model):
        super(CTRoPEInspiredTimeEmbedding, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for CTRoPE-style embedding"
        self.d_model = d_model
        
        # 计算逆频率 (inv_freq)
        # (d_model/2)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq_buf', inv_freq)

    def _rotate_half(self, x):
        """
        将向量的后半部分旋转到前半部分（并取反），将前半部分旋转到后半部分。
        (x1, x2, x3, x4) -> (-x3, -x4, x1, x2)
        """
        x_half = x.shape[-1] // 2
        x1 = x[..., :x_half]
        x2 = x[..., x_half:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x_embed, tt):
        """
        应用旋转。
        
        Args:
            x_embed (torch.Tensor): 值嵌入, 形状 (B, L, D, d_model)
            tt (torch.Tensor): 连续时间戳, 形状 (B, L, D)
            
        Returns:
            torch.Tensor: 旋转后的嵌入, 形状 (B, L, D, d_model)
        """
        # tt: (B, L, D) -> (B, L, D, 1)
        tt = tt.unsqueeze(-1) 
        
        # freqs: (B, L, D, 1) @ (1, d_model/2) -> (B, L, D, d_model/2)
        freqs = tt @ self.inv_freq_buf.unsqueeze(0)
        
        # emb: (B, L, D, d_model)
        emb = torch.cat((freqs, freqs), dim=-1) 
        
        # cos_emb 和 sin_emb: (B, L, D, d_model)
        cos_emb = emb.cos()
        sin_emb = emb.sin()

        # 应用旋转: x_rotated = x * cos + rotate_half(x) * sin
        x_rotated = (x_embed * cos_emb) + (self._rotate_half(x_embed) * sin_emb)
        return x_rotated

class VariableEmbedding(nn.Module):
# ... existing code ...
    def forward(self, x):
        x = self.varible_emb(x.long())
# ... existing code ...
class DataEmbedding_ITS_Ind(nn.Module):
# ... existing code ...
    def forward(self, tt, x, x_mark=None):
# ... existing code ...
        x = x.permute(0, 2, 1, 3).reshape(B*D, L, self.d_model) # (B*D, L+1, d_model)
# ... existing code ...
        return self.dropout(x)
    


class DataEmbedding_ITS_Ind_VarPrompt_indicator(nn.Module):
# ... existing code ...
    def forward(self, tt, x, x_mark=None):
# ... existing code ...
        x = x.permute(0, 2, 1, 3).reshape(B*D, L+1, self.d_model) # (B*D, L+1, d_model)
# ... existing code ...
        return self.dropout(x), vars_prompt


class DataEmbedding_ITS_Ind_VarPrompt(nn.Module):
    def __init__(self, c_in, d_model, n_var, device=None, dropout=0.1, use_te=True):
        super(DataEmbedding_ITS_Ind_VarPrompt, self).__init__()

        self.d_model = d_model
        
        # !!! 关键修改：用 CTRoPEInspiredTimeEmbedding 替换 TimeEmbedding
        # self.time_embedding = TimeEmbedding(d_model=d_model).to(device)
        self.time_embedding = CTRoPEInspiredTimeEmbedding(d_model=d_model).to(device)
        
        self.value_embedding = ValueEmbedding(c_in=2, d_model=d_model).to(device)
        self.variable_embedding = VariableEmbedding(n_var=n_var, d_model=d_model).to(device)
        self.vars = torch.arange(n_var).to(device)
        self.device = device
        self.use_te = use_te
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tt, x, x_mark=None):
        """ 
        tt: (B, L, D)
        x: (B, L, D) tensor containing the observed values.
        x_mark: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        B, L, D = x.shape
        
        # (B, L, D, 1)
        x_val = x.unsqueeze(dim=-1) 
        x_mask_val = x_mark.unsqueeze(dim=-1) 
        # (B, L, D, 2)
        x_int = torch.cat([x_val, x_mask_val], dim=-1) 
        # (B, L, D, d_model)
        value_emb = self.value_embedding(x_int) 

        if(self.use_te):
            # !!! 关键修改：应用旋转嵌入
            
            # (B, L, D, 1)
            time_emb_mask = x_mask_val
            
            # (B, L, D, d_model)
            # 将值嵌入和时间戳传入，得到旋转后的嵌入
            rotated_value_emb = self.time_embedding(value_emb, tt)
            
            # 仅在有观测值 (x_mark=1) 的地方使用旋转后的嵌入
            # 在没有观测值 (x_mark=0) 的地方，使用原始的 value_emb
            # 原始 value_emb 已经编码了 (value=0, mask=0) 的信息
            x = torch.where(time_emb_mask.bool(), rotated_value_emb, value_emb)
        else:
            x = value_emb
        
        vars_prompt = self.variable_embedding(self.vars.view(1, 1, -1).repeat(B, 1, 1))
        # text_embedding 
        x = torch.cat([vars_prompt, x], dim=1)
        # print(x.shape)
        x = x.permute(0, 2, 1, 3).reshape(B*D, L+1, self.d_model) # (B*D, L+1, d_model)

        return self.dropout(x), vars_prompt


# vector-based embedding
class DataEmbedding_ITS_Vector(nn.Module):
# ... existing code ...
