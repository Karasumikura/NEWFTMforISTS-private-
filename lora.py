import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

class LoRALayer(nn.Module):
    """
    LoRA层，包装一个nn.Linear层。
    它实现了 h = Wx + (alpha/r) * B(A(x))
    """
    def __init__(
        self, 
        linear_layer: nn.Linear, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float = 0.0
    ):
        super().__init__()
        self.linear = linear_layer
        
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x # Identity function
            
        # LoRA矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))
        
        # 缩放因子
        if r > 0:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = 0.0
            
        # 初始化权重
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 矩阵初始化为0
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        # 原始线性层的前向传播
        h_orig = self.linear(x)
        
        # LoRA 路径
        if self.r > 0:
            h_lora = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
            return h_orig + h_lora
        else:
            return h_orig

    def __repr__(self):
        return (f"{self"
