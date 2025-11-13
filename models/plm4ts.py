import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List

# ==================================================================================
# (!!!) 新增: 导入 Qwen, LoRA 和 Transformers 库
# ==================================================================================
from transformers import AutoConfig, PreTrainedModel, Qwen2Model, Qwen2Config
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

# (!!!) 假设: GPT2Model_wope 和 BertModel_wope 仍然被其他旧模型使用
# from transformers.models.gpt2.modeling_gpt2_wope import GPT2Model_wope
# from transformers.models.bert.modeling_bert_wope import BertModel_wope

# (!!!) 新增: 从 classification.py 中移植过来的嵌入模块
# (或者, 你可以保留 'from models.embed import *' 如果它们在 embed.py 中)

class TimeEmbedding(nn.Module):
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
    def __init__(self, c_in, d_model):
        super(ValueEmbedding, self).__init__()
        self.projection = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.projection(x)

class VariableEmbedding(nn.Module):
    def __init__(self, n_var, d_model):
        super(VariableEmbedding, self).__init__()
        self.varible_emb
