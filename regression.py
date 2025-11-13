import os
import sys
sys.path.append("..")

import time
import argparse
import numpy as np
import pandas as pd
import datetime
from random import SystemRandom
# 确保 plm4ts 导入
from models.plm4ts import *

import torch
import torch.nn as nn
import torch.optim as optim

import lib.utils as utils
# 假设这些库存在
from lib.parse_datasets import parse_datasets
from lib.evaluation import *

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('ITS Forecasting')

parser.add_argument('--state', type=str, default='def')
# (!!!) 注意: 确保你的 --model 参数与 plm4ts.py 中的类名匹配
parser.add_argument('--model', type=str, default='istsplm_forecast', help='select from [istsplm_forecast, istsplm_vector_forecast, etc.]')

parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='../data/') # 例如: ./data/activity/

parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=2)
# num_types 将由 data_obj 自动设置
parser.add_argument('--num_types', type=int, default=35) 

parser.add_argument('--d_model', type=int, default=1536) # (!!!) 修改: 默认值以匹配 Qwen-1.5B
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-4) # (!!!) 修改: LoRA 微调的推荐 LR

parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--log', type=str, default='./logs/')
parser.add_argument('--save_path', type=str, default='./save/')
parser.add_argument('--seed', type=int, default=2025) # (!!!) 修改: 使用 2025
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')
parser.add_argument('--task', type=str, default='nan') # 似乎用于日志

parser.add_argument('--debug_flag', action='store_true')
parser.add_argument('--dp_flag', action='store_true')
parser.add_argument('--load_in_batch', action='store_true')
parser.add_argument('--history', type=int, default=24, help="number of hours (or months for ushcn) as historical window")
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--median_len', type=int, default=50)
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--dataset', type=str, default='activity', help="Dataset to load. e.g. physionet, mimic, ushcn, activity")
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")

### plm4ists
parser.add_argument('--n_te_plmlayer', type=int, default=6)
parser.add_argument('--n_st_plmlayer', type=int, default=6)
parser.add_argument('--te_model', type=str, default='gpt') # (旧参数，但 Qwen 架构会覆盖它)
parser.add_argument('--st_model', type=str, default='bert') # (旧参数，变量分支默认用 BERT（双向）)
parser.add_argument('--st_plm_type', type=str, default='bert', choices=['bert','qwen'], help='Second-stage PLM type for variable correlation modeling.')
parser.add_argument('--auto_match_var_plm_dim', action='store_true', help='If using Qwen second stage, automatically set d_model to Qwen hidden size to remove projections.')
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--semi_freeze', action='store_true')
parser.add_argument('--sample_rate', type=float, default=1.0)
parser.add_argument('--mask_rate', type=float, default=0.3)
parser.add_argument('--collate', type=str, default='indseq')

# ==================================================================================
# (!!!) 新增/修改的参数: 用于 Qwen + LoRA
# ==================================================================================
parser.add_argument('--plm_path', type=str, default='Qwen/Qwen2-1.5B-Instruct', help="Path to Qwen2/Qwen2.5 model")
parser.add_argument('--use_lora', action='store_true', help="[ADDED] Enable LoRA for Qwen") 
parser.add_argument('--lora_r', type=int, default=8, help="LoRA rank")
parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha")
parser.add_argument('--lora_dropout', type=float, default=0.1, help="[ADDED] LoRA dropout") 
parser.add_argument('--enable_ct_rope', action='store_true', help='Enable CT-RoPE injection (default true).')
parser.add_argument('--ctrope_norm_mode', type=str, default='minmax', choices=['minmax','none','center'], help='Time normalization mode for CT-RoPE.')
parser.add_argument('--prompt_zero_timestamp', action='store_true', help='Use zero timestamp for variable prompt token.')
parser.add_argument('--no_rotate_prompt', action='store_true', help='Do not apply CT-RoPE rotation to the prompt token.')

# ==================================================================================
# (!!!) 新增参数: 变量分支 BERT 与其 LoRA
# ==================================================================================
parser.add_argument('--bert_path', type=str, default='bert-base-uncased', help="[ADDED] Path to BERT for variable branch")
parser.add_argument('--use_lora_bert', action='store_true', help="[ADDED] Enable LoRA for BERT variable branch")

# ==================================================================================
# (!!!) 新增参数: 用于 ISTS-PLM (plm4ts.py 需要)
# ==================================================================================
parser.add_argument('--patch_len', type=int, default=16, help="[ADDED] Patch length for value embedding") 
parser.add_argument('--prompt_len', type=int, default=10, help="[ADDED] Prompt length") 
# ==================================================================================


args = parser.parse_args()
file_name = os.path.basename(__file__)[:-3]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)

#####################################################################################################


if __name__ == '__main__':

    utils.setup_seed(args.seed)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random()*100000)
    
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)
    
    # ==================================================================================
    # (!!!) 修改: 更新日志路径以包含 LoRA 参数
    # ==================================================================================
    if(args.n < 12000 or args.debug_flag):
        args.state = "debug"
        log_path = f"logs/{args.task}_{args.dataset}_{args.model}_{args.state}.log"
    else:
        plm_name = args.plm_path.split('/')[-1]
        bert_name = args.bert_path.split('/')[-1]
        log_path = (
            f"logs/{args.task}_{args.dataset}_{args.model}_{args.state}"
            f"_plm-{plm_name}_bert-{bert_name}_d{args.d_model}"
            f"_loraQwen-{int(args.use_lora)}_loraBert-{int(args.use_lora_bert)}_lr{args.lr}.log"
        )
    # ==================================================================================
    
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")

    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)

    ##################################################################
    data_obj = parse_datasets(args, length_stat=True)
 
    # (!!!) 关键: 将数据加载器中的 'input_dim' (变量数 D) 赋值给 args.num_types
    args.enc_in = data_obj["input_dim"]
    args.num_types =  data_obj["input_dim"]
    args.input_dim =  data_obj["input_dim"] # (旧模型可能会使用这个，保留它)
 
    args.median_len = data_obj["median_len"]
    args.input_len = data_obj["max_input_len"]
    args.pred_len = data_obj["max_pred_len"]
    
    ### Model Config ###
    if(args.model == 'istsplm_forecast'):
        # (!!!) 确保 'istsplm_forecast' 已在 plm4ts.py 中被定义
        model = istsplm_forecast(args).to(args.device)
    elif(args.model == 'istsplm_vector_forecast'):
        model = istsplm_vector_forecast(args).to(args.device)
    elif(args.model == 'istsplm_set_forecast'):
        model = istsplm_set_forecast(args).to(args.device)
    else:
        raise ValueError(f"Model {args.model} not recognized.")
    
    ### Optimizer ###
    # (!!!) 修改: 仅优化可训练参数 (LoRA 或半冻结下的LN/头)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    num_batches = data_obj["n_train_batches"] # n_sample / batch_size
    print("n_train_batches:", num_batches)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    best_val_mse = np.inf
    test_res = None
    for itr in range(args.epoch):
        st = time.time()

        ### Training ###
        model.train()
        for _ in range(num_batches):
            optimizer.zero_grad()
            # utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            # 假设 compute_all_losses 内部调用 model.forecasting(...)
            train_res = compute_all_losses(model, batch_dict, args.dataset)
            
            if train_res["loss"] is not None:
                train_res["loss"].backward()
                optimizer.step()

        ### Validation ###
        model.eval()
        with torch.no_grad():
            # 假设 evaluation 内部调用 model.forecasting(...)
            val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
            
            ### Testing ###
            if(val_res["mse"] < best_val_mse):
                best_val_mse = val_res["mse"]
                best_iter = itr
                test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
            
            logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
            logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item() if train_res["loss"] is not None else -1))
            logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%"
                .format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
            if(test_res != None):
                logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%"
                    .format(best_iter, test_res["loss"], test_res["mse"],
                         test_res["rmse"], test_res["mae"], test_res["mape"]*100))
            logger.info("Time spent: {:.2f}s".format(time.time()-st))

        if(itr - best_iter >= args.patience):
            print("Exp has been early stopped!")
            sys.exit(0)
