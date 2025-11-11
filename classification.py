import os
import sys
import numpy as np
import argparse
import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# 假设的数据加载 (来自你的文件)
from lib.Dataset_MM import get_PAM_data, get_P12_data, get_P19_data, get_P12_data_zeroshot
# 导入我们修复后的新模型
from models.plm4ts import * from lib.utils import EarlyStopping, log_info, gen_log, setup_seed

# 导入 sklearn 用于评估
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score

eps=1e-7

# ==================================================================================
# 补全的辅助模块 (Classifier)
# ==================================================================================
class Classifier(nn.Module):
    """一个简单的分类头"""
    def __init__(self, d_model, n_classes, dropout=0.1):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model // 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model // 2, n_classes)

    def forward(self, x):
        # 假设 x 的形状是 (B, D*d_model)
        # 我们需要先对 D 维度进行平均池化，或者调整模型输出
        #
        # 假设 ists_plm 的输出是 (B, D*d_model)
        # 我们需要将其 reshape 并池化
        
        # 暂时假设模型的输出已经是 (B, d_model) 或 (B, D*d_model)
        # 如果是 (B, D*d_model)， classification.py 的原始逻辑可能需要调整
        # 假设你的 `ists_plm` 输出 (B, D*d_model)，我们需要一个池化层
        #
        # 你的 ists_plm 输出是 (B, D*d_model)
        # 你的 train_epoch 逻辑中 `out = model(...)` 后面没有池化
        # 这意味着分类器需要处理 (B, D*d_model)
        
        # 调整分类器以匹配 (B, D*d_model) 的输入
        # 我们使用一个简单的投影层
        d_input = x.shape[-1] # (D * d_model)
        self.fc = nn.Linear(d_input, n_classes).to(x.device)
        return self.fc(x)

# ==================================================================================
# 补全的训练/评估函数
# ==================================================================================

def train_epoch(model, training_data, optimizer, pred_loss_func, opt, classifier, scaler):
    """ 训练一个 epoch """
    model.train()
    classifier.train()
    
    total_loss = 0
    total_pred = 0
    total_num = 0
    
    # 你的原始代码中使用了 sampled_training_data
    # 假设 training_data 是 DataLoader
    sampled_training_data = tqdm(training_data, mininterval=2,
                                  desc='  - (Training)   ', leave=False)

    for train_batch in sampled_training_data:
        # (B, L, D)
        observed_data, observed_mask, observed_tp, labels, _ = train_batch
        observed_data = observed_data.to(opt.device).float()
        observed_mask = observed_mask.to(opt.device).float()
        observed_tp = observed_tp.to(opt.device).float()
        labels = labels.to(opt.device).long()
        
        optimizer.zero_grad()

        with autocast(enabled=opt.fp16):
            # out 形状: (B, D*d_model)
            out = model(observed_tp, observed_data, observed_mask, opt) 
            sup_pred = classifier(out) # (B, n_classes)
            
            # (计算损失)
            # 假设 labels (B, 1) or (B,)
            loss = pred_loss_func(sup_pred, labels.squeeze())
        
        if opt.fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(labels)
        total_num += len(labels)
        
    return total_loss / total_num


def eval_epoch(model, validation_data, pred_loss_func, opt, classifier, save_res=False):
    """ 评估一个 epoch """
    model.eval()
    classifier.eval()

    total_loss = 0
    total_num = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for valid_batch in tqdm(validation_data, mininterval=2,
                                desc='  - (Validation) ', leave=False):
            
            observed_data, observed_mask, observed_tp, labels, _ = valid_batch
            observed_data = observed_data.to(opt.device).float()
            observed_mask = observed_mask.to(opt.device).float()
            observed_tp = observed_tp.to(opt.device).float()
            labels = labels.to(opt.device).long()

            with autocast(enabled=opt.fp16):
                # out 形状: (B, D*d_model)
                out = model(observed_tp, observed_data, observed_mask, opt) 
                sup_pred = classifier(out) # (B, n_classes)
                
                loss = pred_loss_func(sup_pred, labels.squeeze())

            total_loss += loss.item() * len(labels)
            total_num += len(labels)
            
            # (收集预测结果)
            if opt.n_classes == 2:
                # 二分类 (AUROC/AUPRC)
                # 使用 softmax 获取概率
                preds_prob = F.softmax(sup_pred, dim=1)[:, 1] # 取 P(class=1)
                all_preds.append(preds_prob.cpu().numpy())
            else:
                # 多分类 (Accuracy)
                preds_label = torch.argmax(sup_pred, dim=1)
                all_preds.append(preds_label.cpu().numpy())
                
            all_labels.append(labels.squeeze().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # (计算指标)
    if opt.n_classes == 2:
        valid_auroc = roc_auc_score(all_labels, all_preds)
        valid_auprc = average_precision_score(all_labels, all_preds)
        
        # (计算 F1/Precision/Recall)
        preds_binary = (all_preds > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds_binary, average='binary')
        acc = accuracy_score(all_labels, preds_binary)
        
        return valid_auroc, valid_auprc, acc, precision, recall, f1, total_loss / total_num
    else:
        # (多分类指标)
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        
        return 0, 0, acc, precision, recall, f1, total_loss / total_num # AUROC/AUPRC 设为0


def run_experiment(model, training_data, validation_data, testing_data, optimizer, scheduler, pred_loss_func, opt, \
                        early_stopping=None, classifier=None, save_path=None):
    """ 完整的实验运行 """
    
    # (检查是否有早停)
    if early_stopping is None:
        print("[Info] No early stopping")
    
    scaler = GradScaler(enabled=opt.fp16)

    for epoch in range(opt.epoch):
        print(f'[ Epoch {epoch} ]')
        
        start = time.time()
        train_loss = train_epoch(model, training_data, optimizer, pred_loss_func, opt, classifier, scaler)
        log_info(opt, 'Train', epoch, 0, auroc=0, auprc=0, start=start, precision=0, recall=0, F1=0, loss=train_loss, save=True)

        if not opt.retrain:
            start = time.time()
            valid_auroc, valid_auprc, valid_acc, valid_precision, valid_recall, valid_F1, valid_loss = eval_epoch(model, validation_data, pred_loss_func, opt, classifier)
            log_info(opt, 'Valid', epoch, valid_acc, auroc=valid_auroc, auprc=valid_auprc, start=start, precision=valid_precision, recall=valid_recall, F1=valid_F1, loss=valid_loss, save=True)
            
            start = time.time()
            test_auroc, test_auprc, test_acc, test_precision, test_recall, test_F1, test_loss = eval_epoch(model, testing_data, pred_loss_func, opt, classifier)
            log_info(opt, 'Test', epoch, test_acc, auroc=test_auroc, auprc=test_auprc, start=start, precision=test_precision, recall=test_recall, F1=test_F1, loss=test_loss, save=True)

            if scheduler is not None:
                scheduler.step(valid_auroc)
                
            if early_stopping is not None:
                # 假设我们根据 valid_auroc 来早停 (越高越好)
                early_stopping(-valid_auroc, model, classifier, epoch=epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
    
    # (加载最佳模型)
    if not opt.retrain and early_stopping is not None:
        print("Loading best model for final testing...")
        model.load_state_dict(early_stopping.best_model_dict)
        classifier.load_state_dict(early_stopping.best_classifier_dict)

    # (最终测试)
    start = time.time()
    test_auroc, test_auprc, test_acc, test_precision, test_recall, test_F1, test_loss = eval_epoch(model, testing_data, pred_loss_func, opt, classifier)
    log_info(opt, 'Final Test', opt.epoch, test_acc, auroc=test_auroc, auprc=test_auprc, start=start, precision=test_precision, recall=test_recall, F1=test_F1, loss=test_loss, save=True)

# ==================================================================================
# 补全的 Main 函数
# ==================================================================================

def main():
    """ Main function. """
    parser = argparse.ArgumentParser()

    # (复制你所有的参数)
    parser.add_argument('--state', type=str, default='def')
    parser.add_argument('--task', type=str, default='PAM', help="[PAM, P12, P19]")
    parser.add_argument('--model', type=str, default='ists_plm')
    parser.add_argument('--input_dim', type=int, default=2) # (Value, Mask)
    parser.add_argument('--log', type=str, default='./logs/')
    parser.add_argument('--seed', type=int, default=2025)
    
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dp_flag', action='store_true')
    parser.add_argument('--semi_freeze', action='store_true')
    
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_classes',  type=int, default=2)

    ### plm4ists (这些参数现在被新的 ists_plm 使用)
    parser.add_argument('--d_model', type=int, default=768, help="d_model of the PLM. 768 for Bert/GPT, 1536 for Qwen2-1.5B")
    parser.add_argument('--n_te_plmlayer', type=int, default=6, help="Num layers for Time-Aware PLM")
    parser.add_argument('--n_st_plmlayer', type=int, default=6, help="Num layers for Variable-Aware PLM")
    
    # (这两个参数在我们的新模型中不再使用，但保留)
    parser.add_argument('--te_model', type=str, default='gpt')
    parser.add_argument('--st_model', type=str, default='bert')
    
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_gument('--zero_shot_age', action='store_true')
    parser.add_argument('--zero_shot_ICU', action='store_true')
    
    # !!! 关键修改：添加 PLM 路径和 LoRA 参数 (来自你的文件)
    parser.add_argument('--plm_path', type=str, default='Qwen/Qwen2-1.5B-Instruct', help="Path to Qwen2 model")
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA rank")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha")

    # dataset
    parser.add_argument('--split', type=str, default='1')
    parser.add_argument('--datapath', type=str, default='./data/')
    parser.add_argument('--data_type', type=str, default='PAM')

    opt = parser.parse_args()
    
    # (设置设备和种子)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    opt.device = torch.device(opt.device)
    setup_seed(opt.seed)
    
    # (加载数据)
    print("[Info] Loading data...")
    if opt.task == 'PAM':
        trainloader, validloader, testloader, opt.num_types, max_len = get_PAM_data(opt, opt.device)
    elif opt.task == 'P12':
        trainloader, validloader, testloader, opt.num_types, max_len = get_P12_data(opt, opt.device)
    elif opt.task == 'P19':
        trainloader, validloader, testloader, opt.num_types, max_len = get_P19_data(opt, opt.device)
    else:
        raise ValueError("Unknown task")
    
    if opt.max_len == -1:
        opt.max_len = max_len

    # ======================================================================
    # !!! 关键: 实例化我们新的 ists_plm 模型 (来自 plm4ts.py) !!!
    # ======================================================================
    print(f"! The backbone model is: {opt.model}")
    # (确保 opt.input_dim=2, opt.num_types 已被数据加载器设置)
    
    # (根据你的 plm4ts.py，opt.model 应该是 'ists_plm')
    if opt.model == 'ists_plm':
        model = ists_plm(opt).to(opt.device)
    else:
        # (这里可以添加 istsplm_vector, istsplm_set 等)
        raise ValueError("Model type not supported in this script")
    
    # (实例化分类器)
    # 我们的 ists_plm 输出是 (B, D*d_model) = (B, num_types * d_model)
    classifier_input_dim = opt.num_types * opt.d_model
    mort_classifier = Classifier(classifier_input_dim, opt.n_classes).to(opt.device)
    
    # (设置优化器)
    para_list = list(model.parameters()) + list(mort_classifier.parameters())
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, para_list), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)
    
    # (设置损失函数)
    pred_loss_func = nn.CrossEntropyLoss(reduction='mean')

    if opt.dp_flag:
        model = nn.DataParallel(model)
        mort_classifier = nn.DataParallel(mort_classifier)

    # (设置日志)
    if opt.debug:
