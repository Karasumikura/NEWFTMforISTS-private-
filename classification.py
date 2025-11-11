import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# ... existing code ...
import time

import torch
# ... existing code ...
from lib.Dataset_MM import get_PAM_data, get_P12_data, get_P19_data, get_P12_data_zeroshot
from models.plm4ts import *

eps=1e-7
# ... existing code ...
def train_epoch(model, training_data, optimizer, pred_loss_func, opt, classifier, scaler):
# ... existing code ...
    for train_batch in tqdm(sampled_training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
# ... existing code ...
        optimizer.zero_grad()

        out = model(observed_tp, observed_data, observed_mask, opt) # [B,D]
# ... existing code ...
        B, L = observed_mask.size(0), observed_mask.size(1)
        
        optimizer.step()
# ... existing code ...
def eval_epoch(model, validation_data, pred_loss_func, opt, classifier, save_res=False):
# ... existing code ...
            out = model(observed_tp, observed_data, observed_mask, opt) # [B,L,K,D]
            sup_pred = classifier(out)
# ... existing code ...
def run_experiment(model, training_data, validation_data, testing_data, optimizer, scheduler, pred_loss_func, opt, \
                        early_stopping=None, classifier=None, save_path=None):

# ... existing code ...
            if not opt.retrain:
                start = time.time()
# ... existing code ...
                log_info(opt, 'Valid', epoch, valid_acc, auroc=valid_auroc, auprc=valid_auprc, start=start, precision=valid_precision, recall=valid_recall, F1=valid_F1, loss=valid_loss, save=True)
                
                start = time.time()
# ... existing code ...
                if early_stopping is not None:
                    early_stopping(-valid_auroc, model, classifier, epoch=epoch)
# ... existing code ...
def main():
    """ Main function. """
# ... existing code ...
    parser = argparse.ArgumentParser()

    parser.add_argument('--state', type=str, default='def')
# ... existing code ...
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--gpu', type=str, default='0')
# ... existing code ...
    parser.add_argument('--n_classes',  type=int, default=2)

    ### plm4ists
    parser.add_argument('--d_model', type=int, default=768, help="d_model of the PLM. 768 for Bert/GPT, 1536 for Qwen2-1.5B")
    parser.add_argument('--n_te_plmlayer', type=int, default=6)
# ... existing code ...
    parser.add_argument('--te_model', type=str, default='gpt')
    parser.add_argument('--st_model', type=str, default='bert')
    parser.add_argument('--max_len', type=int, default=-1)
# ... existing code ...
    parser.add_argument('--zero_shot_age', action='store_true')
    parser.add_argument('--zero_shot_ICU', action='store_true')
    
    # !!! 关键修改：添加 PLM 路径和 LoRA 参数
    parser.add_argument('--plm_path', type=str, default='Qwen/Qwen2-1.5B-Instruct', help="Path to Qwen3/Qwen2 model")
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA rank")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha")


    # dataset
    parser.add_argument('--split', type=str, default='1')
# ... existing code ...
    if opt.task == 'PAM':
        trainloader, validloader, testloader, opt.num_types, max_len = get_PAM_data(opt, opt.device)
# ... existing code ...
    elif opt.task == 'P19':
        trainloader, validloader, testloader, opt.num_types, max_len = get_P19_data(opt, opt.device)
# ... existing code ...
    print("! The backbone model is:", opt.model)

    para_list = list(model.parameters())
# ... existing code ...
        mort_classifier = Classifier(opt.d_model, opt.n_classes)
    
    para_list += list(mort_classifier.parameters())
# ... existing code ...
    if opt.dp_flag:
        model = nn.DataParallel(model)

    if opt.debug:
# ... existing code ...
        exp_desc = f"{opt.task}_{opt.model}_{opt.state}"
    else:
        # !!! 关键修改：更新日志文件名以包含 LoRA 参数
        plm_name = opt.plm_path.split('/')[-1]
        exp_desc = f"{opt.task}_{opt.model}_{opt.state}_plm-{plm_name}_d{opt.d_model}_lora-r{opt.lora_r}_lr{opt.lr}"
    
    opt.log = f"{opt.log}{exp_desc}.log"
# ... existing code ...
    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))
# ... existing code ...
