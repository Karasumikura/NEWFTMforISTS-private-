import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

import lib.utils as utils
from lib.generate_timeseries import Periodic_1d
from torch.distributions import uniform

from torch.utils.data import DataLoader
from lib.physionet import *
from lib.person_activity import *
from lib.mimic import *
from lib.ushcn import *
from sklearn import model_selection

#####################################################################################################

def task_mask(args, total_dataset):
    total_dataset_new = []
    for n, (record_id, tt, vals, mask) in enumerate(total_dataset):
        if (args.task == 'forecasting'):
            mask_observed_tp = torch.lt(tt, args.history)
        elif (args.task == 'imputation'):
            mask_observed_tp = torch.ones_like(tt).bool()
            rng = np.random.default_rng(n)
            mask_inds = rng.choice(len(tt), size=int(len(tt) * args.mask_rate), replace=False)
            mask_observed_tp[mask_inds] = False
            if args.dataset in ["physionet"]:
                mask_observed_tp[0] = True  # 某些数据集只有 t=0 有值
        else:
            raise Exception('{}: Wrong task specified!'.format(args.task))
        total_dataset_new.append((record_id, tt, vals, mask, mask_observed_tp))

    return total_dataset_new

def _filter_empty_history(dataset, history, mask_rate):
    """ 删除历史窗口为空的样本（原版 ISTS-PLM 会在样本构造阶段跳过这类样本） """
    filtered = []
    dropped = 0
    for (record_id, tt, vals, mask, mask_observed_tp) in dataset:
        if mask_observed_tp.sum() == 0:
            dropped += 1
            continue
        filtered.append((record_id, tt, vals, mask, mask_observed_tp))
    if dropped > 0:
        print(f"[parse_datasets] Dropped {dropped} samples with empty history (history={history}, mask_rate={mask_rate}).")
    return filtered

def parse_datasets(args, length_stat=False):

    device = args.device
    dataset_name = args.dataset

    ##################################################################
    ### PhysioNet & Mimic dataset ###
    if dataset_name in ["physionet", "mimic"]:

        if dataset_name == "physionet":
            total_dataset = PhysioNet('./data/physionet',
                                      quantization=args.quantization,
                                      download=True,
                                      n_samples=args.n,
                                      device=device)
        elif dataset_name == "mimic":
            total_dataset = MIMIC('./data/mimic/',
                                  n_samples=args.n,
                                  device=device)

        total_dataset = task_mask(args, total_dataset)
        total_dataset = _filter_empty_history(total_dataset, args.history, args.mask_rate)

        # Split
        seen_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                                random_state=42, shuffle=True)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.75,
                                                                random_state=42, shuffle=False)
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
        test_record_ids = [record_id for record_id, tt, vals, mask, mask_observed in test_data]
        print("Test record ids (first 20):", test_record_ids[:20])
        print("Test record ids (last 20):", test_record_ids[-20:])

        record_id, tt, vals, mask, mask_observed = train_data[0]
        n_samples = len(total_dataset)
        input_dim = vals.size(-1)

        batch_size = min(min(len(seen_data), args.batch_size), args.n)
        data_min, data_max, time_max = get_data_min_max(seen_data, device)

        if (args.collate == 'indseq'):
            collate_fn = variable_time_collate_series
        else:
            collate_fn = variable_time_collate_fn

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda b: collate_fn(b, args, device=device,
                                                                       data_type="train",
                                                                       data_min=data_min,
                                                                       data_max=data_max,
                                                                       time_max=time_max))
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                    collate_fn=lambda b: collate_fn(b, args, device=device,
                                                                     data_type="val",
                                                                     data_min=data_min,
                                                                     data_max=data_max,
                                                                     time_max=time_max))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                     collate_fn=lambda b: collate_fn(b, args, device=device,
                                                                      data_type="test",
                                                                      data_min=data_min,
                                                                      data_max=data_max,
                                                                      time_max=time_max))

        max_input_len, max_pred_len, median_len = get_seq_length(args, train_data)

        n_train_batches = len(train_dataloader)
        n_val_batches = len(val_dataloader)
        n_test_batches = len(test_dataloader)

        data_obj = {"train_dataloader": iter(train_dataloader),
                    "val_dataloader": iter(val_dataloader),
                    "test_dataloader": iter(test_dataloader),
                    "n_train_batches": n_train_batches,
                    "n_val_batches": n_val_batches,
                    "n_test_batches": n_test_batches,
                    "input_dim": input_dim,
                    "max_input_len": max_input_len,
                    "max_pred_len": max_pred_len,
                    "median_len": median_len}

        return data_obj

    ##################################################################
    ### Person Activity ###
    if dataset_name in ["activity"]:
        total_dataset = PersonActivity('data/PersonActivity', download=True)
        # 首先进行时间切块
        chunk_data = Activity_time_chunk(total_dataset, args, device)
        # 应用任务 mask
        total_dataset = task_mask(args, chunk_data)
        total_dataset = _filter_empty_history(total_dataset, args.history, args.mask_rate)

        seen_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                                random_state=42, shuffle=True)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.75,
                                                                random_state=42, shuffle=False)

        record_id, tt, vals, mask, mask_observed = train_data[0]
        input_dim = vals.size(-1)
        batch_size = min(min(len(seen_data), args.batch_size), args.n)

        # Activity collate（假设使用 variable_time_collate_fn_activity，如果不是请替换成你当前使用的）
        collate_fn_activity = variable_time_collate_fn_activity

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda b: collate_fn_activity(b, args, device=device,
                                                                               data_type="train"))
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                    collate_fn=lambda b: collate_fn_activity(b, args, device=device,
                                                                             data_type="val"))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                     collate_fn=lambda b: collate_fn_activity(b, args, device=device,
                                                                              data_type="test"))

        max_input_len, max_pred_len, median_len = Activity_get_seq_length(args, train_data)
        n_train_batches = len(train_dataloader)
        n_val_batches = len(val_dataloader)
        n_test_batches = len(test_dataloader)

        data_obj = {"train_dataloader": iter(train_dataloader),
                    "val_dataloader": iter(val_dataloader),
                    "test_dataloader": iter(test_dataloader),
                    "n_train_batches": n_train_batches,
                    "n_val_batches": n_val_batches,
                    "n_test_batches": n_test_batches,
                    "input_dim": input_dim,
                    "max_input_len": max_input_len,
                    "max_pred_len": max_pred_len,
                    "median_len": median_len}
        return data_obj

    ##################################################################
    ### USHCN ###
    if dataset_name in ["ushcn"]:
        total_dataset = USHCN('./data/USHCN', download=True, n_samples=args.n, device=device)
        total_dataset = task_mask(args, total_dataset)
        total_dataset = _filter_empty_history(total_dataset, args.history, args.mask_rate)

        seen_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                                random_state=42, shuffle=True)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.75,
                                                                random_state=42, shuffle=False)

        record_id, tt, vals, mask, mask_observed = train_data[0]
        input_dim = vals.size(-1)
        batch_size = min(min(len(seen_data), args.batch_size), args.n)

        data_min, data_max, time_max = get_data_min_max(seen_data, device)

        collate_fn_ushcn = USHCN_variable_time_collate_fn

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda b: collate_fn_ushcn(b, args, device=device,
                                                                            data_type="train",
                                                                            data_min=data_min,
                                                                            data_max=data_max,
                                                                            time_max=time_max))
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                    collate_fn=lambda b: collate_fn_ushcn(b, args, device=device,
                                                                          data_type="val",
                                                                          data_min=data_min,
                                                                          data_max=data_max,
                                                                          time_max=time_max))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                     collate_fn=lambda b: collate_fn_ushcn(b, args, device=device,
                                                                           data_type="test",
                                                                           data_min=data_min,
                                                                           data_max=data_max,
                                                                           time_max=time_max))

        max_input_len, max_pred_len, median_len = USHCN_get_seq_length(args, train_data)
        n_train_batches = len(train_dataloader)
        n_val_batches = len(val_dataloader)
        n_test_batches = len(test_dataloader)

        data_obj = {"train_dataloader": iter(train_dataloader),
                    "val_dataloader": iter(val_dataloader),
                    "test_dataloader": iter(test_dataloader),
                    "n_train_batches": n_train_batches,
                    "n_val_batches": n_val_batches,
                    "n_test_batches": n_test_batches,
                    "input_dim": input_dim,
                    "max_input_len": max_input_len,
                    "max_pred_len": max_pred_len,
                    "median_len": median_len}
        return data_obj

    raise ValueError(f"Dataset {dataset_name} not supported.")
