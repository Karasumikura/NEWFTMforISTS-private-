# 仅展示增加过滤的 collate 函数段落，其余保持原文件
def USHCN_variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train",
    data_min = None, data_max = None, time_max = None):
    """
    新增与原版一致的空历史过滤，不改变其它行为。
    """
    filtered = []
    dropped = 0
    for item in batch:
        record_id, tt, vals, mask, mask_observed, t_bias = item
        if mask_observed.sum() == 0:
            dropped += 1
            continue
        filtered.append(item)
    if dropped > 0:
        print(f"[USHCN collate] Dropped {dropped} empty-history samples.")
    if len(filtered) == 0:
        raise ValueError("[USHCN collate] All samples empty after filtering.")

    observed_tp = []
    observed_data = []
    observed_mask = []
    predicted_tp = []
    predicted_data = []
    predicted_mask = []

    for b, (record_id, tt, vals, mask, mask_observed, t_bias) in enumerate(filtered):
        tt_shifted = tt + t_bias
        observed_tp.append(tt[mask_observed])
        observed_data.append(vals[mask_observed])
        observed_mask.append(mask[mask_observed])

        mask_predicted = ~mask_observed
        predicted_tp.append(tt[mask_predicted])
        predicted_data.append(vals[mask_predicted])
        predicted_mask.append(mask[mask_predicted])

    observed_tp = pad_sequence(observed_tp, batch_first=True)
    observed_data = pad_sequence(observed_data, batch_first=True)
    observed_mask = pad_sequence(observed_mask, batch_first=True)
    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)

    observed_tp = utils.normalize_masked_tp(observed_tp, att_min=0, att_max=time_max)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min=0, att_max=time_max)

    data_dict = {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask
    }
    return data_dict
