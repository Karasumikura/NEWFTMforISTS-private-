# 在文件中增加 / 或修改 activity 专用 collate 函数 (假设名为 variable_time_collate_fn_activity)
from torch.nn.utils.rnn import pad_sequence
import lib.utils as utils

def variable_time_collate_fn_activity(batch, args, device = torch.device("cpu"), data_type="train"):
    """
    活动数据集的批次构造 (与 physionet/mimic 风格一致)。
    新增：过滤 mask_observed_tp 全 False 的样本。
    输入 batch: List[(record_id, tt, vals, mask, mask_observed_tp)]
    """
    filtered = []
    dropped = 0
    for item in batch:
        record_id, tt, vals, mask, mask_observed_tp = item
        if mask_observed_tp.sum() == 0:
            dropped += 1
            continue
        filtered.append(item)
    if dropped > 0:
        print(f"[activity collate] Dropped {dropped} empty-history samples.")
    if len(filtered) == 0:
        raise ValueError("[activity collate] All samples empty after filtering.")

    observed_tp = []
    observed_data = []
    observed_mask = []
    predicted_tp = []
    predicted_data = []
    predicted_mask = []

    for (record_id, tt, vals, mask, mask_observed_tp) in filtered:
        observed_tp.append(tt[mask_observed_tp])
        observed_data.append(vals[mask_observed_tp])
        observed_mask.append(mask[mask_observed_tp])

        mask_predicted = ~mask_observed_tp
        predicted_tp.append(tt[mask_predicted])
        predicted_data.append(vals[mask_predicted])
        predicted_mask.append(mask[mask_predicted])

    observed_tp = pad_sequence(observed_tp, batch_first=True)
    observed_data = pad_sequence(observed_data, batch_first=True)
    observed_mask = pad_sequence(observed_mask, batch_first=True)
    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)

    # activity 不一定需要归一化时间；如果之前做过归一化，可保持不变。如需与 physionet 一致，可复用 normalize:
    # observed_tp = utils.normalize_masked_tp(observed_tp, att_min=0, att_max=observed_tp.max())
    # predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min=0, att_max=predicted_tp.max())

    data_dict = {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask
    }
    return data_dict
