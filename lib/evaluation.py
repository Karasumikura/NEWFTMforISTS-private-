import gc
import numpy as np
import sklearn as sk
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.utils import get_device

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent


# -------------------- Device Helper --------------------
def to_device_batch(batch_dict, device):
    for k, v in batch_dict.items():
        if torch.is_tensor(v):
            batch_dict[k] = v.to(device)
    return batch_dict


def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices=None):
    n_data_points = mu_2d.size()[-1]
    if n_data_points > 0:
        gaussian = Independent(Normal(loc=mu_2d, scale=obsrv_std.repeat(n_data_points)), 1)
        log_prob = gaussian.log_prob(data_2d)
        log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros([1]).to(get_device(data_2d)).squeeze()
    return log_prob


def poisson_log_likelihood(masked_log_lambdas, masked_data, indices, int_lambdas):
    n_data_points = masked_data.size()[-1]
    if n_data_points > 0:
        log_prob = torch.sum(masked_log_lambdas) - int_lambdas[indices]
    else:
        log_prob = torch.zeros([1]).to(get_device(masked_data)).squeeze()
    return log_prob


def compute_binary_CE_loss(label_predictions, mortality_label):
    mortality_label = mortality_label.reshape(-1)

    if len(label_predictions.size()) == 1:
        label_predictions = label_predictions.unsqueeze(0)

    n_traj_samples = label_predictions.size(0)
    label_predictions = label_predictions.reshape(n_traj_samples, -1)

    idx_not_nan = ~torch.isnan(mortality_label)
    if len(idx_not_nan) == 0.:
        print("All labels are NaNs!")
        ce_loss = torch.Tensor(0.).to(get_device(mortality_label))

    label_predictions = label_predictions[:, idx_not_nan]
    mortality_label = mortality_label[idx_not_nan]

    if torch.sum(mortality_label == 0.) == 0 or torch.sum(mortality_label == 1.) == 0:
        print("Warning: batch has single class. Increase batch size.")

    assert (not torch.isnan(label_predictions).any())
    assert (not torch.isnan(mortality_label).any())

    mortality_label = mortality_label.repeat(n_traj_samples, 1)
    ce_loss = nn.BCEWithLogitsLoss()(label_predictions, mortality_label)
    ce_loss = ce_loss / n_traj_samples
    return ce_loss


def compute_multiclass_CE_loss(label_predictions, true_label, mask):
    if (len(label_predictions.size()) == 3):
        label_predictions = label_predictions.unsqueeze(0)

    n_traj_samples, n_traj, n_tp, n_dims = label_predictions.size()

    true_label = true_label.repeat(n_traj_samples, 1, 1)

    label_predictions = label_predictions.reshape(n_traj_samples * n_traj * n_tp, n_dims)
    true_label = true_label.reshape(n_traj_samples * n_traj * n_tp, n_dims)

    mask = torch.sum(mask, -1) > 0

    pred_mask = mask.repeat(n_dims, 1, 1).permute(1, 2, 0)
    label_mask = mask
    pred_mask = pred_mask.repeat(n_traj_samples, 1, 1, 1)
    label_mask = label_mask.repeat(n_traj_samples, 1, 1, 1)

    pred_mask = pred_mask.reshape(n_traj_samples * n_traj * n_tp, n_dims)
    label_mask = label_mask.reshape(n_traj_samples * n_traj * n_tp, 1)

    if (label_predictions.size(-1) > 1) and (true_label.size(-1) > 1):
        assert (label_predictions.size(-1) == true_label.size(-1))
        _, true_label = true_label.max(-1)

    res = []
    for i in range(true_label.size(0)):
        pred_masked = torch.masked_select(label_predictions[i], pred_mask[i].bool())
        labels = torch.masked_select(true_label[i], label_mask[i].bool())
        pred_masked = pred_masked.reshape(-1, n_dims)
        if (len(labels) == 0):
            continue
        ce_loss = nn.CrossEntropyLoss()(pred_masked, labels.long())
        res.append(ce_loss)

    ce_loss = torch.stack(res, 0).to(get_device(label_predictions))
    ce_loss = torch.mean(ce_loss)
    return ce_loss


def compute_masked_likelihood(mu, data, mask, likelihood_func):
    n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
    res = []
    for i in range(n_traj_samples):
        for k in range(n_traj):
            for j in range(n_dims):
                data_masked = torch.masked_select(data[i, k, :, j], mask[i, k, :, j].bool())
                mu_masked = torch.masked_select(mu[i, k, :, j], mask[i, k, :, j].bool())
                log_prob = likelihood_func(mu_masked, data_masked, indices=(i, k, j))
                res.append(log_prob)
    res = torch.stack(res, 0).to(get_device(data))
    res = res.reshape((n_traj_samples, n_traj, n_dims))
    res = torch.mean(res, -1)
    res = res.transpose(0, 1)
    return res


def masked_gaussian_log_density(mu, data, obsrv_std, mask=None):
    if (len(mu.size()) == 3):
        mu = mu.unsqueeze(0)
    if (len(data.size()) == 2):
        data = data.unsqueeze(0).unsqueeze(2)
    elif (len(data.size()) == 3):
        data = data.unsqueeze(0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
    assert (data.size()[-1] == n_dims)

    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
        res = res.reshape(n_traj_samples, n_traj).transpose(0, 1)
    else:
        func = lambda mu_, data_, indices: gaussian_log_likelihood(mu_, data_, obsrv_std=obsrv_std, indices=indices)
        res = compute_masked_likelihood(mu, data, mask, func)
    return res


def mse(mu, data, indices=None):
    n_data_points = mu.size()[-1]
    if n_data_points > 0:
        mse_val = nn.MSELoss()(mu, data)
    else:
        mse_val = torch.zeros([1]).to(get_device(data)).squeeze()
    return mse_val


def compute_mse(mu, data, mask=None):
    if (len(mu.size()) == 3):
        mu = mu.unsqueeze(0)
    if (len(data.size()) == 2):
        data = data.unsqueeze(0).unsqueeze(2)
    elif (len(data.size()) == 3):
        data = data.unsqueeze(0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
    assert (data.size()[-1] == n_dims)

    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        res = mse(mu_flat, data_flat)
    else:
        res = compute_masked_likelihood(mu, data, mask, mse)
    return res


def compute_poisson_proc_likelihood(truth, pred_y, info, mask=None):
    if mask is None:
        poisson_log_l = torch.sum(info["log_lambda_y"], 2) - info["int_lambda"]
        poisson_log_l = torch.mean(poisson_log_l, -1)
    else:
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        mask_repeated = mask.repeat(pred_y.size(0), 1, 1, 1)
        int_lambda = info["int_lambda"]
        f = lambda log_lam, data, indices: poisson_log_likelihood(log_lam, data, indices, int_lambda)
        poisson_log_l = compute_masked_likelihood(info["log_lambda_y"], truth_repeated, mask_repeated, f)
        poisson_log_l = poisson_log_l.permute(1, 0)
    return poisson_log_l


def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None):
    if len(pred_y.shape) == 3:
        pred_y = pred_y.unsqueeze(dim=0)

    n_traj_samples, n_batch, n_tp, n_dim = pred_y.size()
    truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
    mask = mask.repeat(pred_y.size(0), 1, 1, 1)

    if (func == "MSE"):
        error = ((truth_repeated - pred_y) ** 2) * mask
    elif (func == "MAE"):
        error = torch.abs(truth_repeated - pred_y) * mask
    elif (func == "MAPE"):
        if norm_dict is None:
            mask = (truth_repeated != 0) * mask
            truth_div = truth_repeated + (truth_repeated == 0) * 1e-8
            error = torch.abs(truth_repeated - pred_y) / truth_div * mask
        else:
            data_max = norm_dict["data_max"]
            data_min = norm_dict["data_min"]
            truth_rescale = truth_repeated * (data_max - data_min) + data_min
            pred_y_rescale = pred_y * (data_max - data_min) + data_min
            mask = (truth_rescale != 0) * mask
            truth_rescale_div = truth_rescale + (truth_rescale == 0) * 1e-8
            error = torch.abs(truth_rescale - pred_y_rescale) / truth_rescale_div * mask
    elif (func == "HUBER"):
        delta = 2
        abs_error = torch.abs(truth_repeated - pred_y)
        quadratic = torch.min(abs_error, torch.tensor(delta))
        linear = abs_error - quadratic
        error = 0.5 * quadratic ** 2 + delta * linear
        error = error * mask
    else:
        raise Exception("Error function not specified")

    error_var_sum = error.reshape(-1, n_dim).sum(dim=0)
    mask_count = mask.reshape(-1, n_dim).sum(dim=0)

    if (reduce == "mean"):
        error_var_avg = error_var_sum / (mask_count + 1e-8)
        n_avai_var = torch.count_nonzero(mask_count)
        error_avg = error_var_avg.sum() / n_avai_var
        return error_avg
    elif (reduce == "sum"):
        return error_var_sum, mask_count
    else:
        raise Exception("Reduce argument not specified!")


# ---------------- All Losses (Forecasting) ----------------
def compute_all_losses(model, batch_dict, dataset=None):
    """
    Batch dict will be moved to model device before computation.
    Returns dict with loss, mse, rmse, mae.
    """
    device = next(model.parameters()).device
    batch_dict = to_device_batch(batch_dict, device)

    pred_y = model.forecasting(batch_dict, n_vars_to_predict=None)

    mse_val = compute_error(batch_dict["data_to_predict"], pred_y,
                            mask=batch_dict["mask_predicted_data"],
                            func="MSE", reduce="mean")
    rmse_val = torch.sqrt(mse_val)
    mae_val = compute_error(batch_dict["data_to_predict"], pred_y,
                            mask=batch_dict["mask_predicted_data"],
                            func="MAE", reduce="mean")

    loss = mse_val
    if dataset == 'ushcn':
        loss = compute_error(batch_dict["data_to_predict"], pred_y,
                             mask=batch_dict["mask_predicted_data"],
                             func="HUBER", reduce="mean")

    results = {
        "loss": loss,
        "mse": mse_val.item(),
        "rmse": rmse_val.item(),
        "mae": mae_val.item()
    }
    return results


def evaluation(model, dataloader, n_batches):
    """
    Aggregate metrics over validation/test sets.
    """
    device = next(model.parameters()).device
    n_eval_samples = 0
    n_eval_samples_mape = 0
    total_results = {
        "loss": 0,
        "mse": 0,
        "mae": 0,
        "rmse": 0,
        "mape": 0
    }

    for _ in range(n_batches):
        batch_dict = utils.get_next_batch(dataloader)
        batch_dict = to_device_batch(batch_dict, device)

        pred_y = model.forecasting(batch_dict, n_vars_to_predict=None)

        se_var_sum, mask_count = compute_error(batch_dict["data_to_predict"], pred_y,
                                               mask=batch_dict["mask_predicted_data"],
                                               func="MSE", reduce="sum")
        ae_var_sum, _ = compute_error(batch_dict["data_to_predict"], pred_y,
                                      mask=batch_dict["mask_predicted_data"],
                                      func="MAE", reduce="sum")
        ape_var_sum, mask_count_mape = compute_error(batch_dict["data_to_predict"], pred_y,
                                                     mask=batch_dict["mask_predicted_data"],
                                                     func="MAPE", reduce="sum")

        total_results["loss"] += se_var_sum
        total_results["mse"] += se_var_sum
        total_results["mae"] += ae_var_sum
        total_results["mape"] += ape_var_sum
        n_eval_samples += mask_count
        n_eval_samples_mape += mask_count_mape

    n_avai_var = torch.count_nonzero(n_eval_samples)
    n_avai_var_mape = torch.count_nonzero(n_eval_samples_mape)

    total_results["loss"] = (total_results["loss"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
    total_results["mse"] = (total_results["mse"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
    total_results["mae"] = (total_results["mae"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
    total_results["rmse"] = torch.sqrt(total_results["mse"])
    total_results["mape"] = (total_results["mape"] / (n_eval_samples_mape + 1e-8)).sum() / n_avai_var_mape

    # Convert tensors to plain floats
    for key, var in total_results.items():
        if isinstance(var, torch.Tensor):
            total_results[key] = var.item()

    return total_results
