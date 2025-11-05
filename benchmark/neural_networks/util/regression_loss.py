# --------------------------------------------------------
# DILATE: DIstortion Loss with shApe and tImE
# By Vincent Le Guen, Nicolas Thome
# https://github.com/vincent-leguen/DILATE
# ---------------------------------------------------------
import torch
import torch.nn.functional as F
#from . import soft_dtw
#from . import path_soft_dtw 

PI = 3.141592653589793

def amp_loss(outputs, targets):
    #outputs = B, T, 1 --> B, 1, T
    B,_, T = outputs.shape
    fft_size = 1 << (2 * T - 1).bit_length()
    out_fourier = torch.fft.fft(outputs, fft_size, dim = -1)
    tgt_fourier = torch.fft.fft(targets, fft_size, dim = -1)

    out_norm = torch.norm(outputs, dim = -1, keepdim = True)
    tgt_norm = torch.norm(targets, dim = -1, keepdim = True)

    #calculate normalized auto correlation
    auto_corr = torch.fft.ifft(tgt_fourier * tgt_fourier.conj(), dim = -1).real
    auto_corr = torch.cat([auto_corr[...,-(T-1):], auto_corr[...,:T]], dim = -1)
    nac_tgt = auto_corr / (tgt_norm * tgt_norm)

    # calculate cross correlation
    cross_corr = torch.fft.ifft(tgt_fourier * out_fourier.conj(), dim = -1).real
    cross_corr = torch.cat([cross_corr[...,-(T-1):], cross_corr[...,:T]], dim = -1)
    nac_out = cross_corr / (tgt_norm * out_norm)
    
    loss = torch.mean(torch.abs(nac_tgt - nac_out))
    return loss


def ashift_loss(outputs, targets):
    B, _, T = outputs.shape
    return T * torch.mean(torch.abs(1 / T - torch.softmax(outputs - targets, dim = -1)))


def phase_loss(outputs, targets):
    B, _, T = outputs.shape
    out_fourier = torch.fft.fft(outputs, dim = -1)
    tgt_fourier = torch.fft.fft(targets, dim = -1)
    tgt_fourier_sq = (tgt_fourier.real ** 2 + tgt_fourier.imag ** 2)
    mask = (tgt_fourier_sq > (T)).float()
    topk_indices = tgt_fourier_sq.topk(k = int(T**0.5), dim = -1).indices
    mask = mask.scatter_(-1, topk_indices, 1.)
    mask[...,0] = 1.
    mask = torch.where(mask > 0, 1., 0.)
    mask = mask.bool()
    not_mask = (~mask).float()
    not_mask /= torch.mean(not_mask)
    out_fourier_sq = (torch.abs(out_fourier.real) + torch.abs(out_fourier.imag))
    zero_error = torch.abs(out_fourier) * not_mask
    zero_error = torch.where(torch.isnan(zero_error), torch.zeros_like(zero_error), zero_error)
    mask = mask.float()
    mask /= torch.mean(mask)
    ae = torch.abs(out_fourier - tgt_fourier) * mask
    ae = torch.where(torch.isnan(ae), torch.zeros_like(ae), ae)
    phase_loss = (torch.mean(zero_error) + torch.mean(ae)) / (T ** .5)
    return phase_loss


def tildeq_loss(outputs, targets, alpha = .5, gamma = .0, beta = .5):
    #outputs = outputs.permute(0,2,1)
    #targets = targets.permute(0,2,1)
    assert not torch.isnan(outputs).any(), "Nan value detected!"
    assert not torch.isinf(outputs).any(), "Inf value detected!"
    B,_, T = outputs.shape
    l_ashift = ashift_loss(outputs, targets)
    l_amp = amp_loss(outputs, targets)
    l_phase = phase_loss(outputs, targets)
    loss = alpha * l_ashift + (1 - alpha) * l_phase + gamma * l_amp

    assert loss == loss, "Loss Nan!"
    return loss, l_amp, l_phase


def pearson_per_trial(y_pred,
                      y_true,
                      eps= 1e-8):
    """
    Compute Pearson r per trial along dim=1 and return the average.

    Args:
        y_pred: Tensor of shape (B, N)
        y_true: Tensor of shape (B, N)
        eps: Small number for numerical stability.

    Returns:
        mean_r: scalar tensor = mean of per-trial Pearson r
        r_per_trial: tensor of shape (B,) with each trial's r
    """
    # means over time axis (dim=1), keep dims for broadcasting
    mu_pred = y_pred.mean(dim=1, keepdim=True)
    mu_true = y_true.mean(dim=1, keepdim=True)

    # zero-center
    pred_z = y_pred - mu_pred      # (B, N)
    true_z = y_true - mu_true      # (B, N)

    # per-trial covariance and variances
    cov = (pred_z * true_z).sum(dim=1)             # (B,)
    sigma_pred = torch.sqrt((pred_z ** 2).sum(dim=1))  # (B,)
    sigma_true = torch.sqrt((true_z ** 2).sum(dim=1))  # (B,)

    # per-trial Pearson r
    r = cov / (sigma_pred * sigma_true + eps)      # (B,)

    # average across batch
    mean_r = r.mean()
    return mean_r


def simple_regression_loss(
    outputs,
    targets,
    mse_weight = 0.5,
    amp_weight= 0.5,
    check_nan= True):
    """
    Computes a weighted sum of MSE and amplitude loss.

    Args:
        outputs: Tensor of shape (batch, channel, 1).
        targets: Tensor of same shape as outputs.
        amp_loss_fn: Function to compute amplitude loss on permuted tensors.
        mse_weight: Weight for MSE component.
        amp_weight: Weight for amplitude component.
        check_nan: If True, raises if outputs contain NaN or Inf.

    Returns:
        loss: Weighted sum of MSE and amplitude losses.
        amp_loss: Amplitude loss term.
        mse_loss: Mean squared error term.
    """
    if check_nan:
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            raise ValueError("Detected NaN or Inf in outputs")

    # Compute MSE directly without re-creating modules
    mse_loss = F.mse_loss(outputs, targets, reduction='mean')
    #loss_amp = amp_loss(outputs, targets)
    
    loss_cor = 0.5*(1-pearson_per_trial(outputs.squeeze(), targets.squeeze()))
    #comb_loss = 0.8*mse_loss+0.2*loss_cor
    
    loss = mse_weight * mse_loss + amp_weight * loss_cor

    if check_nan and not torch.isfinite(loss):
        raise ValueError("NaN or Inf in computed loss")

    return loss, mse_loss, loss_cor