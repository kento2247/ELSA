import torch

def si_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    zero_mean: bool = True,
    return_db: bool = True,
) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).

    Args:
        estimate: Tensor of shape (..., T)
        target:   Tensor of shape (..., T)
        eps: numerical stability constant
        zero_mean: if True, subtract mean over time dimension before computing SI-SDR
        return_db: if True, return SI-SDR in dB; else return linear ratio

    Returns:
        si_sdr value(s) with shape (...)
    """
    if estimate.shape != target.shape:
        raise ValueError(f"Shape mismatch: estimate {estimate.shape} vs target {target.shape}")

    estimate = estimate.float()
    target = target.float()

    if zero_mean:
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

    # <estimate, target>
    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    # ||target||^2
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True).clamp_min(eps)

    # alpha = <estimate, target> / ||target||^2
    alpha = dot / target_energy

    # s_target = alpha * target
    s_target = alpha * target
    e_noise = estimate - s_target

    # energies
    s_target_energy = torch.sum(s_target ** 2, dim=-1).clamp_min(eps)
    e_noise_energy = torch.sum(e_noise ** 2, dim=-1).clamp_min(eps)

    ratio = s_target_energy / e_noise_energy
    if return_db:
        return 10.0 * torch.log10(ratio)
    return ratio
