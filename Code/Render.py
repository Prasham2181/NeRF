import torch
import torch.nn.functional as F
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_pdf(bins, weights, N_samples, det=False, eps=1e-5):
    """
    Hierarchical sampling using inverse transform sampling.
    
    Args:
        bins (torch.Tensor): Tensor of shape (N_rays, N_bins+1) representing bin edges.
        weights (torch.Tensor): Tensor of shape (N_rays, N_bins) representing importance weights.
        N_samples (int): Number of samples to draw.
        det (bool): Deterministic (linspace) or random sampling.
        eps (float): Small number to prevent division by zero.
    
    Returns:
        samples (torch.Tensor): Tensor of shape (N_rays, N_samples) with samples along the ray.
    """
    # Add eps to weights and normalize to get a PDF.
    weights = weights + eps
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_bins)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_bins)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (N_rays, N_bins+1)

    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=DEVICE)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=DEVICE)

    # Invert CDF.
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_samples, 2)

    # Gather cdf and bin values.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), -1, inds_g)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples

def render_scene(neural_field, origins, directions, tn=2.0, tf=6.0, samples=192,
                 hierarchical=False, N_importance=64, clear_bg=False, debug=False):
    """
    Render the scene by querying the neural field along each ray.
    
    This function performs stratified sampling along each ray (the coarse pass). If hierarchical
    sampling is enabled, it uses the coarse weights to sample additional fine points (the fine pass)
    and then recomputes the rendered color using all samples.
    
    Args:
        neural_field: The trained NeRF model.
        origins (torch.Tensor): (N, 3) ray origins.
        directions (torch.Tensor): (N, 3) ray directions.
        tn (float): Near bound.
        tf (float): Far bound.
        samples (int): Number of coarse samples per ray.
        hierarchical (bool): Whether to perform hierarchical (importance) sampling.
        N_importance (int): Number of additional fine samples to draw.
        clear_bg (bool): Whether to composite over a background.
        debug (bool): If True, prints debug information.
    
    Returns:
        rendered_rgb (torch.Tensor): (N, 3) final rendered colors in [-1,1].
    """
    N = origins.shape[0]  # Number of rays

    # --- Coarse Sampling (Stratified Sampling) ---
    t_vals = torch.linspace(tn, tf, samples, device=DEVICE)  # (samples,)
    t_vals = t_vals.expand(N, samples)  # (N, samples)
    
    # Stratified sampling: add randomness within each interval.
    mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
    lower = torch.cat([t_vals[:, :1], mids], dim=-1)
    upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
    t_rand = torch.rand(t_vals.shape, device=DEVICE)
    sampled_t = lower + (upper - lower) * t_rand  # (N, samples)
    
    # Compute coarse 3D points.
    coarse_points = origins.unsqueeze(1) + sampled_t.unsqueeze(-1) * directions.unsqueeze(1)
    coarse_points_flat = coarse_points.reshape(-1, 3)
    dirs_expanded = directions.unsqueeze(1).expand(-1, samples, -1).reshape(-1, 3)
    
    # Query the neural field with the coarse points.
    coarse_rgb, coarse_sigma = neural_field(coarse_points_flat, dirs_expanded)
    if coarse_rgb is None or coarse_sigma is None:
        raise ValueError("Neural field returned None for coarse predictions.")
    
    coarse_rgb = coarse_rgb.view(N, samples, 3)
    coarse_sigma = coarse_sigma.view(N, samples)
    
    # Volume rendering for the coarse pass.
    deltas = sampled_t[:, 1:] - sampled_t[:, :-1]
    delta_inf = torch.full((N, 1), 1e10, device=DEVICE)
    deltas = torch.cat([deltas, delta_inf], dim=-1)  # (N, samples)
    alphas = 1.0 - torch.exp(-coarse_sigma * deltas)
    ones = torch.ones((N, 1), device=DEVICE)
    T = torch.cumprod(torch.cat([ones, 1.0 - alphas + 1e-10], dim=-1), dim=-1)[:, :-1]
    weights = T * alphas  # (N, samples)
    coarse_rendered = torch.sum(weights.unsqueeze(-1) * coarse_rgb, dim=1)  # (N, 3)
    
    if not hierarchical:
        if clear_bg:
            total_weight = weights.sum(dim=-1, keepdim=True)
            coarse_rendered = coarse_rendered + (1.0 - total_weight)
        if debug:
            print("Coarse pass: Sigma mean:", coarse_sigma.mean().item(),
                  "Weights sum mean:", weights.sum(dim=-1).mean().item())
        return coarse_rendered

    # --- Hierarchical (Fine) Sampling ---
    # Create bins for sampling: use midpoints of the coarse samples.
    bins = 0.5 * (sampled_t[:, :-1] + sampled_t[:, 1:])  # shape: (N, samples-1)
    # Use coarse weights (excluding the first and last) to form a PDF.
    # (Often, one may exclude the first and last weights to avoid boundary issues.)
    weights_for_pdf = weights[:, 1:-1]
    # Draw N_importance additional samples along each ray using the PDF.
    fine_t = sample_pdf(bins, weights_for_pdf, N_importance, det=False)  # (N, N_importance)
    
    # Combine coarse and fine sample locations and sort them.
    all_t = torch.cat([sampled_t, fine_t], -1)  # (N, samples + N_importance)
    all_t, _ = torch.sort(all_t, -1)
    
    # Compute fine 3D points.
    fine_points = origins.unsqueeze(1) + all_t.unsqueeze(-1) * directions.unsqueeze(1)
    fine_points_flat = fine_points.reshape(-1, 3)
    dirs_expanded_fine = directions.unsqueeze(1).expand(-1, all_t.shape[1], -1).reshape(-1, 3)
    
    # Query the neural field on all (fine) points.
    fine_rgb, fine_sigma = neural_field(fine_points_flat, dirs_expanded_fine)
    if fine_rgb is None or fine_sigma is None:
        raise ValueError("Neural field returned None for fine predictions.")
    fine_rgb = fine_rgb.view(N, all_t.shape[1], 3)
    fine_sigma = fine_sigma.view(N, all_t.shape[1])
    
    # Volume rendering for the fine pass.
    deltas_fine = all_t[:, 1:] - all_t[:, :-1]
    delta_inf_fine = torch.full((N, 1), 1e10, device=DEVICE)
    deltas_fine = torch.cat([deltas_fine, delta_inf_fine], dim=-1)
    alphas_fine = 1.0 - torch.exp(-fine_sigma * deltas_fine)
    T_fine = torch.cumprod(torch.cat([ones, 1.0 - alphas_fine + 1e-10], dim=-1), dim=-1)[:, :-1]
    weights_fine = T_fine * alphas_fine
    rendered_rgb = torch.sum(weights_fine.unsqueeze(-1) * fine_rgb, dim=1)
    
    if clear_bg:
        total_weight = weights_fine.sum(dim=-1, keepdim=True)
        rendered_rgb = rendered_rgb + (1.0 - total_weight)
    
    if debug:
        print("Hierarchical sampling enabled.")
        print("Coarse Sigma mean:", coarse_sigma.mean().item(),
              "Fine Sigma mean:", fine_sigma.mean().item())
        print("Fine weights sum mean:", weights_fine.sum(dim=-1).mean().item())
    
    return rendered_rgb
