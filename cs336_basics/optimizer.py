from collections.abc import Iterable
import numpy as np
import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) are modified in-place if clipping is needed.
    This implementation handles dense and sparse gradients.
    """
    if max_l2_norm <= 0:
        return

    total_norm_sq = 0.0
    for p in parameters:
        g = p.grad
        if g is None:
            continue
        if getattr(g, "is_sparse", False):
            g = g.coalesce()
            vals = g._values()
            total_norm_sq += float(vals.double().pow(2).sum().item())
        else:
            total_norm_sq += float(torch.sum(g.detach() ** 2).item())

    total_norm = total_norm_sq ** 0.5
    if total_norm == 0.0:
        return

    if total_norm <= max_l2_norm:
        return
    clip_coef = max_l2_norm / (total_norm + 1e-6)

    # apply scaling in-place (dense) or recreate sparse grad with scaled values
    for p in parameters:
        g = p.grad
        if g is None:
            continue
        if getattr(g, "is_sparse", False):
            g = g.coalesce()
            indices = g._indices()
            values = g._values()
            values = values.mul(clip_coef)
            p.grad = torch.sparse_coo_tensor(indices, values, g.shape)
        else:
            p.grad.mul_(clip_coef)


def gradient_clipping_old(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) are modified in-place if clipping is needed.
    This implementation handles dense and sparse gradients.
    """
    if max_l2_norm <= 0:
        return

    total_norm_sq = 0.0
    gradients = [p.grad for p in parameters if p.grad is not None]
    total_norm_sq = torch.stack([torch.sum(g.pow(2))for g in gradients]).sum().item()

    total_norm = total_norm_sq ** 0.5
    if total_norm == 0.0:
        return

    if total_norm <= max_l2_norm:
        return
    clip_coef = max_l2_norm / (total_norm + 1e-6)

    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(clip_coef)


class SimpleAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        default = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, default)
    
    @torch.no_grad()
    def step(self): 
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                lr = group['lr']
                grad = p.grad

                p.data.mul_(1 - lr * group['weight_decay'])

                state = self.state[p]
                if len(state)==0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                beta1 = group['betas'][0]
                beta2 = group['betas'][1]
                state['step'] = state['step'] + 1 ## todo: tensor位置
                state['exp_avg'].lerp_(grad, 1 - beta1)
                state['exp_avg_sq'].lerp_(grad**2, 1 - beta2) # .mul_(beta2).addcmul_(grad, grad, 1-beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = lr / bias_correction1
                denom = torch.add((state['exp_avg_sq'].sqrt() / torch.tensor(bias_correction2).sqrt()), (group['eps']))

                p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)


def get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    """Get learning rate at iteration it following a cosine schedule with warmup.

    Args:
        it (int): current iteration (0-indexed).
        max_learning_rate (float): maximum learning rate.
        min_learning_rate (float): minimum learning rate.
        warmup_iters (int): number of warmup iterations.
        cosine_cycle_iters (int): number of iterations in the cosine cycle.

    Returns:
        float: learning rate at iteration it.
    """
    if it <= warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1.0 + np.cos(np.pi * progress))
    else:
        return min_learning_rate
    
