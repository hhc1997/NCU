import numpy as np    


def cosine_scheduler(optimizer, base_lr, num_warmup_steps, total_steps):
    initial_lrs = [group['lr'] for group in optimizer.param_groups]

    def _scheduler(current_step):
        if (current_step < num_warmup_steps):
            scale = (current_step + 1) / num_warmup_steps
        else:
            n = current_step - num_warmup_steps
            d = total_steps - num_warmup_steps
            scale = 0.5 * (1 + np.cos(np.pi * n / d))
        for param_group, initial_lr in zip(optimizer.param_groups, initial_lrs):
            param_group["lr"] = initial_lr * scale

    return _scheduler