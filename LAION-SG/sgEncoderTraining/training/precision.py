import torch
from contextlib import suppress

def get_autocast(precision, dtype_override=None):
    """
    Returns a context manager for autocast based on the requested precision.
    dtype_override allows the caller to override the dtype used for mixed precision.
    """
    if dtype_override is not None:
        return lambda: torch.cuda.amp.autocast(dtype=dtype_override)

    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16':
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress