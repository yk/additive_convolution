#!/usr/bin/env python3

import numpy as np

def batch_add_conv_1d(x, k, reduction_fn=np.max):
    """Performs additive batch convolution with a 1D kernel.

    Args:
        x: data, can be batched.
        k: kernel, must be 1D
        reduction_fn: a function for reduction in the final step. must accept `axis` argument. defaults to max.

    Returns:
        x additively convoluted with k over the last axis
    """
    xpk = x[..., None, :] + k[:, None]
    pad_size = k.size // 2
    pad_axes = np.zeros((len(xpk.shape), 2), dtype=np.int)
    pad_axes[-1, :] = pad_size
    xpk_pad = np.pad(xpk, pad_axes, mode='constant')
    xpk_shifted = np.lib.stride_tricks.as_strided(
        xpk_pad, 
        shape=xpk.shape, 
        strides=(
            xpk_pad.strides[:-2] + 
            (xpk_pad.strides[-2] + xpk_pad.strides[-1], xpk_pad.strides[-1])
        )
    )
    xpk_red = reduction_fn(xpk_shifted, axis=-2)
    return xpk_red
    
    
