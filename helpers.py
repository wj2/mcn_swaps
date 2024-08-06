
import numpy as np


def make_unit_vector(v, squeeze=True, dim=-1):
    v = np.array(v)
    if len(v.shape) == 1:
        v = np.expand_dims(v, 0)
    v_len = np.sqrt(np.sum(v**2, axis=dim, keepdims=True))
    vl_full = np.repeat(v_len, v.shape[dim], axis=dim)
    mask_full = vl_full > 0
    v_norm = np.zeros_like(v, dtype=float)
    v_set = v[mask_full] / vl_full[mask_full]
    v_norm[mask_full] = v_set  
    if squeeze:
        v_norm = np.squeeze(v_norm)
    return v_norm


def normalize_periodic_range(
    diff, cent=0, radians=True, const=None, convert_array=True
):
    if convert_array:
        diff = np.array(diff)
    if radians and const is None:
        const = np.pi
    elif const is None:
        const = 180
    m = np.mod(diff + const, 2 * const)
    m = np.mod(m + 2 * const, 2 * const) - const + cent
    # diff = np.array(diff) - cent
    # g_mask = diff > const
    # l_mask = diff < -const
    # diff[g_mask] = -const + (diff[g_mask] - const)
    # diff[l_mask] = const + (diff[l_mask] + const)
    return m


def radian_to_sincos(rs, axis=-1):
    arr = np.stack((np.sin(rs), np.cos(rs)), axis=axis)
    return arr


def sincos_to_radian(s1, s2):
    rs = np.arctan2(s1, s2)
    return rs
