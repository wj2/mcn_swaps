import numpy as np
import torch
import matplotlib.pyplot as plt

import general.utility as u

import mcn_swaps.helpers as msh


def sample_model_responses(task, model, n_samples=1000):
    info, inputs, _ = task.sample_trials(n_samples)
    resps = np.zeros(n_samples)
    targs = np.zeros_like(resps)
    dists = np.zeros_like(resps)
    for i, inp in enumerate(inputs):
        inp_use = torch.from_numpy(inp).type(torch.float).to(model.device)
        last_act = model(inp_use)[-1]
        uv = msh.make_unit_vector(last_act.detach().cpu().numpy())
        resps[i] = msh.sincos_to_radian(*uv)
        targs[i] = info["target_color"][i]
        dists[i] = info["distractor_color"][i]
    return targs, dists, resps


def plot_sweep_responses(ax_dict, out_dict, run_ind=0, axs=None, fwid=1, ms=.1):
    if axs is None:
        plot_shape = out_dict["targets"].shape[: len(ax_dict)]
        fs = tuple(x * fwid for x in plot_shape[::-1])
        f, axs = plt.subplots(
            *plot_shape,
            sharex=True,
            sharey=True,
            figsize=fs,
        )
    errs = msh.normalize_periodic_range(out_dict["responses"] - out_dict["targets"])
    dist_errs = msh.normalize_periodic_range(
        out_dict["responses"] - out_dict["distractors"],
    )

    for ind in u.make_array_ind_iterator(axs.shape):
        ax_ind = axs[ind]
        xs, ys = list(ax_dict.values())        
        ax_ind.plot(errs[ind][run_ind], dist_errs[ind][run_ind], "o", ms=ms)
    return axs
