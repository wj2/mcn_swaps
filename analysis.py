
import numpy as np
import torch

import mcn_swaps.helpers as msh


def sample_model_responses(task, model, n_samples=1000):
    info, inputs, _ = task.sample_trials(n_samples)
    resps = np.zeros(n_samples)
    targs = np.zeros_like(resps)
    dists = np.zeros_like(resps)
    for i, inp in enumerate(inputs):
        inp_use = torch.from_numpy(inp).type(torch.float)
        last_act = model(inp_use)[-1]
        uv = msh.make_unit_vector(last_act.detach().numpy())
        resps[i] = msh.sincos_to_radian(*uv)
        targs[i] = info["target_color"][i]
        dists[i] = info["distractor_color"][i]
    return targs, dists, resps
