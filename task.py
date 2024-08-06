

import numpy as np
import neurogym as ngym
from neurogym import spaces
import pandas as pd

import mcn_swaps.helpers as msh


class RetrospectiveContinuousReportTask(ngym.TrialEnv):
    """Retrospectively cued task with report of a continuous stimulus.

    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent
         dynamics in prefrontal cortex''',
        'tags': ['perceptual', 'context dependent', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=0, n_cols=64):
        super().__init__(dt=dt)

        # trial conditions
        self.cues = [0, 1]  # index for context inputs
        # color responses
        self.possible_colors = np.linspace(0, np.pi * 2, n_cols + 1)[:-1]
        self.choices = msh.radian_to_sincos(self.possible_colors)
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.fix_thr = .9
        self.resp_thr = .9
        self.reward_thr = .3

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 300,
            'stimulus': 750,
            'delay1': ngym.random.TruncExp(600, 300, 1000),
            "cue": 400,
            'delay2': ngym.random.TruncExp(600, 300, 1000),
            'decision': 500}
        if timing is not None:
            self.timing.update(timing)

        self.abort = False

        # set action and observation space
        names = ['fixation', 'color1_sin', 'color1_cos',
                 'color2_sin', 'color2_cos', 'cue_1', "cue_2"]
        name = {name: i for i, name in enumerate(names)}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(names),),
                                            dtype=np.float32, name=name)
        self.obs_dims = len(names)

        name = {'eye_x': 0, 'eye_y': 1,}
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(len(name),), name=name)
        self.action_dims = len(name)

    def _new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        trial = {
            'color1': self.rng.choice(self.possible_colors),
            'color2': self.rng.choice(self.possible_colors),
            'cue': self.rng.choice(self.cues),
        }
        trial.update(kwargs)

        color1, color2 = trial['color1'], trial['color2']
        c1_sin, c1_cos = msh.radian_to_sincos(color1)
        c2_sin, c2_cos = msh.radian_to_sincos(color2)
        
        # -----------------------------------------------------------------------
        # Periods
        # -----------------------------------------------------------------------
        periods = ['fixation', 'stimulus', 'delay1', "cue", "delay2", 'decision']
        self.add_period(periods)

        self.add_ob(1, where='fixation')
        self.set_ob(0, period="decision", where="fixation")
        self.add_ob(c1_sin, period='stimulus', where='color1_sin')
        self.add_ob(c1_cos, period='stimulus', where='color1_cos')
        self.add_ob(c2_sin, period='stimulus', where='color2_sin')
        self.add_ob(c2_cos, period='stimulus', where='color2_cos')
        self.add_randn(0, self.sigma)

        if trial['cue'] == 1:
            self.add_ob(1, period="cue", where='cue_1')
            tc_sin, tc_cos = msh.radian_to_sincos(color1)
            dc_sin, dc_cos = msh.radian_to_sincos(color2)
            trial["target_color"] = color1
            trial["distractor_color"] = color2
        else:
            self.add_ob(1, period="cue", where='cue_2')
            tc_sin, tc_cos = msh.radian_to_sincos(color2)
            dc_sin, dc_cos = msh.radian_to_sincos(color1)
            trial["target_color"] = color2
            trial["distractor_color"] = color1

        self.set_groundtruth((tc_sin, tc_cos), period="decision")

        return trial

    def _step(self, action):
        ob = self.ob_now
        gt = self.gt_now

        action_dev = np.sqrt(np.sum(action ** 2))
        new_trial = False
        reward = 0
        if self.in_period('fixation'):
            if action_dev > self.fix_thr:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action_dev > self.resp_thr:  # broke fixation
                new_trial = True
                if np.sqrt(np.sum((action - gt) ** 2)) < self.reward_thr:
                    reward = self.rewards['correct']
                    self.performance = 1

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}

    def sample_trials(self, n_trls, **kwargs):
        info = []
        inputs = []
        targs = []
        for i in range(n_trls):
            info_i = self.new_trial()
            inp = self.ob
            targ = self.gt
            inputs.append(inp)
            targs.append(targ)
            info.append(info_i)
        info = pd.DataFrame(info)
        return info, inputs, targs
