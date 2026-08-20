# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""rsl_rl PPO config for the bimanual handoff task.

One gotcha worth remembering: set every field explicitly. rsl_rl's .to_dict()
serialises any unset nested default as a raw dict, and the runner then chokes on
it — so I don't rely on defaults for the policy/algorithm blocks.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HandoffPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 48
    max_iterations: int = 5000
    save_interval: int = 100          # checkpoint every 100 iters -> model_*.pt
    experiment_name: str = "p3_handoff"
    resume: bool = False              # explicit: never silently resume the old run.
    #                                   (forgetting rm -rf logs/ once cost me an evening —
    #                                    a stale critic + new reward = no exploration back out)

    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=0.5,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=3e-4,
        clip_param=0.2,
        entropy_coef=0.01,
        value_loss_coef=1.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        desired_kl=0.01,
        schedule="adaptive",
    )