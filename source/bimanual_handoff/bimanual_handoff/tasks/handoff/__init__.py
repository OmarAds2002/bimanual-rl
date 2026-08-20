# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""Registers the handoff task with gymnasium.

Importing this module (e.g. `import bimanual_handoff.tasks.handoff`) is what makes
`gym.make("Isaac-Handoff-Direct-v0", ...)` work — train.py / play.py import it for
the side effect.
"""

import gymnasium as gym

from .handoff_env import HandoffEnv
from .handoff_env_cfg import HandoffEnvCfg

gym.register(
    id="Isaac-Handoff-Direct-v0",
    entry_point="bimanual_handoff.tasks.handoff.handoff_env:HandoffEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": HandoffEnvCfg},
)