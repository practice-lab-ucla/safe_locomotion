# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# Base Environment
gym.register(
    id="Safe-Locomotion-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_locomotion_env_cfg:BaseCfg",
    },
)

gym.register(
    id="Safe-Locomotion-Base-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_locomotion_env_cfg:BaseCfg1",
    },
)

gym.register(
    id="Safe-Locomotion-Base-Visualize-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_locomotion_env_cfg:BaseCfgVisualize",
    },
)

gym.register(
    id="Safe-Locomotion-Reward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_locomotion_env_cfg:MoreRewardCfg",
    },
)

gym.register(
    id="Safe-Locomotion-Fast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_locomotion_env_cfg:FasterCfg",
    },
)

gym.register(
    id="Safe-Locomotion-DistLow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.safe_locomotion_env_cfg:RandLowCfg",
    },
)

gym.register(
    id="Safe-Locomotion-DistHigh-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.safe_locomotion_env_cfg:RandHighCfg",
    },
)

# Inclined Terrain

gym.register(
    id="Safe-Locomotion-Incline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_locomotion_env_cfg:InclineLevel0",
    },
)

gym.register(
    id="Safe-Locomotion-Incline-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_locomotion_env_cfg:InclineLevel1",
    },
)

gym.register(
    id="Safe-Locomotion-Incline-Visualize-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_locomotion_env_cfg:InclineLevel1Visualize",
    },
)

gym.register(
    id="Safe-Locomotion-Incline-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_locomotion_env_cfg:InclineLevel2",
    },
)

gym.register(
    id="Safe-Locomotion-Friction-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_locomotion_env_cfg:FrictionLevel0",
    },
)

gym.register(
    id="Safe-Locomotion-Friction-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_locomotion_env_cfg:FrictionLevel1",
    },
)

gym.register(
    id="Safe-Locomotion-Friction-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_locomotion_env_cfg:FrictionLevel2",
    },
)

gym.register(
    id="Safe-Locomotion-Restitution-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_locomotion_env_cfg:RestitutionLevel0",
    },
)

gym.register(
    id="Safe-Locomotion-Restitution-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_locomotion_env_cfg:RestitutionLevel1",
    },
)

# Rugged Terrain

gym.register(
    id="Safe-Locomotion-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_locomotion_env_cfg:RoughLevel0",
    },
)

gym.register(
    id="Safe-Locomotion-Rough-Visualize-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_locomotion_env_cfg:RoughLevel0Visualize",
    },
)

# Larger Gain

gym.register(
    id="Safe-Locomotion-Gain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mechanics_locomotion_env_cfg:GainLevel0",
    },
)

gym.register(
    id="Safe-Locomotion-Gain-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mechanics_locomotion_env_cfg:GainLevel1",
    },
)

# Jitter in actuation

gym.register(
    id="Safe-Locomotion-Jitter-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mechanics_locomotion_env_cfg:JitterLevel0",
    },
)

# Delay in observation

gym.register(
    id="Safe-Locomotion-Delay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obs_locomotion_env_cfg:DelayLevel0",
    },
)

gym.register(
    id="Safe-Locomotion-Delay-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obs_locomotion_env_cfg:DelayLevel1",
    },
)

gym.register(
    id="Safe-Locomotion-Delay-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obs_locomotion_env_cfg:DelayLevel2",
    },
)

# Larger command speed

gym.register(
    id="Safe-Locomotion-Command-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obs_locomotion_env_cfg:CommandLevel0",
    },
)

gym.register(
    id="Safe-Locomotion-Command-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obs_locomotion_env_cfg:CommandLevel1",
    },
)

# Brownian sensor noise

gym.register(
    id="Safe-Locomotion-Brownian-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obs_locomotion_env_cfg:BrownianLevel0",
    },
)

gym.register(
    id="Safe-Locomotion-Brownian-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obs_locomotion_env_cfg:BrownianLevel1",
    },
)


gym.register(
    id="Safe-Locomotion-Push-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_locomotion_env_cfg:PushLevel0",
    },
)

gym.register(
    id="Safe-Locomotion-Push-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_locomotion_env_cfg:PushLevel1",
    },
)

gym.register(
    id="Safe-Locomotion-Push-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_locomotion_env_cfg:PushLevel2",
    },
)

gym.register(
    id="Safe-Locomotion-Push-Visualize-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_locomotion_env_cfg:PushLevel2Visualize",
    },
)