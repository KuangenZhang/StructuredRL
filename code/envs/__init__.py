import gym
from gym.envs.registration import registry, make, spec

def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)


register(
    id='MiniCheetahVd-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahVdEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahVdS-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahVdSEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahVdL-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahVdLEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahVdNoCtrl-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahVdNoCtrlEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)


register(
    id='MiniCheetahVdOnly-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahVdOnlyEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahPolarVd-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahPolarVdEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahPolarVdS-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahPolarVdSEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahPolarVdL-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahPolarVdLEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahPolarVdNoCtrl-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahPolarVdNoCtrlEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahPolarVdOnly-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahPolarVdOnlyEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahPolarDirectVd-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahPolarDirectVdEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahPolarDirectVdS-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahPolarDirectVdSEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahPolarDirectVdL-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahPolarDirectVdLEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahPolarDirectVdNoCtrl-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahPolarDirectVdNoCtrlEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='MiniCheetahPolarDirectVdOnly-v0',
    entry_point='envs.cheetah_mujoco:MiniCheetahPolarDirectVdOnlyEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)
