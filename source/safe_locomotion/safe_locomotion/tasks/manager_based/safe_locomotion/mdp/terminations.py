from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity import mdp


def check_fall(env: ManagerBasedEnv):
    return mdp.projected_gravity(env=env)[:, 2] > 0.5
