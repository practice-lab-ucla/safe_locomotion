import sys


def make_gym_environment(task: str, num_envs: int, headless: bool=False, video: bool=True):
    import argparse
    import gymnasium
    try:
        from omni.isaac.lab.app import AppLauncher
    except ModuleNotFoundError:
        from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser("Isaac Lab: Omniverse Robotics Environments!")
    parser.add_argument("--num_envs", type=int, default=num_envs, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=task, help="Name of the task.")
    
    if headless:
        sys.argv.append("--headless")

        if video:
            parser.add_argument("--video_length", type=int, default=120)
            parser.add_argument("--video_interval", type=int, default=1000)

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)

    import safe_locomotion.tasks
    try:
        import omni.isaac.lab_tasks  # type: ignore
        from omni.isaac.lab_tasks.utils import parse_env_cfg  # type: ignore
    except ModuleNotFoundError:
        import isaaclab_tasks  # type: ignore
        from isaaclab_tasks.utils import parse_env_cfg  # type: ignore

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=True)
    env = gymnasium.make(args.task, cfg=cfg, render_mode=None)

    return env