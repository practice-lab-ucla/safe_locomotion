import sys


def make_isaac_environment(
    task: str, num_envs: int, headless: bool = False, video: bool = True
):
    # # sanitize the argument so Isaac Sim Launcher will not see hydra arg
    clean = [sys.argv[0]]
    # it = iter(sys.argv[1:])
    # for arg in it:
    #     if arg == "-m" or arg in [
    #         "--env_id",
    #         "--exp",
    #         "--exps"
    #         "--checkpoint_dir",
    #         "--result_dir",
    #     ]:
    #         next(it, None)
    #         continue
    #     elif arg.startswith("agent"):
    #         next(it, None)
    #         continue
    #     clean.append(arg)
    sys.argv = clean

    import argparse
    import gymnasium

    try:
        from omni.isaac.lab.app import AppLauncher
    except ModuleNotFoundError:
        from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser("Isaac Lab: Omniverse Robotics Environments!")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=num_envs,
        help="Number of environments to simulate.",
    )
    parser.add_argument("--task", type=str, default=task, help="Name of the task.")

    if headless:
        sys.argv.append("--headless")

        if video:
            parser.add_argument("--video_length", type=int, default=120)
            parser.add_argument("--video_interval", type=int, default=1000)

    parser.add_argument("--seed", type=str, default=42, help="Seed the environment")

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

    cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs, use_fabric=True
    )
    env = gymnasium.make(args.task, cfg=cfg, render_mode=None)

    return env, app_launcher
