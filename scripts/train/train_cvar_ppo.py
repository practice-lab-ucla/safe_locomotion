import hydra
from omegaconf import DictConfig, OmegaConf
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed
from skrl.trainers.torch import SequentialTrainer

from utils import make_isaac_environment
from networks import StateCriticMLP, StochasticActorMLP
from agents import CVARPPO


@hydra.main(version_base="1.1", config_path="../../conf", config_name="cvar_ppo_train")
def main(cfg: DictConfig):
    print(f"\n>>> starting run with lam_max = {cfg.agent.lam_max}\n")
    set_seed(cfg.seed)
    env, app_launcher = make_isaac_environment(
        task=cfg.env.task, num_envs=cfg.env.num_envs, headless=cfg.env.headless
    )
    env = wrap_env(env)
    device = env.device

    memory = RandomMemory(
        memory_size=cfg.agent.memory.memory_size, num_envs=env.num_envs, device=device
    )
    models = {
        "policy": StochasticActorMLP(
            env.observation_space, env.action_space, device, elementwise_affine=cfg.agent.elementwise_affine
        ),
        "value": StateCriticMLP(env.observation_space, env.action_space, device),
    }

    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveRL

    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
    agent_cfg["learning_rate_scheduler"] = eval(
        agent_cfg.get("learning_rate_scheduler", "None")
    )

    sp = agent_cfg.get("state_preprocessor")
    if isinstance(sp, str):
        # e.g. sp == "RunningStandardScaler()"
        agent_cfg["state_preprocessor"] = eval(sp) # (agent_cfg.)
    else:
        agent_cfg["state_preprocessor"] = None


    vp = agent_cfg.get("value_preprocessor")
    if isinstance(vp, str):
        agent_cfg["value_preprocessor"] = eval(vp)
    else:
        agent_cfg["value_preprocessor"] = None

    agent_cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": device,
    }
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    agent = CVARPPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    trainer = SequentialTrainer(
        cfg=OmegaConf.to_container(cfg.trainer, resolve=True), env=env, agents=agent
    )

    trainer.train()

    env.close()
    app_launcher.app.close()


if __name__ == "__main__":
    main()
