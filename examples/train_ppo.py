from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import typer


def main(
    env_name: str = "Humanoid-v5",
    timesteps: int = 10_000,
    learning_rate: float = 3e-4,
    seed: int = 42,
    model_out: Path = Path("models/ppo_humanoid.zip"),
    experiment_name: str = "mujoco-ppo",
    mlflow_uri: str = "file:mlruns",
) -> None:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    with mlflow.start_run(run_name=f"ppo-{env_name}-{timesteps}") as run:
        mlflow.log_params(
            {
                "env_name": env_name,
                "algorithm": "PPO",
                "total_timesteps": timesteps,
                "learning_rate": learning_rate,
                "seed": seed,
            }
        )

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            seed=seed,
            verbose=1,
        )
        model.learn(total_timesteps=timesteps)

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
        mlflow.log_metrics({"mean_reward": float(mean_reward), "std_reward": float(std_reward)})

        model_out.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_out)
        mlflow.log_artifact(str(model_out), artifact_path="model")

        print(f"Run ID: {run.info.run_id}")
        print(f"Saved PPO model to {model_out}")
        print(f"Evaluation reward: mean={mean_reward:.2f}, std={std_reward:.2f}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    typer.run(main)
