from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import typer


def _as_list(value: np.ndarray | float | int) -> list[float]:
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def main(
    env_name: str = "Humanoid-v5",
    model_path: Path = Path("models/ppo_humanoid.zip"),
    episodes: int = 5,
    max_steps: int = 1_000,
    out: Path = Path("data/eval/humanoid_evaluation_rollout.parquet"),
    experiment_name: str = "mujoco-ppo",
    mlflow_uri: str = "file:mlruns",
) -> None:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    env = gym.make(env_name)
    model = PPO.load(model_path, env=env)
    rows: list[dict[str, object]] = []
    episode_rewards: list[float] = []

    with mlflow.start_run(run_name=f"eval-{env_name}") as run:
        mlflow.log_params(
            {
                "env_name": env_name,
                "model_path": str(model_path),
                "eval_episodes": episodes,
                "max_steps": max_steps,
            }
        )

        for episode_id in range(episodes):
            observation, _ = env.reset()
            total_reward = 0.0

            for step in range(max_steps):
                action, _ = model.predict(observation, deterministic=True)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)

                rows.append(
                    {
                        "episode_id": episode_id,
                        "step": step,
                        "env_name": env_name,
                        "observation": _as_list(observation),
                        "action": _as_list(action),
                        "reward": float(reward),
                        "next_observation": _as_list(next_observation),
                        "done": bool(done),
                        "policy": "ppo",
                    }
                )

                observation = next_observation
                if done:
                    break

            episode_rewards.append(total_reward)
            mlflow.log_metric("episode_reward", total_reward, step=episode_id)

        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out, index=False)

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mlflow.log_metrics({"eval_mean_reward": mean_reward, "eval_std_reward": std_reward})
        mlflow.log_artifact(str(out), artifact_path="evaluation")

        print(f"Run ID: {run.info.run_id}")
        print(f"Wrote {len(rows)} evaluation transitions to {out}")
        print(f"Evaluation reward: mean={mean_reward:.2f}, std={std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    typer.run(main)
