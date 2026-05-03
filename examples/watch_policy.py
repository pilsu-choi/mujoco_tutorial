from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
import typer


def main(
    env_name: str = "Humanoid-v5",
    model_path: Path = Path("models/ppo_humanoid.zip"),
    episodes: int = 3,
    max_steps: int = 1_000,
    sleep_seconds: float = 0.01,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run examples/train_ppo.py first or pass --model-path."
        )

    env = gym.make(env_name, render_mode="human")
    model = PPO.load(model_path, env=env)

    try:
        for episode_id in range(episodes):
            observation, _ = env.reset()
            total_reward = 0.0

            for _step in range(max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                time.sleep(sleep_seconds)

                if terminated or truncated:
                    break

            print(f"Episode {episode_id}: reward={total_reward:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    typer.run(main)
