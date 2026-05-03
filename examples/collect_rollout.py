from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import typer


def _as_list(value: np.ndarray | float | int) -> list[float]:
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def main(
    env_name: str = "Humanoid-v5",
    episodes: int = 3,
    max_steps: int = 500,
    out: Path = Path("data/rollouts/humanoid_random_policy.parquet"),
    seed: int = 42,
) -> None:
    env = gym.make(env_name)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for episode_id in range(episodes):
        observation, _ = env.reset(seed=seed + episode_id)

        for step in range(max_steps):
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

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
                    "policy": "random",
                    "seed": seed,
                }
            )

            observation = next_observation
            if done:
                break

        # Keep action sampling reproducible enough for a tutorial without depending on env internals.
        env.action_space.seed(int(rng.integers(0, 1_000_000)))

    env.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"Wrote {len(rows)} transitions from {episodes} episodes to {out}")


if __name__ == "__main__":
    typer.run(main)
