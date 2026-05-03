from pathlib import Path

import mujoco
import numpy as np
import typer


MODEL_XML = """
<mujoco model="simple_pendulum">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="pendulum" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.6" size="0.035" rgba="0.2 0.4 0.9 1"/>
      <geom name="bob" type="sphere" pos="0 0 -0.7" size="0.08" rgba="0.9 0.3 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge" gear="0.15"/>
  </actuator>
</mujoco>
"""


def main(steps: int = 500, out: Path = Path("data/rollout.csv")) -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    rows: list[str] = ["step,time,qpos,qvel,ctrl"]

    for step in range(steps):
        data.ctrl[0] = 0.5 * np.sin(step * 0.03)
        mujoco.mj_step(model, data)
        rows.append(
            f"{step},{data.time:.6f},{data.qpos[0]:.8f},{data.qvel[0]:.8f},{data.ctrl[0]:.8f}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {steps} MuJoCo telemetry rows to {out}")


if __name__ == "__main__":
    typer.run(main)
