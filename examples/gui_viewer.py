import time

import mujoco
import mujoco.viewer
import numpy as np


MODEL_XML = """
<mujoco model="simple_pendulum_viewer">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual>
    <global azimuth="120" elevation="-20"/>
  </visual>
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


def main() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    step = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[0] = 0.5 * np.sin(step * 0.03)
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
