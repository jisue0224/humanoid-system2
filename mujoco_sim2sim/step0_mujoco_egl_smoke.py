#!/usr/bin/env python3
"""Minimal MuJoCo EGL rendering smoke test."""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image


XML = """
<mujoco model="egl_smoke">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global azimuth="135" elevation="-25"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256"
             rgb1=".2 .3 .4" rgb2=".1 .15 .2"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.1"/>
    <material name="body_mat" rgba="0.8 0.25 0.15 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 4"/>
    <geom name="floor" type="plane" size="2 2 .1" material="grid"/>
    <body name="falling_body" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size=".15 .15 .15" material="body_mat"/>
    </body>
    <camera name="overhead" pos="0 -2.2 1.6" xyaxes="1 0 0 0 0.6 0.8" fovy="55"/>
  </worldbody>
</mujoco>
"""


def main() -> None:
    output_dir = Path("mujoco_sim2sim/artifacts/step0")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    for _ in range(50):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=256, width=256)
    renderer.update_scene(data, camera="overhead")
    pixels = renderer.render()
    renderer.close()

    image_path = output_dir / "egl_smoke_rgb.png"
    Image.fromarray(pixels).save(image_path)

    print(f"mujoco_version={mujoco.__version__}")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"image={image_path}")


if __name__ == "__main__":
    main()

