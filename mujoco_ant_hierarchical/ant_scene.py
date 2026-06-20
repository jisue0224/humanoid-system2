"""Custom Ant-v4 scene generation for obstacle and camera experiments."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
import gymnasium.envs.mujoco


ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "generated"
DEFAULT_GOAL_XY = (5.0, 0.0)
DEFAULT_OBSTACLE_POS = (2.5, 0.0, 0.45)
DEFAULT_OBSTACLE_SIZE = (0.45, 1.15, 0.45)


def ant_asset_path() -> Path:
    return Path(gymnasium.envs.mujoco.__file__).parent / "assets" / "ant.xml"


def _append_common_scene(worldbody: ET.Element, goal_xy: tuple[float, float]) -> None:
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overhead",
            "pos": f"{goal_xy[0] * 0.5:.3f} 0 8.0",
            "xyaxes": "1 0 0 0 1 0",
            "fovy": "45",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "goal_marker",
            "type": "sphere",
            "pos": f"{goal_xy[0]:.3f} {goal_xy[1]:.3f} 0.08",
            "size": "0.18",
            "rgba": "0.1 0.85 0.15 0.85",
            "contype": "0",
            "conaffinity": "0",
        },
    )


def _append_ego_camera(worldbody: ET.Element) -> None:
    torso = worldbody.find(".//body[@name='torso']")
    if torso is None:
        raise RuntimeError("Could not find Ant torso body in XML")
    ET.SubElement(
        torso,
        "camera",
        {
            "name": "ego",
            "pos": "0.28 0 0.16",
            "xyaxes": "0 -1 0 0 0 1",
            "fovy": "80",
        },
    )


def build_ant_scene_xml(
    *,
    with_obstacle: bool,
    goal_xy: tuple[float, float] = DEFAULT_GOAL_XY,
    obstacle_pos: tuple[float, float, float] = DEFAULT_OBSTACLE_POS,
    obstacle_size: tuple[float, float, float] = DEFAULT_OBSTACLE_SIZE,
) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(ant_asset_path())
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Could not find worldbody in Ant XML")

    _append_common_scene(worldbody, goal_xy)
    _append_ego_camera(worldbody)
    if with_obstacle:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "static_obstacle",
                "type": "box",
                "pos": f"{obstacle_pos[0]:.3f} {obstacle_pos[1]:.3f} {obstacle_pos[2]:.3f}",
                "size": f"{obstacle_size[0]:.3f} {obstacle_size[1]:.3f} {obstacle_size[2]:.3f}",
                "rgba": "0.85 0.12 0.10 1.0",
                "friction": "1 0.5 0.5",
                "conaffinity": "1",
                "contype": "1",
            },
        )

    filename = "ant_hier_obstacle.xml" if with_obstacle else "ant_hier_open.xml"
    path = GENERATED_DIR / filename
    tree.write(path, encoding="unicode")
    return path


def make_scene_env(
    *,
    with_obstacle: bool,
    render_mode: str | None = None,
    camera_name: str | None = None,
    width: int = 640,
    height: int = 480,
    goal_xy: tuple[float, float] = DEFAULT_GOAL_XY,
):
    xml_path = build_ant_scene_xml(with_obstacle=with_obstacle, goal_xy=goal_xy)
    return gym.make(
        "Ant-v4",
        xml_file=str(xml_path),
        render_mode=render_mode,
        camera_name=camera_name,
        width=width,
        height=height,
    )
