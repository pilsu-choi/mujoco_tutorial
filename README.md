# MuJoCo Tutorial

WSL2/WSLg 환경에서 MuJoCo GUI 실행 가능 여부를 확인하고, headless rollout으로 synthetic robot telemetry를 만드는 최소 튜토리얼입니다.

## Setup

```bash
cd ~/resume/mujoco_tutorial
uv venv
source .venv/bin/activate
uv sync
```

## WSL GUI Check

```bash
uv run python scripts/check_wsl_gui.py
```

`DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`처럼 값이 있으면 WSLg GUI를 사용할 가능성이 높습니다.

## Run GUI Viewer

```bash
uv run python examples/gui_viewer.py
```

GUI가 뜨지 않고 OpenGL/GLFW 에러가 나면 Ubuntu 패키지가 필요할 수 있습니다.

```bash
sudo apt update
sudo apt install -y libgl1 libglfw3 libglew2.2 libosmesa6
```

## Run Headless Rollout

```bash
uv run python examples/headless_rollout.py --steps 500 --out data/rollout.csv
```

GUI가 필요 없는 포트폴리오용 synthetic telemetry 생성은 headless 실행으로 충분합니다.

```bash
MUJOCO_GL=osmesa uv run python examples/headless_rollout.py
```
