import os
import platform
import shutil


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"DISPLAY: {os.getenv('DISPLAY')}")
    print(f"WAYLAND_DISPLAY: {os.getenv('WAYLAND_DISPLAY')}")
    print(f"XDG_RUNTIME_DIR: {os.getenv('XDG_RUNTIME_DIR')}")
    print(f"MUJOCO_GL: {os.getenv('MUJOCO_GL')}")

    for command in ["glxinfo", "weston", "python"]:
        print(f"{command}: {shutil.which(command)}")

    try:
        import mujoco

        print(f"MuJoCo: {mujoco.__version__}")
        print("MuJoCo import: OK")
    except Exception as exc:
        print(f"MuJoCo import: FAILED ({exc})")
        raise


if __name__ == "__main__":
    main()
