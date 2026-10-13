"""用于启动 WSL/OpenDroneMap 四波段流程的 Windows 脚本。"""

import subprocess

from src.m3m.config import project_dir


def run_orthorectification() -> None:
    script = project_dir / "scripts" / "orthorectify_wsl.sh"
    if not script.is_file():
        raise FileNotFoundError(script)
    script_wsl = "/mnt/e/Projects/DJi/M3M_Project/scripts/orthorectify_wsl.sh"
    subprocess.run(
        ["wsl.exe", "-d", "Ubuntu", "-u", "root", "--", "bash", script_wsl],
        check=True,
    )


if __name__ == "__main__":
    run_orthorectification()
