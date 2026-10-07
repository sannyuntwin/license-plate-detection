import os
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    source_path = str(project_root / "src")
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_path, existing_pythonpath) if path
    )
    return subprocess.call(
        [sys.executable, "-m", "license_plate_detection.pipeline", *sys.argv[1:]],
        cwd=project_root,
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())