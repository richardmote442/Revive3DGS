import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TURTLE_DIR = ROOT_DIR / "Turtle"
BASICSR_DIR = TURTLE_DIR / "basicsr"
INFERENCE_SCRIPT = BASICSR_DIR / "inference.py"


def run_step(name, cmd, cwd):
    print("=" * 80)
    print(f"Running step: {name}")
    print("=" * 80)

    subprocess.run(
        cmd,
        cwd=cwd,
        check=True
    )


def main():
    run_step(
        name="Turtle weather restoration",
        cmd=["python", "inference.py"],
        cwd=BASICSR_DIR
    )


if __name__ == "__main__":
    main()