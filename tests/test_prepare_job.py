import shlex
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(*args: str) -> list[str]:
    out = subprocess.run([sys.executable, SCRIPTS / "prepare_job.py", "--dry-run", *args], check=True,
                         capture_output=True, text=True).stdout
    return shlex.split(out)


def test_pod_script_archives_then_exports_converts_pushes_in_order():
    cmd = run()
    assert cmd[:3] == ["hf", "jobs", "run"] and cmd[-4:-1] == ["python:3.12", "bash", "-c"]
    assert "--secrets" in cmd and "VIAM_API_KEY" in cmd
    script = cmd[-1]
    steps = ["create_tag", "viam dataset export --destination=export --dataset-id=6aa954ab0ccd60006b98069e",
             "viam-seq-to-lerobot export --output-root=lerobot --repo-id=viamrobotics/box-opener-subtasks "
             "--camera=webcam --camera=realsense-webcam --fps=30",
             "push_to_hub"]
    positions = [script.index(s) for s in steps]
    assert positions == sorted(positions)
    assert "lerobot-train" not in script


def test_then_train_appends_detached_training_submit():
    script = run("--then-train", str(SCRIPTS / "train_job.toml"))[-1]
    last = shlex.split(script.splitlines()[-1])
    assert last[:3] == ["hf", "jobs", "run"] and "--detach" in last and "lerobot-train" in last
    assert script.index("push_to_hub") < script.index("lerobot-train")
