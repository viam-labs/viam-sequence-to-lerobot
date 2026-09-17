import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_job.py"


def run(*args: str) -> str:
    return subprocess.run([sys.executable, SCRIPT, "--dry-run", *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def test_fresh_run_matches_hand_written_command():
    cmd = run()
    assert cmd.startswith("hf jobs run --namespace viamrobotics --flavor a100-large --secrets HF_TOKEN --detach huggingface/lerobot-gpu lerobot-train ")
    assert "--policy.path=lerobot/smolvla_base" in cmd
    assert "--save_checkpoint=true --save_freq=5000 --save_checkpoint_to_hub=true --steps=30000" in cmd
    assert """'--rename_map={"observation.images.webcam": "observation.images.camera1", "observation.images.realsense_webcam": "observation.images.camera2"}'""" in cmd
    assert "--resume" not in cmd


def test_resume_swaps_policy_path_for_config_path_and_overrides_apply():
    cmd = run("--resume", "--steps=50000")
    assert "lerobot-train --resume=true --config_path=viamrobotics/smolvla-box-bot-subtasks " in cmd
    assert "--policy.path" not in cmd and "--policy.repo_id" not in cmd
    assert "--steps=50000" in cmd and "--steps=30000" not in cmd
    assert "--config_path=other/repo" in run("--resume", "other/repo")
