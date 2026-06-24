"""ffmpeg x11grab helpers shared by franka_sim_rollout_record and eval_base_policy_sim."""
import os
import re
import subprocess


def start_recorder(out_path, display, size, fps, crop=None):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "x11grab", "-video_size", size,
           "-framerate", str(fps), "-i", display]
    if crop:
        cmd += ["-vf", crop]
    cmd += ["-pix_fmt", "yuv420p", out_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_recorder(proc):
    try:
        proc.communicate(input=b"q", timeout=10)
    except Exception:
        proc.terminate()
        proc.wait(timeout=10)


def detect_crop(display, size, fps):
    """Capture a few seconds, run cropdetect, return 'crop=W:H:X:Y' or None."""
    cmd = ["ffmpeg", "-y", "-f", "x11grab", "-video_size", size,
           "-framerate", str(fps), "-i", display,
           "-t", "2", "-vf", "cropdetect",
           "-pix_fmt", "yuv420p", "-f", "null", "/dev/null"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    matches = re.findall(r"crop=(\d+:\d+:\d+:\d+)", proc.stderr)
    return f"crop={matches[-1]}" if matches else None
