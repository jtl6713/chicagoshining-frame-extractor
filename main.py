import os
import uuid
import json
import math
import shutil
import subprocess
from pathlib import Path

import requests
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

app = FastAPI()

BASE_DIR = Path(__file__).parent
FRAME_DIR = BASE_DIR / "frames"
TMP_DIR = BASE_DIR / "tmp"

FRAME_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

app.mount("/frames", StaticFiles(directory=str(FRAME_DIR)), name="frames")


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


def download_video(video_url: str, output_path: Path):
    with requests.get(video_url, stream=True, timeout=120) as r:
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Could not download video. Status: {r.status_code}")
        with open(output_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)


def get_video_metadata(video_path: Path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]
    data = json.loads(run_cmd(cmd))

    video_stream = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise RuntimeError("No video stream found.")

    duration = float(data["format"].get("duration", 0))
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    orientation = "Portrait" if height > width else "Landscape" if width > height else "Square"

    if orientation == "Portrait":
        aspect_ratio = "9:16"
    elif orientation == "Landscape":
        aspect_ratio = "16:9"
    else:
        aspect_ratio = "1:1"

    return {
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "orientation": orientation,
        "aspect_ratio": aspect_ratio
    }


def extract_frames(video_path: Path, record_id: str, frame_count: int):
    metadata = get_video_metadata(video_path)
    duration = metadata["duration_seconds"]

    if duration <= 0:
        raise RuntimeError("Invalid video duration.")

    interval = duration / frame_count
    frame_urls = []

    for i in range(frame_count):
        timestamp = interval * i
        # Avoid exact end of file.
        timestamp = min(timestamp, max(duration - 0.5, 0))

        frame_name = f"{record_id}_frame_{i+1:02d}.jpg"
        frame_path = FRAME_DIR / frame_name

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(frame_path)
        ]
        run_cmd(cmd)
        frame_urls.append(f"/frames/{frame_name}")

    metadata["frame_interval_seconds"] = interval
    return metadata, frame_urls

class ExtractRequest(BaseModel):
    video_url: str
    asset_name: str = ""
    record_id: str = ""
    frame_count: int = 10
    
@app.post("/extract")
async def extract(body: ExtractRequest, request: Request):

    video_url = body.video_url
    record_id = body.record_id or str(uuid.uuid4())
    frame_count = body.frame_count

    if not video_url:
        raise HTTPException(status_code=400, detail="video_url is required")

    job_id = str(uuid.uuid4())
    video_path = TMP_DIR / f"{job_id}.mp4"

    try:
        download_video(video_url, video_path)
        metadata, frame_paths = extract_frames(video_path, record_id, frame_count)

        base_url = str(request.base_url).rstrip("/")
        frame_urls = [base_url + p for p in frame_paths]

        return {
            "record_id": record_id,
            **metadata,
            "frames": frame_urls
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if video_path.exists():
            video_path.unlink()


@app.get("/")
def health():
    return {"status": "ok", "service": "ChicagoShining Frame Extractor"}
