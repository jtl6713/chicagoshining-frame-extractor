import os
import uuid
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

app = FastAPI()

BASE_DIR = Path(__file__).parent

# Use /tmp on Render so temporary files do not accumulate inside the app folder.
FRAME_DIR = Path("/tmp/chicagoshining_frames")
TMP_DIR = Path("/tmp/chicagoshining_tmp")

FRAME_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/frames", StaticFiles(directory=str(FRAME_DIR)), name="frames")


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


def download_video(video_url: str, output_path: Path):
    with requests.get(video_url, stream=True, timeout=180, allow_redirects=True) as r:
        if r.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Could not download video. Status: {r.status_code}"
            )

        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


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

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None
    )

    if not video_stream:
        raise RuntimeError("No video stream found.")

    duration = float(data.get("format", {}).get("duration", 0))
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    if width <= 0 or height <= 0:
        orientation = "Unknown"
        aspect_ratio = "Unknown"
    elif height > width:
        orientation = "Portrait"
        aspect_ratio = "9:16"
    elif width > height:
        orientation = "Landscape"
        aspect_ratio = "16:9"
    else:
        orientation = "Square"
        aspect_ratio = "1:1"

    return {
        "duration_seconds": round(duration, 2),
        "width": width,
        "height": height,
        "orientation": orientation,
        "aspect_ratio": aspect_ratio
    }


def get_frame_timestamps(duration: float, frame_count: int):
    """
    Avoid 0% and 100% because those are often black frames, title cards, or EOF issues.
    """
    safe_percentages = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]

    frame_count = max(1, min(frame_count, len(safe_percentages)))

    return [round(duration * pct, 2) for pct in safe_percentages[:frame_count]]


def extract_frames(
    video_path: Path,
    record_id: str,
    frame_count: int,
    frame_quality: int,
    max_width: int
):
    metadata = get_video_metadata(video_path)
    duration = metadata["duration_seconds"]

    if duration <= 0:
        raise RuntimeError("Invalid video duration.")

    # Safety caps for Render memory/disk.
    frame_count = max(1, min(frame_count, 8))
    frame_quality = max(2, min(frame_quality, 31))
    max_width = max(480, min(max_width, 1920))

    timestamps = get_frame_timestamps(duration, frame_count)

    frame_urls = []

    for i, timestamp in enumerate(timestamps, start=1):
        frame_name = f"{record_id}_frame_{i:02d}_{uuid.uuid4().hex[:8]}.jpg"
        frame_path = FRAME_DIR / frame_name

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", f"scale='min({max_width},iw)':-2",
            "-q:v", str(frame_quality),
            str(frame_path)
        ]

        run_cmd(cmd)

        if not frame_path.exists() or frame_path.stat().st_size == 0:
            raise RuntimeError(f"Frame extraction failed at timestamp {timestamp}")

        frame_urls.append(f"/frames/{frame_name}")

    metadata["frame_timestamps_seconds"] = timestamps
    metadata["frames_analyzed_count"] = len(frame_urls)

    return metadata, frame_urls


class ExtractRequest(BaseModel):
    # New Make payload field
    download_url: Optional[str] = None

    # Backward-compatible field from your original version
    video_url: Optional[str] = None

    # Helpful metadata from Make
    google_drive_file_id: Optional[str] = None
    asset_name: str = ""
    record_id: str = ""
    mime_type: Optional[str] = None
    file_size_mb: Optional[str] = None

    # Controls from Make
    frame_count: int = 6
    frame_quality: int = 70
    max_width: int = 1280


@app.post("/extract")
async def extract(body: ExtractRequest, request: Request):
    record_id = body.record_id or str(uuid.uuid4())

    # Support both the new Make field and your old field.
    video_url = body.download_url or body.video_url

    if not video_url:
        raise HTTPException(
            status_code=400,
            detail="download_url or video_url is required"
        )

    job_id = str(uuid.uuid4())
    video_path = TMP_DIR / f"{job_id}.mp4"

    try:
        download_video(video_url, video_path)

        metadata, frame_paths = extract_frames(
            video_path=video_path,
            record_id=record_id,
            frame_count=body.frame_count,
            frame_quality=body.frame_quality,
            max_width=body.max_width
        )

        base_url = str(request.base_url).rstrip("/")
        frame_urls = [base_url + p for p in frame_paths]

        return {
            "success": True,
            "record_id": record_id,
            "asset_name": body.asset_name,
            "google_drive_file_id": body.google_drive_file_id,
            "mime_type": body.mime_type,
            "file_size_mb": body.file_size_mb,
            **metadata,
            "frames": frame_urls
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "record_id": record_id,
                "asset_name": body.asset_name,
                "error": str(e)
            }
        )

    finally:
        if video_path.exists():
            video_path.unlink()


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "ChicagoShining Frame Extractor"
    }
