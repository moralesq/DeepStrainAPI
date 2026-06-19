#!/usr/bin/env python3

import base64
import os
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.request import urlopen

import nibabel as nib
import numpy as np
import replicate
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from replicate.exceptions import ReplicateError


INDEX_HTML = Path(__file__).with_name("index.html")

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="DeepStrain API")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(filename: str) -> str:
    return Path(filename).name.replace("/", "_").replace("\\", "_")


def deployment_ref() -> str | None:
    return os.environ.get("DEEPSTRAIN_REPLICATE_DEPLOYMENT")


def model_ref() -> str | None:
    return os.environ.get("DEEPSTRAIN_REPLICATE_MODEL")


def version_ref() -> str | None:
    explicit = os.environ.get("DEEPSTRAIN_REPLICATE_VERSION")
    if explicit:
        return explicit

    model = model_ref()
    if not model or ":" not in model:
        return None

    _, version = model.split(":", 1)
    return version or None


def replicate_ready() -> tuple[bool, str, str]:
    if not os.environ.get("REPLICATE_API_TOKEN"):
        return False, "Missing REPLICATE_API_TOKEN on the server.", ""
    deployment = deployment_ref()
    if deployment:
        return True, deployment, "deployment"
    version = version_ref()
    if version:
        return True, version, "version"
    model = model_ref()
    if model:
        return True, model.split(":", 1)[0], "model"
    return (
        False,
        "Missing DEEPSTRAIN_REPLICATE_DEPLOYMENT, DEEPSTRAIN_REPLICATE_VERSION, or DEEPSTRAIN_REPLICATE_MODEL on the server.",
        "",
    )


def external_url(url: str | None) -> str | None:
    return url or None


def make_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "filename": job["filename"],
        "source_url": job.get("source_url"),
        "uploaded_at": job["uploaded_at"],
        "analysis_status": job["analysis_status"],
        "analysis_log": job["analysis_log"],
        "prediction_id": job.get("prediction_id"),
        "prediction_url": external_url(job.get("prediction_url")),
        "uploaded_cine_url": None,
        "segmentation_url": external_url(job.get("segmentation_url")),
        "downloaded_segmentation_url": None,
        "overlay_preview_url": f"/artifacts/{job['id']}/overlay_preview" if job.get("preview_svg") else None,
        "preview_info": job.get("preview_info"),
    }


def read_job(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        return make_job_payload(job)


def get_job_raw(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def append_log(job_id: str, text: str) -> None:
    with JOBS_LOCK:
        JOBS[job_id]["analysis_log"] += text


def update_job(job_id: str, **changes: Any) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(changes)


def extract_output_url(output: Any) -> str | None:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        for item in output:
            url = extract_output_url(item)
            if url:
                return url
        return None
    if isinstance(output, dict):
        for value in output.values():
            url = extract_output_url(value)
            if url:
                return url
        return None
    return None


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        destination.write_bytes(response.read())
    return destination


def normalize_to_uint8(image_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_2d, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    values = arr[finite]
    lower = float(np.percentile(values, 1))
    upper = float(np.percentile(values, 99))
    if upper <= lower:
        lower = float(values.min())
        upper = float(values.max())
    if upper <= lower:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = np.clip((arr - lower) / (upper - lower), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def grayscale_bmp_bytes(image_2d_uint8: np.ndarray) -> bytes:
    height, width = image_2d_uint8.shape
    row_size = width * 3
    padding = (4 - (row_size % 4)) % 4
    pixel_rows: list[bytes] = []
    for row in image_2d_uint8[::-1]:
        rgb = bytearray()
        for value in row:
            gray = int(value)
            rgb.extend((gray, gray, gray))
        rgb.extend(b"\x00" * padding)
        pixel_rows.append(bytes(rgb))

    pixel_data = b"".join(pixel_rows)
    file_size = 54 + len(pixel_data)
    header = bytearray()
    header.extend(b"BM")
    header.extend(file_size.to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend((54).to_bytes(4, "little"))
    header.extend((40).to_bytes(4, "little"))
    header.extend(width.to_bytes(4, "little", signed=True))
    header.extend(height.to_bytes(4, "little", signed=True))
    header.extend((1).to_bytes(2, "little"))
    header.extend((24).to_bytes(2, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend(len(pixel_data).to_bytes(4, "little"))
    header.extend((2835).to_bytes(4, "little"))
    header.extend((2835).to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))
    return bytes(header) + pixel_data


def binary_boundary(mask_2d: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_2d, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return np.zeros(mask.shape, dtype=bool)

    eroded = mask.copy()
    eroded[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return mask & ~eroded


def choose_preview_indices(cine_shape: tuple[int, ...], seg_proxy: Any) -> tuple[int, int]:
    seg_shape = tuple(int(dim) for dim in seg_proxy.shape)
    if len(seg_shape) == 4:
        active_times: list[int] = []
        for time_idx in range(seg_shape[3]):
            seg_volume = np.asarray(seg_proxy[:, :, :, time_idx])
            if np.any(seg_volume > 0):
                active_times.append(time_idx)
        time_idx = active_times[len(active_times) // 2] if active_times else 0
        seg_volume = np.asarray(seg_proxy[:, :, :, time_idx])
    else:
        time_idx = 0
        seg_volume = np.asarray(seg_proxy[:, :, :])

    seg_slice_mask = np.any(seg_volume > 0, axis=(0, 1))
    slice_indices = np.flatnonzero(seg_slice_mask)
    if len(slice_indices):
        slice_idx = int(slice_indices[len(slice_indices) // 2])
    else:
        slice_idx = int(seg_shape[2] // 2)

    if len(cine_shape) == 4 and time_idx >= cine_shape[3]:
        time_idx = 0

    return slice_idx, time_idx


def render_overlay_preview(
    cine_path: Path,
    segmentation_path: Path,
    preview_path: Path,
) -> dict[str, int | list[int]]:
    cine_img = nib.load(str(cine_path))
    seg_img = nib.load(str(segmentation_path))
    cine_shape = tuple(int(dim) for dim in cine_img.shape)
    seg_shape = tuple(int(dim) for dim in seg_img.shape)

    if len(cine_shape) not in {3, 4}:
        raise RuntimeError(f"Expected 3D or 4D cine NIfTI, got shape {cine_shape}.")
    if len(seg_shape) not in {3, 4}:
        raise RuntimeError(f"Expected 3D or 4D segmentation NIfTI, got shape {seg_shape}.")
    if cine_shape[:3] != seg_shape[:3]:
        raise RuntimeError(
            f"Cine and segmentation spatial shapes do not match: {cine_shape[:3]} vs {seg_shape[:3]}."
        )

    seg_proxy = seg_img.dataobj
    cine_proxy = cine_img.dataobj
    slice_idx, time_idx = choose_preview_indices(cine_shape, seg_proxy)

    if len(cine_shape) == 4:
        cine_slice = np.asarray(cine_proxy[:, :, slice_idx, time_idx])
    else:
        cine_slice = np.asarray(cine_proxy[:, :, slice_idx])

    if len(seg_shape) == 4:
        seg_slice = np.asarray(seg_proxy[:, :, slice_idx, time_idx])
    else:
        seg_slice = np.asarray(seg_proxy[:, :, slice_idx])

    image_uint8 = normalize_to_uint8(cine_slice)
    bmp_base64 = base64.b64encode(grayscale_bmp_bytes(image_uint8)).decode("ascii")

    label_values = [int(value) for value in np.unique(seg_slice) if value > 0]
    colors = ["#ff4d4f", "#00c853", "#00bcd4", "#ffb300", "#7c4dff", "#ff6f61"]
    overlay_parts: list[str] = []
    for index, label in enumerate(label_values):
        boundary = binary_boundary(np.isclose(seg_slice, label))
        points = np.argwhere(boundary)
        color = colors[index % len(colors)]
        for row, col in points:
            overlay_parts.append(
                f'<rect x="{int(col)}" y="{int(row)}" width="1" height="1" fill="{color}" />'
            )

    height, width = image_uint8.shape
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">'
            f'<image href="data:image/bmp;base64,{bmp_base64}" x="0" y="0" width="{width}" height="{height}" />'
            f'<g shape-rendering="crispEdges">{"".join(overlay_parts)}</g>'
            "</svg>"
        ),
        encoding="utf-8",
    )

    return {
        "slice_index": int(slice_idx),
        "time_index": int(time_idx),
        "shape": [int(dim) for dim in cine_shape],
        "labels": label_values,
    }


def create_prediction_with_fallback(
    client: replicate.Client,
    target: str,
    target_kind: str,
    input_value: str | Path,
    job_id: str,
):
    if target_kind == "deployment":
        return client.deployments.predictions.create(
            deployment=target,
            input={"cine": input_value},
            wait=False,
        )

    if target_kind == "version":
        return client.predictions.create(
            version=target,
            input={"cine": input_value},
            wait=False,
        )

    try:
        return client.models.predictions.create(
            model=target,
            input={"cine": input_value},
            wait=False,
        )
    except ReplicateError as exc:
        if exc.status != 404:
            raise

        append_log(
            job_id,
            f"Model endpoint returned 404 for {target}. Falling back to latest model version.\n",
        )
        model = client.models.get(target)
        latest_version = getattr(model, "latest_version", None)
        if latest_version is None or not getattr(latest_version, "id", None):
            raise RuntimeError(
                f"Replicate model {target} has no latest_version available for fallback."
            ) from exc

        append_log(job_id, f"Resolved latest version: {latest_version.id}\n")
        return client.predictions.create(
            version=latest_version.id,
            input={"cine": input_value},
            wait=False,
        )


def analysis_worker(job_id: str) -> None:
    ready, target_or_error, target_kind = replicate_ready()
    if not ready:
        update_job(job_id, analysis_status="failed")
        append_log(job_id, target_or_error + "\n")
        return

    update_job(job_id, analysis_status="running", analysis_log="")
    job = get_job_raw(job_id)
    assert job is not None
    source_url = job.get("source_url")
    input_value: str | Path
    if source_url:
        input_value = source_url
        append_log(job_id, f"Using hosted cine URL: {source_url}\n")
    else:
        input_value = Path(f"/tmp/{job_id}_{safe_name(job['filename'])}")
        input_bytes = job.get("input_bytes")
        if not input_bytes:
            update_job(job_id, analysis_status="failed")
            append_log(job_id, "Uploaded cine bytes are missing from the job state.\n")
            return
        input_value.write_bytes(input_bytes)
    target = target_or_error

    try:
        append_log(job_id, f"Submitting cine to Replicate {target_kind} {target}\n")
        client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
        prediction = create_prediction_with_fallback(
            client=client,
            target=target,
            target_kind=target_kind,
            input_value=input_value,
            job_id=job_id,
        )
        update_job(
            job_id,
            prediction_id=prediction.id,
            prediction_url=(prediction.urls or {}).get("get"),
        )
        append_log(job_id, f"Prediction created: {prediction.id}\n")

        previous_logs = prediction.logs or ""
        if previous_logs:
            append_log(job_id, previous_logs + "\n")

        while prediction.status not in {"succeeded", "failed", "canceled"}:
            prediction.reload()
            new_logs = prediction.logs or ""
            if len(new_logs) > len(previous_logs):
                append_log(job_id, new_logs[len(previous_logs):])
                previous_logs = new_logs

        if prediction.status != "succeeded":
            append_log(job_id, f"\nPrediction ended with status={prediction.status}\n")
            if prediction.error:
                append_log(job_id, prediction.error + "\n")
            update_job(job_id, analysis_status="failed")
            return

        segmentation_url = extract_output_url(prediction.output)
        if not segmentation_url:
            append_log(job_id, "\nPrediction succeeded, but no output URL was returned.\n")
            update_job(job_id, analysis_status="failed")
            return

        with TemporaryDirectory(prefix=f"deepstrain_{job_id}_") as temp_dir:
            temp_root = Path(temp_dir)
            cine_path = temp_root / safe_name(job["filename"])
            segmentation_path = temp_root / "segmentation_output.nii.gz"
            overlay_path = temp_root / "center_slice_overlay.svg"

            if source_url:
                append_log(job_id, f"Downloading hosted cine to {cine_path.name}\n")
                download_file(source_url, cine_path)
            else:
                cine_path.write_bytes(job["input_bytes"])

            append_log(job_id, f"Downloading segmentation to {segmentation_path.name}\n")
            download_file(segmentation_url, segmentation_path)

            cine_img = nib.load(str(cine_path))
            seg_img = nib.load(str(segmentation_path))
            append_log(
                job_id,
                (
                    f"Cine header shape: {cine_img.shape}, "
                    f"dtype={cine_img.get_data_dtype()}\n"
                ),
            )
            append_log(
                job_id,
                (
                    f"Segmentation header shape: {seg_img.shape}, "
                    f"dtype={seg_img.get_data_dtype()}\n"
                ),
            )
            append_log(job_id, "Rendering center-slice contour preview\n")
            preview_info = render_overlay_preview(cine_path, segmentation_path, overlay_path)
            preview_svg = overlay_path.read_text(encoding="utf-8")

        update_job(
            job_id,
            analysis_status="complete",
            segmentation_url=segmentation_url,
            preview_info=preview_info,
            preview_svg=preview_svg,
            input_bytes=None,
        )
        append_log(job_id, f"\nSegmentation ready: {segmentation_url}\n")
        append_log(
            job_id,
            (
                f"Preview slice z={preview_info['slice_index']}, "
                f"t={preview_info['time_index']}, labels={preview_info['labels']}\n"
            ),
        )
    except ReplicateError as exc:
        append_log(job_id, f"Replicate API error ({exc.status}): {exc.detail}\n")
        update_job(job_id, analysis_status="failed")
    except Exception:
        append_log(job_id, traceback.format_exc())
        update_job(job_id, analysis_status="failed")
    finally:
        if not source_url and isinstance(input_value, Path) and input_value.exists():
            input_value.unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    ready, target_or_error, target_kind = replicate_ready()
    return {
        "ok": True,
        "replicate_ready": ready,
        "target": target_or_error if ready else None,
        "target_kind": target_kind if ready else None,
    }


@app.get("/api/config")
def api_config() -> JSONResponse:
    ready, target_or_error, target_kind = replicate_ready()
    if ready:
        return JSONResponse({"ready": True, "target": target_or_error, "target_kind": target_kind})
    return JSONResponse({"ready": False, "error": target_or_error})


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> JSONResponse:
    filename = safe_name(file.filename or "cine.nii.gz")
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        raise HTTPException(status_code=400, detail="Expected a .nii or .nii.gz file.")

    job_id = uuid.uuid4().hex[:10]
    input_bytes = await file.read()

    job = {
        "id": job_id,
        "filename": filename,
        "source_url": None,
        "uploaded_at": now_iso(),
        "analysis_status": "uploaded",
        "analysis_log": "",
        "prediction_id": None,
        "prediction_url": None,
        "segmentation_url": None,
        "preview_info": None,
        "preview_svg": None,
        "input_bytes": input_bytes,
    }

    with JOBS_LOCK:
        JOBS[job_id] = job

    return JSONResponse(make_job_payload(job))


@app.post("/api/jobs/url")
def create_job_from_url(payload: dict[str, Any]) -> JSONResponse:
    source_url = str(payload.get("url", "")).strip()
    if not source_url.startswith("http://") and not source_url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Expected a public http(s) cine URL.")

    filename = Path(source_url.split("?", 1)[0]).name or "cine.nii.gz"
    job_id = uuid.uuid4().hex[:10]
    job = {
        "id": job_id,
        "filename": filename,
        "source_url": source_url,
        "uploaded_at": now_iso(),
        "analysis_status": "uploaded",
        "analysis_log": "",
        "prediction_id": None,
        "prediction_url": None,
        "segmentation_url": None,
        "preview_info": None,
        "preview_svg": None,
        "input_bytes": None,
    }

    with JOBS_LOCK:
        JOBS[job_id] = job

    return JSONResponse(make_job_payload(job))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    payload = read_job(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(payload)


@app.post("/api/jobs/{job_id}/analysis")
def start_analysis(job_id: str) -> JSONResponse:
    job = get_job_raw(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["analysis_status"] == "running":
        raise HTTPException(status_code=409, detail="Analysis already running.")

    thread = threading.Thread(target=analysis_worker, args=(job_id,), daemon=True)
    thread.start()
    payload = read_job(job_id)
    assert payload is not None
    return JSONResponse(payload)


@app.get("/artifacts/{job_id}/{key}")
def get_artifact(job_id: str, key: str) -> Response:
    job = get_job_raw(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if key != "overlay_preview" or not job.get("preview_svg"):
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return Response(content=job["preview_svg"], media_type="image/svg+xml")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "10000")),
        reload=False,
    )
