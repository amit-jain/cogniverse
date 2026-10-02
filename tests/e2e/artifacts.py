"""Image, audio and PDF test artifacts derived from tracked repository data."""

from __future__ import annotations

import tempfile
from pathlib import Path


def _atomic_artifact(dest: Path, writer) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dest.parent,
        prefix=f".{dest.name}.",
        delete=False,
    ) as handle:
        staged = Path(handle.name)
    try:
        writer(staged)
        if not staged.exists() or staged.stat().st_size == 0:
            raise RuntimeError(f"E2E artifact writer produced an empty file: {dest}")
        staged.replace(dest)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return dest


def _write_pdf_fixture(dest: Path, text: str) -> Path:
    lines = []
    for line in text.splitlines():
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        lines.append(f"({escaped}) Tj")
    content = "BT\n/F1 12 Tf\n72 720 Td\n14 TL\n" + "\nT*\n".join(lines) + "\nET\n"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        f"<< /Length {len(content.encode('latin-1'))} >>\nstream\n{content}endstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, value in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{object_number} 0 obj\n{value}\nendobj\n".encode("latin-1"))
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return _atomic_artifact(dest, lambda staged: staged.write_bytes(payload))


def _extract_image_fixture(source_video: Path, dest: Path) -> Path:
    if not source_video.exists():
        raise FileNotFoundError(f"E2E source video does not exist: {source_video}")

    def write_image(staged: Path) -> None:
        import av

        with av.open(str(source_video)) as container:
            for frame in container.decode(video=0):
                frame.to_image().save(staged, format="JPEG", quality=92)
                return
        raise RuntimeError(
            f"E2E source video contains no decodable frame: {source_video}"
        )

    return _atomic_artifact(dest, write_image)


def _extract_audio_fixture(
    source_video: Path, dest: Path, duration_seconds: int = 10
) -> Path:
    if not source_video.exists():
        raise FileNotFoundError(f"E2E source video does not exist: {source_video}")

    def write_audio(staged: Path) -> None:
        import wave

        import av
        import numpy as np

        target_rate = 16_000
        required_samples = target_rate * duration_seconds
        chunks: list[np.ndarray] = []
        collected = 0
        with av.open(str(source_video)) as container:
            audio_streams = [
                stream for stream in container.streams if stream.type == "audio"
            ]
            if len(audio_streams) != 1:
                raise RuntimeError(
                    f"E2E source video must contain exactly one audio stream: {source_video}"
                )
            resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
            for frame in container.decode(audio_streams[0]):
                for resampled in resampler.resample(frame):
                    samples = resampled.to_ndarray().reshape(-1)
                    chunks.append(samples)
                    collected += samples.size
                    if collected >= required_samples:
                        break
                if collected >= required_samples:
                    break
        if collected < required_samples:
            raise RuntimeError(
                f"E2E source video yielded {collected} audio samples; "
                f"expected {required_samples}: {source_video}"
            )
        samples = np.concatenate(chunks)[:required_samples].astype(np.int16, copy=False)
        with wave.open(str(staged), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(target_rate)
            wav.writeframes(samples.tobytes())

    return _atomic_artifact(dest, write_audio)
