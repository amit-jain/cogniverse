"""Chunked transcription against an OpenAI-compatible Whisper endpoint.

The client splits the audio into the chunks vLLM's Whisper server would cut
itself (at most 30 s, each cut at the quietest 0.1 s window of the chunk's last
second) and sends one request per chunk, so every chunk's answer is checked on
its own. vLLM builds a ``verbose_json`` transcript only from text between
timestamp tokens, so a decode that emits none comes back as HTTP 200 with an
empty transcript: intermittently for any audio on the cluster's ROCm server,
and every time for some audio (repeated sounds, some speech). Sent whole, a
long file loses such a chunk silently inside a transcript that is otherwise
complete. A chunk whose loudest 25 ms frame reaches ``SILENCE_FLOOR_DBFS`` and
comes back empty is asked again, the last of ``TRANSCRIBE_ATTEMPTS`` requests
without timestamps (one segment spanning the chunk), and then raises
``EmptyTranscriptError``. A silent chunk may come back empty.
"""

from __future__ import annotations

import io
import logging
import math
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

WHISPER_SAMPLE_RATE = 16000
# vLLM's Whisper speech-to-text config: max_audio_clip_s, overlap_chunk_second
# and min_energy_split_window_size.
CHUNK_SECONDS = 30
SPLIT_SEARCH_SECONDS = 1
SPLIT_WINDOW_SAMPLES = 1600
# Loudest 25 ms frame below which a chunk counts as silence. Speech scaled to
# -60 dBFS RMS (loudest frames near -40 dBFS) still transcribes, and the model
# answers digital silence with text rather than nothing.
SILENCE_FLOOR_DBFS = -60.0
FRAME_SAMPLES = 400
# Requests per chunk before an empty answer for sound raises; the last asks
# without timestamps. Measured on the cluster's vLLM ROCm Whisper: 3-7% of
# verbose_json requests come back empty at random, and some chunks every time.
TRANSCRIBE_ATTEMPTS = 3
# Languages vLLM joins without a space between chunk texts.
NO_SPACE_LANGUAGES = frozenset({"ja", "zh"})


class EmptyTranscriptError(RuntimeError):
    """A chunk carrying sound came back empty on every attempt."""

    def __init__(
        self,
        source: str,
        chunk_index: int,
        start_s: float,
        end_s: float,
        loudest_frame_dbfs: float,
        attempts: int,
    ) -> None:
        self.source = source
        self.chunk_index = chunk_index
        self.start_s = start_s
        self.end_s = end_s
        self.loudest_frame_dbfs = loudest_frame_dbfs
        self.attempts = attempts
        super().__init__(
            f"{source}: chunk {chunk_index} ({start_s:.2f}-{end_s:.2f}s, loudest "
            f"frame {loudest_frame_dbfs:.1f} dBFS) carries sound but came back "
            f"with an empty transcript on all {attempts} attempts"
        )


@dataclass(frozen=True)
class AudioChunk:
    """One request's worth of 16 kHz mono PCM16 audio."""

    index: int
    start_sample: int
    samples: np.ndarray

    @property
    def start_s(self) -> float:
        return self.start_sample / WHISPER_SAMPLE_RATE

    @property
    def end_s(self) -> float:
        return (self.start_sample + len(self.samples)) / WHISPER_SAMPLE_RATE

    @property
    def loudest_frame_dbfs(self) -> float:
        return loudest_frame_dbfs(self.samples)

    @property
    def carries_sound(self) -> bool:
        return self.loudest_frame_dbfs >= SILENCE_FLOOR_DBFS

    def wav(self) -> bytes:
        return wav_bytes(self.samples)


@dataclass(frozen=True)
class ChunkTranscript:
    """One chunk's answer: its text, language (``None`` when the answer names
    none) and chunk-relative segments."""

    text: str
    language: Optional[str]
    segments: List[Dict[str, Any]]


def response_format(timestamps: bool) -> str:
    """The ``response_format`` a request with or without timestamps asks for."""
    return "verbose_json" if timestamps else "json"


def lenient_chunk_answer(body: Any, chunk: AudioChunk) -> ChunkTranscript:
    """Read a ``verbose_json`` or ``json`` answer, tolerating omitted fields.

    An answer with text but no segments gets one segment spanning the chunk.
    """
    text = body.get("text") or ""
    segments = [
        {
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "text": (segment.get("text") or "").strip(),
        }
        for segment in body.get("segments") or []
        if isinstance(segment, dict)
    ]
    if not segments and text.strip():
        segments = [
            {
                "start": 0.0,
                "end": float(body.get("duration") or chunk.end_s - chunk.start_s),
                "text": text.strip(),
            }
        ]
    return ChunkTranscript(
        text=text, language=body.get("language") or None, segments=segments
    )


def decode_audio(path: Path) -> np.ndarray:
    """The first audio stream of ``path`` as 16 kHz mono PCM16 samples."""
    import av

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"{path}: no audio stream present")
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=WHISPER_SAMPLE_RATE
        )
        frames = [
            resampled.to_ndarray().reshape(-1)
            for frame in container.decode(stream)
            for resampled in resampler.resample(frame)
        ]
        frames.extend(
            resampled.to_ndarray().reshape(-1) for resampled in resampler.resample(None)
        )
    if not frames:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(frames).astype(np.int16, copy=False)


def pcm16_wav_samples(wav: bytes) -> np.ndarray:
    """The samples of a 16 kHz mono PCM16 WAV."""
    with wave.open(io.BytesIO(wav), "rb") as reader:
        shape = (
            reader.getnchannels(),
            reader.getsampwidth(),
            reader.getframerate(),
        )
        if shape != (1, 2, WHISPER_SAMPLE_RATE):
            raise ValueError(
                "expected 16 kHz mono PCM16 WAV, got "
                f"{shape[0]} channel(s), {8 * shape[1]}-bit, {shape[2]} Hz"
            )
        return np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16)


def wav_bytes(samples: np.ndarray) -> bytes:
    """``samples`` as a 16 kHz mono PCM16 WAV."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(WHISPER_SAMPLE_RATE)
        writer.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buffer.getvalue()


def loudest_frame_dbfs(samples: np.ndarray) -> float:
    """RMS level of the loudest 25 ms frame, in dBFS; -inf for silence."""
    if len(samples) == 0:
        return -math.inf
    padded = np.zeros(-(-len(samples) // FRAME_SAMPLES) * FRAME_SAMPLES)
    padded[: len(samples)] = np.asarray(samples, dtype=np.float64) / 32768.0
    frames = padded.reshape(-1, FRAME_SAMPLES)
    loudest = float(np.sqrt(np.mean(frames**2, axis=1)).max())
    return 20.0 * math.log10(loudest) if loudest > 0 else -math.inf


def _quietest_window_start(audio: np.ndarray, start: int, end: int) -> int:
    segment = audio[start:end]
    min_energy = math.inf
    quietest = 0
    for offset in range(0, len(segment) - SPLIT_WINDOW_SAMPLES, SPLIT_WINDOW_SAMPLES):
        window = segment[offset : offset + SPLIT_WINDOW_SAMPLES]
        energy = (window**2).mean() ** 0.5
        if energy < min_energy:
            quietest = offset + start
            min_energy = energy
    return quietest


def split_for_whisper(samples: np.ndarray) -> List[AudioChunk]:
    """Cut ``samples`` where vLLM's Whisper server cuts a long file.

    Audio of at most 30 s is one chunk. Longer audio is cut at most every
    30 s, each cut at the start of the quietest 0.1 s window in that chunk's
    last second, computed on float32 samples as the server computes it.
    """
    samples = np.asarray(samples, dtype=np.int16)
    if len(samples) <= CHUNK_SECONDS * WHISPER_SAMPLE_RATE:
        return [AudioChunk(0, 0, samples)]
    audio = samples.astype(np.float32) / np.float32(32768.0)
    chunk_size = CHUNK_SECONDS * WHISPER_SAMPLE_RATE
    search_size = SPLIT_SEARCH_SECONDS * WHISPER_SAMPLE_RATE
    chunks: List[AudioChunk] = []
    start = 0
    while start < len(samples):
        if start + chunk_size >= len(samples):
            chunks.append(AudioChunk(len(chunks), start, samples[start:]))
            break
        cut = _quietest_window_start(
            audio, start + chunk_size - search_size, start + chunk_size
        )
        chunks.append(AudioChunk(len(chunks), start, samples[start:cut]))
        start = cut
    return chunks


def transcribe_in_chunks(
    samples: np.ndarray,
    transcribe_chunk: Callable[[AudioChunk, Optional[str], bool], ChunkTranscript],
    *,
    language: Optional[str],
    source: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Transcribe ``samples`` one chunk per request and merge the answers.

    ``transcribe_chunk(chunk, language, timestamps)`` sends one chunk and
    parses the answer; a ``None`` language asks the server to detect it, and
    ``timestamps`` selects ``response_format(timestamps)``. As the server does
    for a whole file, the language the first answer names is sent with every
    later chunk when the caller names none.

    Returns ``full_text``, ``language``, ``duration`` (seconds) and
    ``segments`` with ``start``/``end`` relative to the whole audio.
    """
    chunks = split_for_whisper(samples)
    texts: List[str] = []
    segments: List[Dict[str, Any]] = []
    detected: Optional[str] = language
    for chunk in chunks:
        answer = _transcribe_checked(chunk, transcribe_chunk, detected, source, logger)
        if detected is None:
            detected = answer.language
        if answer.text.strip():
            texts.append(answer.text)
        segments.extend(
            dict(
                segment,
                start=chunk.start_s + float(segment["start"]),
                end=chunk.start_s + float(segment["end"]),
            )
            for segment in answer.segments
        )
    separator = "" if (detected or "").lower() in NO_SPACE_LANGUAGES else " "
    return {
        "full_text": separator.join(texts).strip(),
        "language": detected or "unknown",
        "duration": len(samples) / WHISPER_SAMPLE_RATE,
        "segments": segments,
    }


def _transcribe_checked(
    chunk: AudioChunk,
    transcribe_chunk: Callable[[AudioChunk, Optional[str], bool], ChunkTranscript],
    language: Optional[str],
    source: str,
    logger: logging.Logger,
) -> ChunkTranscript:
    loudest = chunk.loudest_frame_dbfs
    named = language
    for attempt in range(1, TRANSCRIBE_ATTEMPTS + 1):
        answer = transcribe_chunk(chunk, language, attempt < TRANSCRIBE_ATTEMPTS)
        # An untimed answer names no language; an empty one before it did.
        named = answer.language or named
        if answer.text.strip() or loudest < SILENCE_FLOOR_DBFS:
            return replace(answer, language=named)
        logger.warning(
            "%s: chunk %d (%.2f-%.2fs, loudest frame %.1f dBFS) came back with an "
            "empty transcript on attempt %d of %d",
            source,
            chunk.index,
            chunk.start_s,
            chunk.end_s,
            loudest,
            attempt,
            TRANSCRIBE_ATTEMPTS,
        )
    raise EmptyTranscriptError(
        source, chunk.index, chunk.start_s, chunk.end_s, loudest, TRANSCRIBE_ATTEMPTS
    )
