"""Chunked transcription against an OpenAI-compatible Whisper endpoint.

The client splits the audio into the chunks vLLM's Whisper server would cut
itself (at most 30 s, each cut at the quietest 0.1 s window of the chunk's last
second) and asks each chunk for ``verbose_json`` (timings). vLLM builds that
transcript only from text between two adjacent timestamp tokens: a decode with
no such pair comes back empty, text decoded after the last pair is dropped, and
a decode that ends on a pair before the end of the chunk leaves the rest
undecoded. Unless the timed segments run to the end of the chunk, the chunk is
also asked for ``json``, which keeps the whole decode, and ``align_text`` times
its words with the timed segments.

An answer that is a repetition loop (compression ratio above
``GARBLED_COMPRESSION_RATIO``), or empty for a chunk whose loudest 25 ms frame
reaches ``SILENCE_FLOOR_DBFS``, is asked again at the next of
``FALLBACK_TEMPERATURES``. Speech that itself repeats compresses like a loop,
but unlike a decode loop it comes back in the same words however it is
sampled. So once a chunk's ``json`` answer at 0.0 has looped, an answer at
``LOOP_BREAKING_TEMPERATURE`` or above is compared with it before any is kept:
one that agrees keeps the 0.0 answer as repeated speech (``repeated_speech``);
otherwise the first that did not loop is kept, after both loop-breaking
temperatures are asked. A chunk with no usable ``json`` answer and no usable
timed answer running to its end raises ``GarbledTranscriptError`` (some answer
looped) or ``EmptyTranscriptError``; one whose ``verbose_json`` answer is
never usable keeps its text as one segment spanning the chunk. A silent chunk
may come back empty.

A chunk's first ``verbose_json`` answer carrying per-segment ``avg_logprob``,
asked at or below ``NON_SPEECH_JUDGED_TEMPERATURE``, is improbable when they
average below ``NON_SPEECH_AVG_LOGPROB``. The chunk is then not speech, and
gets an empty transcript, unless its ``json`` answer carries other speech
words; then no timed answer is used and its text spans the chunk.
``transcribe_in_chunks(drop_garbled=True)`` leaves out a chunk that raises
``GarbledTranscriptError`` and records it under ``dropped_chunks``, raising
only when every chunk does.
"""

from __future__ import annotations

import io
import logging
import math
import re
import wave
import zlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

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
# Whisper's own temperature fallback: an unusable answer is asked again at the
# next temperature. Asked again at 0 a loop repeats; on the cluster's server
# the measured loops all broke by 0.8.
FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TRANSCRIBE_ATTEMPTS = len(FALLBACK_TEMPERATURES)
# Sent with every request, so an answer above temperature 0 is reproducible.
SAMPLING_SEED = 0
# An answer whose text compresses better than this is a repetition loop
# (Whisper's threshold). Measured: answers at most 1.82, loops at least 3.07.
GARBLED_COMPRESSION_RATIO = 2.4
# The temperature by which sampling breaks a decode loop: of 15 recorded loops
# on the cluster's server, none was still a loop at 0.8 or 1.0, and their
# answers there shared at most 0.27 of their words with the loop.
LOOP_BREAKING_TEMPERATURE = 0.8
# Word agreement (difflib ratio) with the temperature-0 answer at or above
# which an answer still repeating at LOOP_BREAKING_TEMPERATURE says the same
# thing. Measured on a ten-second clip played back to back: 0.60 at the tenth
# percentile of 41 such answers; a loop that formed at 0.8 instead scored 0.05.
REPEATED_SPEECH_AGREEMENT = 0.5
# Mean token log-probability of a chunk's timed answer, weighted by segment
# duration, below which the chunk holds no speech (Whisper's
# logprob_threshold). Measured on the cluster's server at 0.0: speech chunks
# at least -0.53; music the server answers with text at most -1.05, but for
# two "Thank you." answers at -0.79 and -0.91.
NON_SPEECH_AVG_LOGPROB = -1.0
# The highest temperature whose timed answer is judged against
# NON_SPEECH_AVG_LOGPROB: the first one carrying log-probabilities, which is
# the answer at 0.2 when the one at 0.0 came back empty. Measured: speech at
# 0.2 at least -0.41; at 0.4 sampling took a speech chunk to -1.38.
NON_SPEECH_JUDGED_TEMPERATURE = 0.2
# Whisper's timestamp step, in seconds.
TIMESTAMP_STEP_S = 0.02
# Languages vLLM joins without a space between chunk texts.
NO_SPACE_LANGUAGES = frozenset({"ja", "zh"})


class GarbledTranscriptError(RuntimeError):
    """A chunk's json answer was a repetition loop or empty on every attempt."""

    def __init__(
        self,
        source: str,
        chunk_index: int,
        start_s: float,
        end_s: float,
        compression_ratios: Tuple[float, ...],
        attempts: int,
    ) -> None:
        self.source = source
        self.chunk_index = chunk_index
        self.start_s = start_s
        self.end_s = end_s
        self.compression_ratios = compression_ratios
        self.attempts = attempts
        ratios = ", ".join(f"{ratio:.2f}" for ratio in compression_ratios)
        super().__init__(
            f"{source}: chunk {chunk_index} ({start_s:.2f}-{end_s:.2f}s) came back "
            f"unusable on all {attempts} attempts: {len(compression_ratios)} "
            f"repetition loops (compression ratio {ratios}, above "
            f"{GARBLED_COMPRESSION_RATIO}) and {attempts - len(compression_ratios)} "
            "empty"
        )


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
    none), chunk-relative segments and the ``segments_avg_logprob`` of the
    segments it came with (``None`` when they carry none)."""

    text: str
    language: Optional[str]
    segments: List[Dict[str, Any]]
    avg_logprob: Optional[float] = None


def segments_avg_logprob(segments: Any) -> Optional[float]:
    """The mean of the segments' ``avg_logprob``, weighted by their duration
    (at least one timestamp step); ``None`` when no segment carries one."""
    weighted = [
        (
            max(
                float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)),
                TIMESTAMP_STEP_S,
            ),
            float(segment["avg_logprob"]),
        )
        for segment in segments or []
        if isinstance(segment, dict)
        and isinstance(segment.get("avg_logprob"), (int, float))
        and not isinstance(segment.get("avg_logprob"), bool)
    ]
    if not weighted:
        return None
    return sum(weight * value for weight, value in weighted) / sum(
        weight for weight, _ in weighted
    )


def response_format(timestamps: bool) -> str:
    """The ``response_format`` a request with or without timestamps asks for."""
    return "verbose_json" if timestamps else "json"


def sampling_fields(temperature: float) -> Dict[str, str]:
    """The form fields that set a request's sampling temperature and seed."""
    return {"temperature": str(temperature), "seed": str(SAMPLING_SEED)}


def compression_ratio(text: str) -> float:
    """UTF-8 length over zlib-compressed length, as vLLM reports per segment;
    0.0 for no text."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


_NOT_WORD = re.compile(r"\W+")


def _units(text: str, no_space: bool) -> List[str]:
    if no_space:
        return [char for char in text if not char.isspace()]
    return text.split()


def _comparable(unit: str) -> str:
    return _NOT_WORD.sub("", unit.casefold())


def align_text(
    text: str,
    timed: List[Dict[str, Any]],
    duration: float,
    *,
    no_space: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Time the words of a json answer with a verbose_json answer's segments.

    Returns the chunk's text and its chunk-relative segments. Words are
    compared case- and punctuation-blind (characters for a no-space language).
    A json word matching a timed word takes its segment; a timed word the json
    answer lacks stays in its segment. Where the answers word the same stretch
    differently, the rendering with more words is kept (the json one on a
    tie), json words spread over the timed words' segments in order. A json
    word the timed answer lacks, falling in time no segment covers (before the
    first, between two that do not meet, after the last), gets a segment
    spanning that gap; elsewhere it joins the segment before it (the first
    segment at the start). Without timed segments the text is one segment
    spanning the chunk. No segment runs past ``duration``, so none reaches
    into the next chunk.
    """
    joiner = "" if no_space else " "
    words = _units(text, no_space)
    spans = [
        (
            clamp_to_duration(float(s["start"]), duration),
            clamp_to_duration(float(s["end"]), duration),
        )
        for s in timed
    ]
    timed_units = [
        (unit, index)
        for index, segment in enumerate(timed)
        for unit in _units(segment["text"], no_space)
    ]
    if not timed_units:
        merged = joiner.join(words)
        return merged, (
            [{"start": 0.0, "end": duration, "text": merged}] if words else []
        )

    owner = [index for _, index in timed_units]
    placed: List[Tuple[str, Tuple[float, float], Tuple[Any, ...]]] = []

    def in_segment(unit: str, position: int) -> None:
        placed.append((unit, spans[owner[position]], ("segment", owner[position])))

    matcher = SequenceMatcher(
        None,
        [_comparable(unit) for unit, _ in timed_units],
        [_comparable(word) for word in words],
        autojunk=False,
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(j2 - j1):
                in_segment(words[j1 + offset], i1 + offset)
        elif tag == "replace" and j2 - j1 >= i2 - i1:
            for j in range(j1, j2):
                in_segment(words[j], i1 + (j - j1) * (i2 - i1) // (j2 - j1))
        elif tag in ("replace", "delete"):
            for i in range(i1, i2):
                in_segment(timed_units[i][0], i)
        else:
            before = owner[i1 - 1] if i1 > 0 else None
            after = owner[i1] if i1 < len(owner) else None
            start = spans[before][1] if before is not None else 0.0
            end = spans[after][0] if after is not None else duration
            if before != after and end - start >= TIMESTAMP_STEP_S / 2:
                for j in range(j1, j2):
                    placed.append((words[j], (start, end), ("gap", before, after)))
            else:
                for j in range(j1, j2):
                    in_segment(words[j], i1 - 1 if before is not None else i1)

    segments: List[Dict[str, Any]] = []
    slot = None
    for unit, (start, end), key in placed:
        if key != slot:
            segments.append({"start": start, "end": end, "units": []})
            slot = key
        segments[-1]["units"].append(unit)
    return joiner.join(unit for unit, _, _ in placed), [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": joiner.join(segment["units"]),
        }
        for segment in segments
    ]


def reaches_chunk_end(segments: List[Dict[str, Any]], duration: float) -> bool:
    """Whether timed segments run to the end of a chunk of ``duration`` s.

    vLLM drops only text decoded after the last timestamp pair, so such an
    answer lost nothing and needs no json answer. The last timestamp can fall
    short of the end by less than ``TIMESTAMP_STEP_S``, since chunk lengths
    are not multiples of it; recorded answers end at or past the end, or at
    least 0.5 s short of it.
    """
    return (
        bool(segments)
        and duration - max(float(segment["end"]) for segment in segments)
        < TIMESTAMP_STEP_S
    )


def clamp_to_duration(seconds: float, duration: float) -> float:
    """``seconds`` capped at ``duration``, logging the original at DEBUG when
    it is past by half a timestamp step or more. vLLM computes timestamps as
    ``0.02 * n``, which can land a float's rounding past an equal duration
    (``0.02 * 1470`` is 29.400000000000002 against 29.4).
    """
    if seconds <= duration:
        return seconds
    if seconds - duration < TIMESTAMP_STEP_S / 2:
        return duration
    logger.debug(
        "segment time %.2fs is past the chunk's %.2fs; clamped", seconds, duration
    )
    return duration


def lenient_chunk_answer(body: Any, chunk: AudioChunk) -> ChunkTranscript:
    """Read a ``verbose_json`` or ``json`` answer, tolerating omitted fields.

    An answer with text but no segments gets one segment spanning the chunk.
    Segment times past the chunk's end, which Whisper gives for the padding
    after short audio, are clamped to it.
    """
    text = body.get("text") or ""
    duration = len(chunk.samples) / WHISPER_SAMPLE_RATE
    segments = [
        {
            "start": clamp_to_duration(float(segment.get("start", 0.0)), duration),
            "end": clamp_to_duration(float(segment.get("end", 0.0)), duration),
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
        text=text,
        language=body.get("language") or None,
        segments=segments,
        avg_logprob=segments_avg_logprob(body.get("segments")),
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


ChunkRequest = Callable[[AudioChunk, Optional[str], bool, float], ChunkTranscript]


def transcribe_in_chunks(
    samples: np.ndarray,
    transcribe_chunk: ChunkRequest,
    *,
    language: Optional[str],
    source: str,
    logger: logging.Logger,
    drop_garbled: bool = False,
) -> Dict[str, Any]:
    """Transcribe ``samples`` chunk by chunk and merge the answers.

    ``transcribe_chunk(chunk, language, timestamps, temperature)`` sends one
    chunk and parses the answer; a ``None`` language asks the server to detect
    it, ``timestamps`` selects ``response_format(timestamps)`` and
    ``temperature`` goes out with ``sampling_fields(temperature)``. Each chunk
    is asked with timestamps, then without, in the language the timed answer
    names when the caller names none; as the server does for a whole file,
    that language is sent with every later chunk.

    Returns ``full_text``, ``language``, ``duration`` (seconds) and
    ``segments`` with ``start``/``end`` relative to the whole audio. With
    ``drop_garbled``, a chunk raising ``GarbledTranscriptError`` contributes
    no text and is listed under ``dropped_chunks`` (``chunk_index``,
    ``start_s``, ``end_s``, ``compression_ratios``); the error is raised only
    when every chunk raises it.
    """
    chunks = split_for_whisper(samples)
    texts: List[str] = []
    segments: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    detected: Optional[str] = language
    for chunk in chunks:
        try:
            answer = _transcribe_checked(
                chunk, transcribe_chunk, detected, source, logger
            )
        except GarbledTranscriptError as garbled:
            if not drop_garbled:
                raise
            dropped.append(
                {
                    "chunk_index": garbled.chunk_index,
                    "start_s": garbled.start_s,
                    "end_s": garbled.end_s,
                    "compression_ratios": list(garbled.compression_ratios),
                }
            )
            if len(dropped) == len(chunks):
                raise
            logger.warning("%s: dropped from the transcript", garbled)
            continue
        if detected is None:
            detected = answer.language
        if answer.text:
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
    transcript: Dict[str, Any] = {
        "full_text": separator.join(texts),
        "language": detected or "unknown",
        "duration": len(samples) / WHISPER_SAMPLE_RATE,
        "segments": segments,
    }
    if drop_garbled:
        transcript["dropped_chunks"] = dropped
    return transcript


def word_agreement(first: str, second: str) -> float:
    """How much of two answers' wording is shared, in order: 1.0 for the same
    words, 0.0 for none in common or for an answer with no words."""
    left = [word for word in map(_comparable, _units(first, False)) if word]
    right = [word for word in map(_comparable, _units(second, False)) if word]
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def repeated_speech(
    answers: Dict[float, ChunkTranscript],
) -> Optional[Tuple[ChunkTranscript, float, float]]:
    """The temperature-0 answer of a chunk whose temperature-0 answer looped,
    when the audio itself repeats.

    ``answers`` maps temperatures asked to their json answers; the one at 0.0
    is the loop. Returns it with the temperature and agreement of the answer
    that corroborates it: the one at ``LOOP_BREAKING_TEMPERATURE`` or above
    that agrees with it most, when that is at least
    ``REPEATED_SPEECH_AGREEMENT``. ``None`` otherwise.
    """
    first = answers.get(FALLBACK_TEMPERATURES[0])
    if first is None:
        return None
    agreement, temperature = max(
        (
            (word_agreement(first.text, answer.text), temperature)
            for temperature, answer in answers.items()
            if temperature >= LOOP_BREAKING_TEMPERATURE
        ),
        default=(0.0, 0.0),
    )
    if agreement < REPEATED_SPEECH_AGREEMENT:
        return None
    return first, temperature, agreement


def _settled(
    first: ChunkTranscript,
    candidates: Dict[float, ChunkTranscript],
    usable_at: List[float],
    chunk: AudioChunk,
    source: str,
    logger: logging.Logger,
) -> Optional[ChunkTranscript]:
    """The json answer a chunk keeps once its answer at 0.0 (``first``) looped,
    or ``None`` while that is not yet decided.

    ``candidates`` are the later json answers by temperature, ``usable_at``
    the temperatures whose answer neither looped nor came back empty. Nothing
    is decided before an answer at ``LOOP_BREAKING_TEMPERATURE`` or above is
    in. One that agrees with ``first`` keeps ``first`` as repeated speech
    (``repeated_speech``). Once every temperature is asked, the
    lowest-temperature candidate that an answer at another loop-breaking
    temperature corroborates (``word_agreement`` at least
    ``REPEATED_SPEECH_AGREEMENT``) is kept — one asked at a loop-breaking
    temperature only when usable, as a loop there formed under sampling.
    Failing that, the answer a majority of the answers agree with
    (``_majority_answer``), else the lowest-temperature usable one when it
    holds no speech words, only sound annotations (``speech_words``). With
    none of these, ``None``: the chunk has no faithful answer.
    """
    breaking = {
        temperature: answer
        for temperature, answer in candidates.items()
        if temperature >= LOOP_BREAKING_TEMPERATURE
    }
    if not breaking:
        return None
    repeated = repeated_speech({FALLBACK_TEMPERATURES[0]: first, **breaking})
    if repeated is not None:
        kept, corroborating, agreement = repeated
        logger.warning(
            "%s: chunk %d (%.2f-%.2fs) looped at temperature 0.0 and its answer "
            "at temperature %.1f agrees with it (%.2f): kept as repeated speech",
            source,
            chunk.index,
            chunk.start_s,
            chunk.end_s,
            corroborating,
            agreement,
        )
        return kept
    if len(candidates) < TRANSCRIBE_ATTEMPTS - 1:
        return None
    for temperature, answer in sorted(candidates.items()):
        if temperature >= LOOP_BREAKING_TEMPERATURE and temperature not in usable_at:
            continue
        corroborating = [
            other
            for other_temperature, other in breaking.items()
            if other_temperature != temperature
            and word_agreement(answer.text, other.text) >= REPEATED_SPEECH_AGREEMENT
        ]
        if corroborating:
            logger.warning(
                "%s: chunk %d (%.2f-%.2fs) looped at temperature 0.0 and no "
                "answer repeats it: kept the answer at temperature %.1f, which "
                "an answer at a loop-breaking temperature agrees with",
                source,
                chunk.index,
                chunk.start_s,
                chunk.end_s,
                temperature,
            )
            return answer
    consensus = _majority_answer(first, candidates, usable_at)
    if consensus is not None:
        temperature, answer, support = consensus
        logger.warning(
            "%s: chunk %d (%.2f-%.2fs) looped at temperature 0.0 and no answer at "
            "a loop-breaking temperature corroborates another: kept the answer at "
            "temperature %.1f, which %d of the other answers agree with",
            source,
            chunk.index,
            chunk.start_s,
            chunk.end_s,
            temperature,
            support,
        )
        return answer
    if usable_at and not speech_words(candidates[min(usable_at)].text):
        return candidates[min(usable_at)]
    return None


# Answers, besides the one kept, that must agree with it for a chunk whose
# loop-breaking answers corroborate nothing: with it, half of the six. On the
# cluster's server repeated speech held at 0.0-0.4 while the answers from 0.6
# up broke; a decode loop reproduced at low temperatures was corroborated
# elsewhere in every recorded chunk.
MAJORITY_SUPPORT = TRANSCRIBE_ATTEMPTS // 2 - 1

_ANNOTATION = re.compile(r"\*[^*]*\*|\[[^\]]*\]|\([^)]*\)|[♪♫]")


def speech_words(text: str) -> List[str]:
    """The words of an answer outside sound annotations (``*Ding*``,
    ``[Music]``, ``(laughs)``, ``♪``)."""
    return [
        word
        for word in map(_comparable, _units(_ANNOTATION.sub(" ", text), False))
        if word
    ]


def _majority_answer(
    first: ChunkTranscript,
    candidates: Dict[float, ChunkTranscript],
    usable_at: List[float],
) -> Optional[Tuple[float, ChunkTranscript, int]]:
    """The lowest-temperature answer at least ``MAJORITY_SUPPORT`` others agree
    with (half the answers with it), with its temperature and how many do, among the answer at 0.0 and
    the later ones; an answer at ``LOOP_BREAKING_TEMPERATURE`` or above takes
    part only when it did not loop."""
    answers = {
        FALLBACK_TEMPERATURES[0]: first,
        **{
            temperature: answer
            for temperature, answer in candidates.items()
            if temperature < LOOP_BREAKING_TEMPERATURE or temperature in usable_at
        },
    }
    for temperature, answer in sorted(answers.items()):
        support = sum(
            1
            for other_temperature, other in answers.items()
            if other_temperature != temperature
            and word_agreement(answer.text, other.text) >= REPEATED_SPEECH_AGREEMENT
        )
        if support >= MAJORITY_SUPPORT:
            return temperature, answer, support
    return None


def _transcribe_checked(
    chunk: AudioChunk,
    transcribe_chunk: ChunkRequest,
    language: Optional[str],
    source: str,
    logger: logging.Logger,
) -> ChunkTranscript:
    loudest = chunk.loudest_frame_dbfs
    named = language
    timed: Optional[ChunkTranscript] = None
    untimed: Optional[ChunkTranscript] = None
    loops: List[float] = []
    looped: Dict[float, ChunkTranscript] = {}

    def usable(answer: ChunkTranscript, timestamps: bool, attempt: int) -> bool:
        ratio = compression_ratio(answer.text)
        if ratio > GARBLED_COMPRESSION_RATIO:
            logger.warning(
                "%s: chunk %d (%.2f-%.2fs) came back as a repetition loop in %s "
                "(compression ratio %.2f > %s) on attempt %d of %d at temperature %.1f",
                source,
                chunk.index,
                chunk.start_s,
                chunk.end_s,
                response_format(timestamps),
                ratio,
                GARBLED_COMPRESSION_RATIO,
                attempt,
                TRANSCRIBE_ATTEMPTS,
                FALLBACK_TEMPERATURES[attempt - 1],
            )
            if not timestamps:
                loops.append(ratio)
                looped[FALLBACK_TEMPERATURES[attempt - 1]] = answer
            return False
        if loudest < SILENCE_FLOOR_DBFS:
            return True
        if answer.text.strip() and (answer.segments or not timestamps):
            return True
        logger.warning(
            "%s: chunk %d (%.2f-%.2fs, loudest frame %.1f dBFS) came back with an "
            "empty %s transcript on attempt %d of %d at temperature %.1f",
            source,
            chunk.index,
            chunk.start_s,
            chunk.end_s,
            loudest,
            response_format(timestamps),
            attempt,
            TRANSCRIBE_ATTEMPTS,
            FALLBACK_TEMPERATURES[attempt - 1],
        )
        return False

    duration = len(chunk.samples) / WHISPER_SAMPLE_RATE
    complete = False
    timed_temperature = 0.0
    # Once the json answer at 0.0 looped, every later json answer by
    # temperature: none is kept before the loop-breaking ones are compared.
    candidates: Dict[float, ChunkTranscript] = {}
    usable_at: List[float] = []
    judged = False
    # The judged timed answer when its mean log-probability is below
    # NON_SPEECH_AVG_LOGPROB, with its temperature; once a json answer
    # carrying other speech refutes it, no timed answer is asked or used.
    improbable: Optional[Tuple[ChunkTranscript, float]] = None
    refuted = False
    for attempt, temperature in enumerate(FALLBACK_TEMPERATURES, start=1):
        asked: Optional[ChunkTranscript] = None
        if timed is None and not refuted:
            answer = transcribe_chunk(chunk, named, True, temperature)
            named = answer.language or named
            if (
                not judged
                and temperature <= NON_SPEECH_JUDGED_TEMPERATURE
                and answer.avg_logprob is not None
            ):
                judged = True
                if answer.avg_logprob < NON_SPEECH_AVG_LOGPROB:
                    improbable = (answer, temperature)
            if improbable is None and usable(answer, True, attempt):
                timed, timed_temperature = answer, temperature
                complete = reaches_chunk_end(timed.segments, duration)
        first = looped.get(FALLBACK_TEMPERATURES[0])
        if untimed is None and complete and first is None:
            break
        if untimed is None:
            answer = asked = transcribe_chunk(chunk, named, False, temperature)
            fit = usable(answer, False, attempt)
            if first is None:
                if fit:
                    untimed = answer
            elif temperature > FALLBACK_TEMPERATURES[0]:
                candidates[temperature] = answer
                if fit:
                    usable_at.append(temperature)
                untimed = _settled(first, candidates, usable_at, chunk, source, logger)
        if improbable is not None and not refuted:
            claimed, claimed_at = improbable
            said = asked if asked is not None else untimed
            agreement = word_agreement(said.text, claimed.text)
            if not speech_words(said.text) or agreement >= REPEATED_SPEECH_AGREEMENT:
                logger.warning(
                    "%s: chunk %d (%.2f-%.2fs) timed answer %r at temperature "
                    "%.1f has mean log-probability %.2f < %s and the json answer "
                    "carries no other speech: not speech, kept no text",
                    source,
                    chunk.index,
                    chunk.start_s,
                    chunk.end_s,
                    claimed.text,
                    claimed_at,
                    claimed.avg_logprob,
                    NON_SPEECH_AVG_LOGPROB,
                )
                return ChunkTranscript(text="", language=None, segments=[])
            logger.warning(
                "%s: chunk %d (%.2f-%.2fs) timed answer %r at temperature %.1f "
                "has mean log-probability %.2f < %s but the json answer carries "
                "other speech (%.2f): transcribed from json answers alone",
                source,
                chunk.index,
                chunk.start_s,
                chunk.end_s,
                claimed.text,
                claimed_at,
                claimed.avg_logprob,
                NON_SPEECH_AVG_LOGPROB,
                agreement,
            )
            refuted = True
        if untimed is not None and (timed is not None or refuted):
            break

    if (
        untimed is not None
        and timed is not None
        and (
            timed_temperature >= LOOP_BREAKING_TEMPERATURE
            or FALLBACK_TEMPERATURES[0] in looped
        )
        and word_agreement(timed.text, untimed.text) < REPEATED_SPEECH_AGREEMENT
    ):
        logger.warning(
            "%s: chunk %d (%.2f-%.2fs) timed answer at temperature %.1f does not "
            "agree with the transcript (%.2f): not used for timing",
            source,
            chunk.index,
            chunk.start_s,
            chunk.end_s,
            timed_temperature,
            word_agreement(timed.text, untimed.text),
        )
        timed = None
    if (
        untimed is None
        and timed is not None
        and timed_temperature >= LOOP_BREAKING_TEMPERATURE
        and speech_words(timed.text)
    ):
        logger.warning(
            "%s: chunk %d (%.2f-%.2fs) has no faithful json answer and its timed "
            "answer at temperature %.1f has nothing to agree with: not used",
            source,
            chunk.index,
            chunk.start_s,
            chunk.end_s,
            timed_temperature,
        )
        timed, complete = None, False
    if untimed is None and not complete:
        if loops:
            raise GarbledTranscriptError(
                source,
                chunk.index,
                chunk.start_s,
                chunk.end_s,
                tuple(loops),
                TRANSCRIBE_ATTEMPTS,
            )
        raise EmptyTranscriptError(
            source,
            chunk.index,
            chunk.start_s,
            chunk.end_s,
            loudest,
            TRANSCRIBE_ATTEMPTS,
        )
    text, segments = align_text(
        untimed.text if untimed is not None else timed.text,
        timed.segments if timed is not None else [],
        duration,
        no_space=(named or "").lower() in NO_SPACE_LANGUAGES,
    )
    return ChunkTranscript(text=text, language=named, segments=segments)
