#!/usr/bin/env python3
"""
Audio Ingestion Script

Ingests audio files with Whisper transcription into Vespa audio_content schema.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import requests

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.app.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)
from src.app.ingestion.processors.audio_transcriber import AudioTranscriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_audio(
    audio_path: Path,
    transcriber: AudioTranscriber,
    embedding_generator: AudioEmbeddingGenerator,
    vespa_endpoint: str,
    app_name: str = "audiosearch"
) -> bool:
    """
    Ingest single audio file into Vespa

    Args:
        audio_path: Path to audio file
        transcriber: AudioTranscriber instance
        embedding_generator: AudioEmbeddingGenerator instance
        vespa_endpoint: Vespa endpoint
        app_name: Vespa application name

    Returns:
        True if successful
    """
    try:
        logger.info(f"Processing {audio_path.name}...")

        # Transcribe audio
        result = transcriber.transcribe_audio(
            video_path=audio_path,
            output_dir=None
        )

        transcript = result.get("full_text", "")

        # Generate embeddings
        logger.info(f"Generating embeddings for {audio_path.name}...")
        acoustic_embedding, semantic_embedding = embedding_generator.generate_embeddings(
            audio_path=audio_path,
            transcript=transcript
        )

        # Create Vespa document
        audio_id = audio_path.stem
        doc = {
            "fields": {
                "audio_id": audio_id,
                "audio_title": audio_path.name,
                "source_url": f"file://{audio_path.absolute()}",
                "duration": result.get("duration", 0.0),
                "transcript": transcript,
                "speaker_labels": [],
                "detected_events": [],
                "language": result.get("language", "unknown"),
                "audio_embedding": acoustic_embedding.tolist(),
                "semantic_embedding": semantic_embedding.tolist(),
            }
        }

        # Upload to Vespa
        doc_url = f"{vespa_endpoint}/document/v1/{app_name}/audio_content/docid/{audio_id}"

        response = requests.post(
            doc_url,
            json=doc,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            logger.info(f"✅ Ingested {audio_path.name} (language: {result.get('language', 'unknown')})")
            return True
        else:
            logger.error(f"❌ Failed to ingest {audio_path.name}: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Error ingesting {audio_path.name}: {e}")
        return False


def ingest_audio_from_directory(
    audio_dir: Path,
    vespa_endpoint: str = "http://localhost:8080",
    whisper_model: str = "base",
    app_name: str = "audiosearch",
    extensions: List[str] = None
):
    """
    Ingest all audio files from a directory

    Args:
        audio_dir: Directory containing audio files
        vespa_endpoint: Vespa endpoint
        whisper_model: Whisper model size
        app_name: Vespa application name
        extensions: Audio file extensions to process
    """
    if extensions is None:
        extensions = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]

    logger.info(f"Starting audio ingestion from {audio_dir}")
    logger.info(f"Vespa endpoint: {vespa_endpoint}")
    logger.info(f"Whisper model: {whisper_model}")

    # Initialize AudioTranscriber
    logger.info("Loading Whisper model...")
    transcriber = AudioTranscriber(model_size=whisper_model)
    logger.info("✅ Whisper model loaded")

    # Initialize AudioEmbeddingGenerator
    logger.info("Loading embedding models...")
    embedding_generator = AudioEmbeddingGenerator()
    logger.info("✅ Embedding models loaded")

    # Find all audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(list(audio_dir.glob(f"*{ext}")))
        audio_files.extend(list(audio_dir.glob(f"*{ext.upper()}")))

    logger.info(f"Found {len(audio_files)} audio files")

    # Ingest audio files
    successful = 0
    failed = 0

    for audio_path in audio_files:
        if ingest_audio(audio_path, transcriber, embedding_generator, vespa_endpoint, app_name):
            successful += 1
        else:
            failed += 1

    logger.info(f"\n{'='*60}")
    logger.info("Ingestion complete!")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Ingest audio with Whisper transcription")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--vespa_endpoint", type=str, default="http://localhost:8080", help="Vespa endpoint")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--app_name", type=str, default="audiosearch", help="Vespa application name")

    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        sys.exit(1)

    ingest_audio_from_directory(
        audio_dir=audio_dir,
        vespa_endpoint=args.vespa_endpoint,
        whisper_model=args.whisper_model,
        app_name=args.app_name
    )


if __name__ == "__main__":
    main()
