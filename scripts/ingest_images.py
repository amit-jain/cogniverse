#!/usr/bin/env python3
"""
Image Ingestion Script

Ingests images with ColPali embeddings into Vespa image_content schema.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import requests
import torch
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.common.models.model_loaders import get_or_load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_image_with_colpali(image_path: Path, model, processor) -> np.ndarray:
    """
    Encode image using ColPali

    Args:
        image_path: Path to image file
        model: ColPali model
        processor: ColPali processor

    Returns:
        ColPali multi-vector embedding [1024, 128]
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Process with ColPali (reusing existing pattern from embedding_generator)
    batch_inputs = processor.process_images([image]).to(model.device)

    # Get embeddings
    with torch.no_grad():
        embeddings = model(**batch_inputs)  # Returns tensor directly

    # Reshape to [1024, 128] format (remove batch dimension)
    embeddings_np = embeddings.squeeze(0).cpu().numpy()

    # Pad or truncate to exactly 1024 patches
    if embeddings_np.shape[0] < 1024:
        padding = np.zeros((1024 - embeddings_np.shape[0], embeddings_np.shape[1]))
        embeddings_np = np.vstack([embeddings_np, padding])
    elif embeddings_np.shape[0] > 1024:
        embeddings_np = embeddings_np[:1024]

    return embeddings_np


def ingest_image(
    image_path: Path,
    model,
    processor,
    vespa_endpoint: str,
    app_name: str = "imagesearch"
) -> bool:
    """
    Ingest single image into Vespa

    Args:
        image_path: Path to image file
        model: ColPali model
        processor: ColPali processor
        vespa_endpoint: Vespa endpoint
        app_name: Vespa application name

    Returns:
        True if successful
    """
    try:
        logger.info(f"Processing {image_path.name}...")

        # Generate ColPali embedding
        embedding = encode_image_with_colpali(image_path, model, processor)

        # Create Vespa document
        image_id = image_path.stem
        doc = {
            "fields": {
                "image_id": image_id,
                "image_title": image_path.name,
                "source_url": f"file://{image_path.absolute()}",
                "creation_timestamp": int(time.time()),
                "image_description": f"Image: {image_path.name}",
                "detected_objects": [],
                "detected_scenes": [],
                "colpali_embedding": embedding.tolist(),
            }
        }

        # Upload to Vespa
        doc_url = f"{vespa_endpoint}/document/v1/{app_name}/image_content/docid/{image_id}"

        response = requests.post(
            doc_url,
            json=doc,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            logger.info(f"✅ Ingested {image_path.name}")
            return True
        else:
            logger.error(f"❌ Failed to ingest {image_path.name}: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Error ingesting {image_path.name}: {e}")
        return False


def ingest_images_from_directory(
    image_dir: Path,
    vespa_endpoint: str = "http://localhost:8080",
    colpali_model: str = "vidore/colsmol-500m",
    app_name: str = "imagesearch",
    extensions: List[str] = None
):
    """
    Ingest all images from a directory

    Args:
        image_dir: Directory containing images
        vespa_endpoint: Vespa endpoint
        colpali_model: ColPali model name
        app_name: Vespa application name
        extensions: Image file extensions to process
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    logger.info(f"Starting image ingestion from {image_dir}")
    logger.info(f"Vespa endpoint: {vespa_endpoint}")
    logger.info(f"ColPali model: {colpali_model}")

    # Load ColPali model
    logger.info("Loading ColPali model...")
    config = {"colpali_model": colpali_model}
    model, processor = get_or_load_model(colpali_model, config, logger)
    logger.info("✅ ColPali model loaded")

    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))

    logger.info(f"Found {len(image_files)} images")

    # Ingest images
    successful = 0
    failed = 0

    for image_path in image_files:
        if ingest_image(image_path, model, processor, vespa_endpoint, app_name):
            successful += 1
        else:
            failed += 1

    logger.info(f"\n{'='*60}")
    logger.info("Ingestion complete!")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Ingest images with ColPali embeddings")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--vespa_endpoint", type=str, default="http://localhost:8080", help="Vespa endpoint")
    parser.add_argument("--colpali_model", type=str, default="vidore/colsmol-500m", help="ColPali model name")
    parser.add_argument("--app_name", type=str, default="imagesearch", help="Vespa application name")

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        sys.exit(1)

    ingest_images_from_directory(
        image_dir=image_dir,
        vespa_endpoint=args.vespa_endpoint,
        colpali_model=args.colpali_model,
        app_name=args.app_name
    )


if __name__ == "__main__":
    main()
