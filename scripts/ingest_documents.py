#!/usr/bin/env python3
"""
Document Ingestion Script - Dual Strategy

Ingests documents using BOTH strategies for comparison:
1. Visual Strategy (ColPali): Treats pages as images
2. Text Strategy: Text extraction + semantic embeddings

Creates two separate Vespa entries per document for evaluation.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
import torch
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.common.models.model_loaders import get_or_load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_pdf_to_images(pdf_path: Path) -> List[Image.Image]:
    """
    Convert PDF pages to images

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of PIL Images, one per page
    """
    try:
        import pdf2image

        logger.info(f"Converting PDF to images: {pdf_path.name}")
        images = pdf2image.convert_from_path(str(pdf_path))
        logger.info(f"✅ Converted {len(images)} pages")
        return images
    except Exception as e:
        logger.error(f"Failed to convert PDF: {e}")
        logger.info("Install pdf2image: pip install pdf2image")
        logger.info("And poppler: brew install poppler (macOS) or apt-get install poppler-utils (Linux)")
        return []


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from PDF

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text
    """
    try:
        import PyPDF2

        logger.info(f"Extracting text from PDF: {pdf_path.name}")
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

        logger.info(f"✅ Extracted {len(text)} characters")
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text: {e}")
        logger.info("Install PyPDF2: pip install PyPDF2")
        return ""


def encode_page_with_colpali(page_image: Image.Image, model, processor) -> np.ndarray:
    """
    Encode document page using ColPali (same as video frames)

    Args:
        page_image: PIL Image of document page
        model: ColPali model
        processor: ColPali processor

    Returns:
        ColPali multi-vector embedding [1024, 128]
    """
    # Process with ColPali (reusing existing pattern)
    batch_inputs = processor.process_images([page_image]).to(model.device)

    # Get embeddings
    with torch.no_grad():
        embeddings = model(**batch_inputs)

    # Reshape to [1024, 128] format (remove batch dimension)
    embeddings_np = embeddings.squeeze(0).cpu().numpy()

    # Pad or truncate to exactly 1024 patches
    if embeddings_np.shape[0] < 1024:
        padding = np.zeros((1024 - embeddings_np.shape[0], embeddings_np.shape[1]))
        embeddings_np = np.vstack([embeddings_np, padding])
    elif embeddings_np.shape[0] > 1024:
        embeddings_np = embeddings_np[:1024]

    return embeddings_np


def ingest_document_visual_strategy(
    document_path: Path,
    colpali_model,
    colpali_processor,
    vespa_endpoint: str,
    app_name: str = "documentsearch"
) -> int:
    """
    Ingest document using visual strategy (ColPali page-as-image)

    Args:
        document_path: Path to PDF document
        colpali_model: ColPali model
        colpali_processor: ColPali processor
        vespa_endpoint: Vespa endpoint
        app_name: Vespa application name

    Returns:
        Number of pages ingested
    """
    logger.info(f"\n{'='*60}")
    logger.info("Visual Strategy (ColPali Page-as-Image)")
    logger.info(f"{'='*60}")

    # Convert PDF to images
    page_images = convert_pdf_to_images(document_path)
    if not page_images:
        return 0

    document_id = document_path.stem
    successful = 0

    for page_num, page_image in enumerate(page_images, 1):
        try:
            logger.info(f"Processing page {page_num}/{len(page_images)}...")

            # Generate ColPali embedding
            colpali_embedding = encode_page_with_colpali(page_image, colpali_model, colpali_processor)

            # Create Vespa document
            doc = {
                "fields": {
                    "document_id": f"{document_id}_p{page_num}",
                    "document_title": document_path.name,
                    "document_type": "pdf",
                    "page_number": page_num,
                    "page_count": len(page_images),
                    "source_url": f"file://{document_path.absolute()}",
                    "creation_timestamp": int(time.time()),
                    "colpali_embedding": colpali_embedding.tolist(),
                }
            }

            # Upload to Vespa
            doc_url = f"{vespa_endpoint}/document/v1/{app_name}/document_visual/docid/{document_id}_p{page_num}"

            response = requests.post(
                doc_url,
                json=doc,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                logger.info(f"✅ Ingested page {page_num} (visual)")
                successful += 1
            else:
                logger.error(f"❌ Failed to ingest page {page_num}: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"❌ Error processing page {page_num}: {e}")

    return successful


def ingest_document_text_strategy(
    document_path: Path,
    text_embedding_model,
    vespa_endpoint: str,
    app_name: str = "documentsearch"
) -> bool:
    """
    Ingest document using text strategy (extraction + semantic embeddings)

    Args:
        document_path: Path to PDF document
        text_embedding_model: Sentence transformer model
        vespa_endpoint: Vespa endpoint
        app_name: Vespa application name

    Returns:
        True if successful
    """
    logger.info(f"\n{'='*60}")
    logger.info("Text Strategy (Extraction + Semantic)")
    logger.info(f"{'='*60}")

    # Extract text
    full_text = extract_text_from_pdf(document_path)
    if not full_text:
        logger.warning("No text extracted, skipping text strategy")
        return False

    # Get page count
    try:
        import PyPDF2
        with open(document_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_count = len(pdf_reader.pages)
    except Exception:
        page_count = 0

    # Generate semantic embedding
    logger.info("Generating semantic embedding...")
    document_embedding = text_embedding_model.encode(
        full_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Create Vespa document
    document_id = document_path.stem
    doc = {
        "fields": {
            "document_id": document_id,
            "document_title": document_path.name,
            "document_type": "pdf",
            "page_count": page_count,
            "source_url": f"file://{document_path.absolute()}",
            "creation_timestamp": int(time.time()),
            "full_text": full_text[:10000],  # Truncate for Vespa limits
            "section_headings": [],
            "key_entities": [],
            "document_embedding": document_embedding.tolist(),
        }
    }

    # Upload to Vespa
    doc_url = f"{vespa_endpoint}/document/v1/{app_name}/document_text/docid/{document_id}"

    response = requests.post(
        doc_url,
        json=doc,
        headers={"Content-Type": "application/json"},
        timeout=30
    )

    if response.status_code == 200:
        logger.info("✅ Ingested document (text)")
        return True
    else:
        logger.error(f"❌ Failed to ingest: {response.status_code} - {response.text}")
        return False


def ingest_document_dual_strategy(
    document_path: Path,
    colpali_model,
    colpali_processor,
    text_embedding_model,
    vespa_endpoint: str,
    app_name: str = "documentsearch"
) -> Dict[str, int]:
    """
    Ingest document using BOTH strategies

    Args:
        document_path: Path to PDF document
        colpali_model: ColPali model
        colpali_processor: ColPali processor
        text_embedding_model: Sentence transformer model
        vespa_endpoint: Vespa endpoint
        app_name: Vespa application name

    Returns:
        Dict with counts for each strategy
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"Ingesting: {document_path.name}")
    logger.info(f"{'#'*80}")

    results = {
        "visual_pages": 0,
        "text_success": False,
    }

    # Strategy 1: Visual (ColPali page-as-image)
    visual_pages = ingest_document_visual_strategy(
        document_path,
        colpali_model,
        colpali_processor,
        vespa_endpoint,
        app_name
    )
    results["visual_pages"] = visual_pages

    # Strategy 2: Text (extraction + semantic)
    text_success = ingest_document_text_strategy(
        document_path,
        text_embedding_model,
        vespa_endpoint,
        app_name
    )
    results["text_success"] = text_success

    return results


def ingest_documents_from_directory(
    document_dir: Path,
    vespa_endpoint: str = "http://localhost:8080",
    colpali_model: str = "vidore/colsmol-500m",
    app_name: str = "documentsearch",
):
    """
    Ingest all PDF documents from a directory using both strategies

    Args:
        document_dir: Directory containing PDF files
        vespa_endpoint: Vespa endpoint
        colpali_model: ColPali model name
        app_name: Vespa application name
    """
    logger.info(f"Starting document ingestion from {document_dir}")
    logger.info(f"Vespa endpoint: {vespa_endpoint}")
    logger.info(f"ColPali model: {colpali_model}")

    # Load ColPali model
    logger.info("\nLoading ColPali model...")
    config = {"colpali_model": colpali_model}
    colpali_model_obj, colpali_processor = get_or_load_model(colpali_model, config, logger)
    logger.info("✅ ColPali model loaded")

    # Load text embedding model
    logger.info("\nLoading text embedding model...")
    from sentence_transformers import SentenceTransformer
    text_embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    logger.info("✅ Text embedding model loaded")

    # Find all PDF files
    pdf_files = list(document_dir.glob("*.pdf")) + list(document_dir.glob("*.PDF"))
    logger.info(f"\nFound {len(pdf_files)} PDF files")

    # Ingest documents
    total_visual_pages = 0
    total_text_docs = 0

    for pdf_path in pdf_files:
        results = ingest_document_dual_strategy(
            pdf_path,
            colpali_model_obj,
            colpali_processor,
            text_embedding_model,
            vespa_endpoint,
            app_name
        )
        total_visual_pages += results["visual_pages"]
        if results["text_success"]:
            total_text_docs += 1

    logger.info(f"\n{'='*80}")
    logger.info("Ingestion Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Visual Strategy: {total_visual_pages} pages ingested")
    logger.info(f"Text Strategy: {total_text_docs} documents ingested")
    logger.info(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents with dual strategy")
    parser.add_argument("--document_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--vespa_endpoint", type=str, default="http://localhost:8080", help="Vespa endpoint")
    parser.add_argument("--colpali_model", type=str, default="vidore/colsmol-500m", help="ColPali model name")
    parser.add_argument("--app_name", type=str, default="documentsearch", help="Vespa application name")

    args = parser.parse_args()

    document_dir = Path(args.document_dir)
    if not document_dir.exists():
        logger.error(f"Document directory not found: {document_dir}")
        sys.exit(1)

    ingest_documents_from_directory(
        document_dir=document_dir,
        vespa_endpoint=args.vespa_endpoint,
        colpali_model=args.colpali_model,
        app_name=args.app_name
    )


if __name__ == "__main__":
    main()
