"""Embedding Atlas tab — UMAP + Apple's embedding-atlas component.

Lazy-imports ``umap``, ``embedding_atlas`` and (indirectly) ``sklearn`` so
the main dashboard image doesn't need to load them at startup. The tab
renders an install-instructions message if any optional dep is missing.

Reads parquet embeddings from ``outputs/embeddings/`` (the path an
ingestion run writes to via ``scripts/export_backend_embeddings.py``);
falls back to a file-upload widget if no local files exist.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

_EMBEDDINGS_DIR = Path("outputs/embeddings")
_OPTIONAL_DEPS = ("umap", "embedding_atlas.streamlit")


def _load_optional_deps() -> Tuple[bool, Optional[str]]:
    """Return (ok, missing_module). Lazy import keeps cold-start cheap."""
    for mod in _OPTIONAL_DEPS:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            return False, f"{mod} ({exc})"
    return True, None


def _render_missing_deps_hint(missing: str) -> None:
    st.warning(f"Embedding Atlas needs extra libraries: `{missing}`")
    st.code(
        "uv pip install umap-learn embedding-atlas",
        language="bash",
    )
    st.caption(
        "These libs pull in numpy/scipy/sklearn/pyarrow and are not "
        "installed in the default dashboard image."
    )


def _pick_embedding_file() -> Optional[Path]:
    """Return the parquet file to visualise. Prefers anything the caller
    stashed in ``st.session_state['embedding_atlas_file']`` (e.g. the
    Interactive Search tab's 'Visualise this search' action); otherwise
    lists whatever is in ``outputs/embeddings/`` as a selectbox."""
    session_path = st.session_state.get("embedding_atlas_file")
    if session_path:
        path = Path(session_path)
        if path.exists():
            return path
        st.warning(f"Session path {session_path} no longer exists; pick another.")

    if not _EMBEDDINGS_DIR.exists():
        st.info(
            f"No embeddings directory at `{_EMBEDDINGS_DIR}/`. Run an export "
            "via `scripts/export_backend_embeddings.py` first."
        )
        return None

    parquets = sorted(_EMBEDDINGS_DIR.glob("*.parquet"))
    if not parquets:
        st.info(
            f"No parquet files in `{_EMBEDDINGS_DIR}/`. Export embeddings "
            "with `scripts/export_backend_embeddings.py --tenant <id>`."
        )
        return None

    return st.selectbox(
        "Choose embedding file",
        parquets,
        format_func=lambda p: p.name,
        index=len(parquets) - 1,  # default to most recent
    )


def _ensure_2d_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Compute UMAP x/y columns if not already present. Exporters should
    pre-compute x/y to keep the tab responsive; we do it here as a fallback."""
    if "x" in df.columns and "y" in df.columns:
        return df
    if "embedding" not in df.columns or df.empty:
        return df
    import numpy as np
    from umap import UMAP

    with st.spinner("Computing 2D projection (UMAP)..."):
        embeddings = np.array(df["embedding"].tolist())
        reducer = UMAP(
            n_components=2,
            n_neighbors=min(15, max(len(df) - 1, 2)),
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)
        df = df.copy()
        df["x"] = coords[:, 0]
        df["y"] = coords[:, 1]
    return df


def _label_points(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Pick a label column for colouring. If the DataFrame has an
    ``is_query`` flag, mark query rows so they stand out from documents;
    otherwise fall back to embedding-atlas's automatic clustering."""
    if "is_query" not in df.columns or not df["is_query"].any():
        return df, "automatic"
    df = df.copy()
    df["point_type"] = df["is_query"].apply(lambda q: "Query" if q else "Document")
    query_rows = df.index[df["is_query"]].tolist()
    for i, idx in enumerate(query_rows):
        df.loc[idx, "point_type"] = f"Query {i + 1}"
    return df, "point_type"


def _render_query_similarity(df: pd.DataFrame) -> None:
    """If the parquet includes ``query_similarity_<idx>`` columns, show
    the top-3 most similar documents per query underneath the atlas."""
    if "is_query" not in df.columns:
        return
    query_df = df[df["is_query"]]
    if query_df.empty:
        return
    doc_df = df[~df["is_query"]]

    st.markdown("---")
    st.subheader("🎯 Query Analysis")
    for i, (idx, query_row) in enumerate(query_df.iterrows()):
        query_text = query_row.get("text", "").replace("QUERY: ", "")
        st.write(f"**Query {i + 1}:** {query_text}")
        sim_col = f"query_similarity_{idx}"
        if sim_col not in doc_df.columns:
            continue
        top_k = min(3, len(doc_df))
        top_docs = doc_df.nlargest(top_k, sim_col)
        with st.expander(f"Top {top_k} similar documents"):
            for _, doc in top_docs.iterrows():
                title = doc.get("video_title")
                if pd.notna(title):
                    display = str(title)
                    frame = doc.get("frame_number")
                    if pd.notna(frame):
                        display += f" (Frame {int(frame)})"
                elif pd.notna(doc.get("text")):
                    display = str(doc["text"])[:100] + "..."
                else:
                    display = f"Document {doc.name}"
                st.write(f"• {display} (Similarity: {doc[sim_col]:.3f})")


def render_embedding_atlas_tab() -> None:
    """Top-level tab renderer invoked from ``cogniverse_dashboard.app``."""
    st.header("🗺️ Embedding Atlas")
    st.caption(
        "UMAP projection + Apple's embedding-atlas component. Visualises "
        "embeddings exported via `scripts/export_backend_embeddings.py`."
    )

    ok, missing = _load_optional_deps()
    if not ok:
        _render_missing_deps_hint(missing or "unknown")
        return

    file_path = _pick_embedding_file()
    if file_path is None:
        return

    try:
        df = pd.read_parquet(file_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read {file_path.name}: {exc}")
        return

    if df.empty:
        st.warning(f"{file_path.name} contains no rows.")
        return

    df = _ensure_2d_coords(df)
    if "x" not in df.columns or "y" not in df.columns:
        st.error(
            f"{file_path.name} lacks `x`/`y` columns and no `embedding` "
            "column to derive them from. Re-export with UMAP enabled."
        )
        return

    df, labels_column = _label_points(df)

    size_mb = file_path.stat().st_size / (1024 * 1024)
    n_queries = int(df["is_query"].sum()) if "is_query" in df.columns else 0
    query_suffix = f" • {n_queries} query rows" if n_queries else ""
    st.info(
        f"**{len(df):,} points** from `{file_path.name}` "
        f"({size_mb:.2f} MB){query_suffix}"
    )

    from embedding_atlas.streamlit import embedding_atlas

    result = embedding_atlas(
        data_frame=df,
        x="x",
        y="y",
        text="text" if "text" in df.columns else None,
        labels=labels_column,
        show_table=True,
        show_charts=True,
        show_embedding=True,
    )

    _render_query_similarity(df)

    if result and result.get("predicate"):
        st.info(f"Current selection: {result['predicate']}")
