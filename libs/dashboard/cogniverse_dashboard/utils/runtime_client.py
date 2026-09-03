"""Shared HTTP client for dashboard-to-runtime calls."""

import httpx
import streamlit as st


@st.cache_resource
def get_runtime_client() -> httpx.Client:
    """One pooled client per dashboard process. A fresh client per action
    pays pool construction and teardown on every interaction."""
    return httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
