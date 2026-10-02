"""Shared HTTP client for dashboard-to-runtime calls."""

import httpx
import streamlit as st


@st.cache_resource
def get_runtime_client() -> httpx.Client:
    """One pooled client per dashboard process. A fresh client per action
    pays pool construction and teardown on every interaction."""
    return httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))


def runtime_error_message(response) -> str:
    """What to show a reader for a runtime answer that is not a success.

    A runtime failure answers ``{"detail": {"error", "message", ...}}``, whose
    ``message`` is the sentence written for a reader; a ``detail`` that is a
    plain string (a 4xx naming the bad input) is shown as it is. Any other
    body is shown as its text. Takes an ``httpx`` or ``requests`` response.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        return detail["message"]
    if isinstance(detail, str):
        return detail
    return response.text
