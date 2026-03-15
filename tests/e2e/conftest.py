"""Shared fixtures and helpers for E2E tests.

Provides availability checks, skip markers, and Streamlit interaction helpers
for both API (httpx) and dashboard (Playwright) E2E tests.

Test artifact paths (real data used for ingestion tests):
- Video: data/testset/evaluation/sample_videos/v_-nl4G-00PtA.mp4
- Image: data/testset/evaluation/processed/keyframes/big_buck_bunny_clip/frame_0000.jpg
- Audio: extracted via ffmpeg from sample video (real speech/sound)
- PDF: Video-ChatGPT paper from arxiv (related to the test dataset)
- Document: data/testset/dataset_summary.md (real markdown about the evaluation set)
"""

import subprocess
import tempfile
import uuid
from pathlib import Path

import httpx
import pytest

RUNTIME = "http://localhost:8000"
DASHBOARD = "http://localhost:8501"
TENANT_ID = "flywheel_org:production"

# Paths to real test artifacts in the repo
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
E2E_ARTIFACT_DIR = Path(tempfile.gettempdir()) / "cogniverse_e2e_artifacts"


def runtime_available() -> bool:
    try:
        r = httpx.get(f"{RUNTIME}/health", timeout=5.0)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.ReadTimeout):
        return False


def dashboard_available() -> bool:
    try:
        r = httpx.get(DASHBOARD, timeout=5.0)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.ReadTimeout):
        return False


skip_if_no_runtime = pytest.mark.skipif(
    not runtime_available(),
    reason="Runtime not available at localhost:8000",
)

skip_if_no_dashboard = pytest.mark.skipif(
    not dashboard_available(),
    reason="Dashboard not available at localhost:8501",
)


def unique_id(prefix: str = "e2e") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def browser_type_launch_args():
    return {"headless": True}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1920, "height": 1080}}


def wait_for_streamlit(page, timeout: int = 30_000):
    """Wait for Streamlit app to fully render."""
    page.wait_for_selector(
        '[data-testid="stAppViewContainer"]', timeout=timeout
    )
    page.wait_for_load_state("networkidle")


def _click_tab_by_label(page, label: str, retries: int = 3):
    """Click a Streamlit tab by matching its visible text (ignoring emojis).

    Streamlit renders tabs as hidden DOM elements, so we use JS click
    which bypasses visibility checks. Retries handle the case where
    sub-tabs haven't rendered yet after a top-tab switch.
    """
    for attempt in range(retries):
        tabs = page.locator('button[role="tab"]')
        count = tabs.count()
        for i in range(count):
            tab = tabs.nth(i)
            text = tab.text_content() or ""
            if label.lower() in text.lower():
                # Use force=True to click hidden elements while still
                # triggering Playwright's full click event sequence
                # (mousedown→mouseup→click) which properly activates
                # React/Streamlit event handlers, unlike raw el.click()
                tab.click(force=True)
                page.wait_for_timeout(2_000)
                page.wait_for_load_state("networkidle")
                return
        if attempt < retries - 1:
            page.wait_for_timeout(3_000)
    tab_texts = [
        tabs.nth(i).text_content() or "" for i in range(tabs.count())
    ]
    raise ValueError(
        f"Tab '{label}' not found after {retries} attempts. "
        f"Available tabs: {tab_texts}"
    )


def click_top_tab(page, label: str, timeout: int = 10_000):
    """Click a top-level Streamlit tab."""
    _click_tab_by_label(page, label)


def click_sub_tab(page, label: str, timeout: int = 10_000):
    """Click a sub-level Streamlit tab."""
    _click_tab_by_label(page, label)


def fill_input(locator, value: str):
    """Fill a Streamlit input, handling both visible and hidden elements.

    Uses keyboard approach (click + type) for visible elements to ensure
    Streamlit picks up the value. Falls back to JS for hidden elements.
    """
    if locator.is_visible():
        locator.click(click_count=3)
        locator.press("Delete")
        locator.type(value, delay=5)
        locator.press("Enter")
    else:
        locator.evaluate(
            """(el, value) => {
                el.focus();
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, 'value'
                ).set;
                nativeSetter.call(el, value);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.blur();
            }""",
            value,
        )


def fill_textarea(locator, value: str):
    """Fill a Streamlit textarea, handling both visible and hidden elements.

    Uses keyboard approach for visible elements. Streamlit textareas
    commit their value on Ctrl+Enter (Enter just adds a newline).
    Falls back to JS for hidden elements.
    """
    if locator.is_visible():
        locator.click(click_count=3)
        locator.press("Delete")
        locator.type(value, delay=5)
        locator.press("Control+Enter")
    else:
        locator.evaluate(
            """(el, value) => {
                el.focus();
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(el, value);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.blur();
            }""",
            value,
        )


def click_button(page, text: str):
    """Click a Streamlit button by text, excluding tab buttons.

    Uses JS click to bypass visibility checks. Excludes buttons with
    role="tab" to avoid accidentally clicking tabs instead of form buttons.
    """
    btn = page.locator(f'button:not([role="tab"]):has-text("{text}")')
    if btn.count() > 0:
        btn.first.evaluate("el => el.click()")
        page.wait_for_timeout(2_000)
        page.wait_for_load_state("networkidle")
        return True
    return False


def expand_sidebar(page):
    """Expand the sidebar if it's collapsed (common in headless mode)."""
    # Streamlit collapses sidebar in narrow viewports / headless
    collapse_btn = page.locator(
        '[data-testid="stSidebarCollapsedControl"], '
        'button[aria-label="Open sidebar"], '
        '[data-testid="collapsedControl"]'
    )
    if collapse_btn.count() > 0 and collapse_btn.first.is_visible():
        collapse_btn.first.click()
        page.wait_for_timeout(1_000)


def set_tenant(page, tenant_id: str, retries: int = 3):
    """Set the active tenant in the sidebar with retry.

    Targets the 'Active Tenant' input specifically (not 'Tenant ID').
    Retries if the value doesn't stick (Streamlit session state timing).
    """
    expand_sidebar(page)

    sidebar = page.locator('[data-testid="stSidebar"]')
    tenant_input = sidebar.locator('input[aria-label="Active Tenant"]')

    for attempt in range(retries):
        tenant_input.click(click_count=3, force=True)
        page.keyboard.press("Delete")
        tenant_input.type(tenant_id, delay=30)
        tenant_input.press("Enter")
        page.wait_for_timeout(4_000)
        page.wait_for_load_state("networkidle")

        # Verify tenant was committed to Streamlit session state
        # by checking for the confirmation alert
        tenant_alert = page.locator(
            '[data-testid="stAlert"]:has-text("Current tenant")'
        )
        if tenant_alert.count() > 0:
            return
    raise RuntimeError(
        f"set_tenant failed: tenant '{tenant_id}' was not committed to "
        f"Streamlit session state after {retries} attempts. "
        "Expected 'Current tenant' confirmation alert to appear."
    )


# ---------------------------------------------------------------------------
# Real test artifact fixtures for ingestion tests
# ---------------------------------------------------------------------------

# Video-ChatGPT paper (arxiv) — directly related to the test dataset
ARXIV_PDF_URL = "https://arxiv.org/pdf/2306.05424"


def _download_if_missing(url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Download a file if it doesn't already exist (cached across test runs)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


@pytest.fixture(scope="session")
def real_pdf_path():
    """Download Video-ChatGPT arxiv paper (related to test videos)."""
    dest = E2E_ARTIFACT_DIR / "video_chatgpt_2306.05424.pdf"
    try:
        return _download_if_missing(ARXIV_PDF_URL, dest)
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"Cannot download test PDF: {exc}")


@pytest.fixture(scope="session")
def real_image_path():
    """Real 1280x720 Big Buck Bunny keyframe from processed data."""
    path = (
        DATA_ROOT / "testset" / "evaluation" / "processed"
        / "keyframes" / "big_buck_bunny_clip" / "frame_0000.jpg"
    )
    if not path.exists():
        pytest.skip(f"Keyframe image not found: {path}")
    return path


@pytest.fixture(scope="session")
def real_video_path():
    """Real 874KB ActivityNet sample video."""
    path = DATA_ROOT / "testset" / "evaluation" / "sample_videos" / "v_-nl4G-00PtA.mp4"
    if not path.exists():
        pytest.skip(f"Sample video not found: {path}")
    return path


@pytest.fixture(scope="session")
def real_document_path():
    """Real dataset summary markdown from the repo."""
    path = DATA_ROOT / "testset" / "dataset_summary.md"
    if not path.exists():
        pytest.skip(f"Document not found: {path}")
    return path


@pytest.fixture(scope="session")
def extracted_audio_path(real_video_path):
    """Extract real audio from sample video using ffmpeg."""
    dest = E2E_ARTIFACT_DIR / "extracted_video_audio.wav"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", str(real_video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-t", "10", str(dest), "-y",
            ],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"Cannot extract audio via ffmpeg: {exc}")
    return dest
