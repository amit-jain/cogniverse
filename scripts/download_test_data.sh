#!/usr/bin/env bash
#
# Downloads evaluation dataset for Cogniverse.
#
# Sources:
#   - lmms-lab/VideoChatGPT on HuggingFace: 500 test videos (videos.zip),
#     QA parquet files (Generic, Temporal, Consistency)
#   - 10 ActivityNet test videos: extracted from the HF videos.zip via
#     range requests (the original YouTube IDs are no longer reachable)
#   - Internet Archive: Big Buck Bunny, Elephants Dream, For Bigger Blazes
#   - Human-annotated captions: SharePoint behind auth, manual download only
#
# Usage:
#   ./scripts/download_test_data.sh              # Download everything
#   ./scripts/download_test_data.sh --test-only   # Download only 10+3 test videos + QA
#   ./scripts/download_test_data.sh --no-videos   # Download QA only (skip videos)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/testset"

DOWNLOAD_FAILURES=()

# HuggingFace mirror for Video-ChatGPT (Generic/Temporal/Consistency QA + videos.zip)
HF_BASE="https://huggingface.co/datasets/lmms-lab/VideoChatGPT/resolve/main"
HF_VIDEOS_ZIP_URL="$HF_BASE/videos.zip"

# 10 ActivityNet test video IDs (referenced by tests/e2e/conftest.py and others)
TEST_VIDEO_IDS=(
    "-6dz6tBH77I"
    "-D1gdv_gQyw"
    "-HpCLXdtcas"
    "-IMXSEIabMM"
    "-MbZ-W0AbN0"
    "-cAcA8dO7kA"
    "-nl4G-00PtA"
    "-pkfcMUIEMo"
    "-uJnucdW6DY"
    "-vnSFKJNB94"
)

# Open-source test videos (Internet Archive mirrors; the gtv-videos-bucket is offline)
BIG_BUCK_BUNNY_URL="https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4"
ELEPHANTS_DREAM_URL="https://archive.org/download/ElephantsDream/ed_1024_512kb.mp4"
FOR_BIGGER_BLAZES_URL="https://archive.org/download/for-bigger-blazes/ForBiggerBlazes.mp4"

# Human-annotated captions (SharePoint personal share — works with a browser UA)
CAPTIONS_URL="https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EYqblLdszspJkayPvVIm5s0BCvl0m6q6B-ipmrNg-pqn6A?e=QFzc1U&download=1"

# SharePoint and some CDNs reject default curl UAs; impersonate a recent browser
BROWSER_UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'

# --- Helpers ---

check_deps() {
    local missing=()
    command -v curl    >/dev/null 2>&1 || missing+=(curl)
    command -v unzip   >/dev/null 2>&1 || missing+=(unzip)
    command -v python3 >/dev/null 2>&1 || missing+=(python3)
    command -v uv      >/dev/null 2>&1 || missing+=(uv)

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "ERROR: Missing required tools: ${missing[*]}"
        echo "       Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

download_file() {
    local url="$1"
    local output="$2"
    local description="$3"

    if [[ -f "$output" ]]; then
        echo "  SKIP: $description (already exists)"
        return 0
    fi

    echo "  Downloading: $description ..."
    if curl -fSL -A "$BROWSER_UA" --cookie-jar /dev/null -o "$output" "$url"; then
        return 0
    fi

    rm -f "$output"
    echo "  WARN: Failed to download $description"
    echo "        URL: $url"
    DOWNLOAD_FAILURES+=("$description ($url)")
    return 1
}

# --- Download functions ---

download_qa_files() {
    echo ""
    echo "=== QA Files (Video-ChatGPT via HuggingFace) ==="
    local qa_dir="$DATA_DIR/queries"
    mkdir -p "$qa_dir"

    local tmp_dir="$DATA_DIR/.qa_parquet"
    mkdir -p "$tmp_dir"

    local mappings=(
        "Generic:generic_qa.json"
        "Temporal:temporal_qa.json"
        "Consistency:consistency_qa.json"
    )

    for entry in "${mappings[@]}"; do
        local cfg="${entry%:*}"
        local out="${entry#*:}"
        local out_path="$qa_dir/$out"

        if [[ -f "$out_path" ]]; then
            echo "  SKIP: $out (already exists)"
            continue
        fi

        local pq_path="$tmp_dir/${cfg}.parquet"
        local pq_url="$HF_BASE/${cfg}/test-00000-of-00001.parquet"

        download_file "$pq_url" "$pq_path" "$cfg parquet" || continue

        echo "  Converting $cfg parquet -> $out ..."
        if uv run --quiet --with pyarrow python - "$pq_path" "$out_path" <<'PY'
import json, sys
import pyarrow.parquet as pq
records = pq.read_table(sys.argv[1]).to_pylist()
with open(sys.argv[2], 'w') as f:
    json.dump(records, f, indent=2)
print(f"  OK: {len(records)} records -> {sys.argv[2]}")
PY
        then
            :
        else
            echo "  WARN: Conversion failed for $cfg"
            DOWNLOAD_FAILURES+=("$out (parquet->json conversion failed)")
        fi
    done

    rm -rf "$tmp_dir"
}

download_captions() {
    echo ""
    echo "=== Human Annotated Captions ==="
    local captions_dir="$DATA_DIR/Test_Human_Annotated_Captions"
    mkdir -p "$captions_dir"

    if [[ "$(ls -A "$captions_dir" 2>/dev/null)" ]]; then
        echo "  SKIP: Captions directory already populated"
        return 0
    fi

    local tmp_file="$DATA_DIR/.captions_download.zip"
    download_file "$CAPTIONS_URL" "$tmp_file" "Test_Human_Annotated_Captions.zip" || return 1

    echo "  Extracting captions..."
    if unzip -qoj "$tmp_file" -d "$captions_dir" 2>/dev/null; then
        rm -f "$tmp_file"
        echo "  OK: $(ls "$captions_dir" | wc -l | tr -d ' ') caption files"
    else
        echo "  WARN: Could not unzip $tmp_file"
        DOWNLOAD_FAILURES+=("captions unzip failed; archive at $tmp_file")
        return 1
    fi
}

extract_from_hf_zip() {
    local dest="$1"
    shift
    local -a names=("$@")

    if [[ ${#names[@]} -eq 0 ]]; then
        return 0
    fi

    mkdir -p "$dest"

    uv run --quiet --with remotezip python - "$HF_VIDEOS_ZIP_URL" "$dest" "${names[@]}" <<'PY'
import os
import sys
from remotezip import RemoteZip

url = sys.argv[1]
dest = sys.argv[2]
wanted = set(sys.argv[3:])
created_subdirs = set()

def stem(path):
    base = path.rsplit('/', 1)[-1]
    return base.rsplit('.', 1)[0] if '.' in base else base

with RemoteZip(url) as zf:
    all_names = zf.namelist()
    by_stem = {}
    for n in all_names:
        s = stem(n)
        if s:
            by_stem.setdefault(s, []).append(n)

    for w in wanted:
        candidates = by_stem.get(w, [])
        if not candidates:
            print(f"  MISS: {w}")
            continue
        member = candidates[0]
        zf.extract(member, path=dest)
        out_basename = member.rsplit('/', 1)[-1]
        src = os.path.join(dest, member)
        tgt = os.path.join(dest, out_basename)
        if src != tgt:
            os.makedirs(os.path.dirname(tgt) or ".", exist_ok=True)
            os.replace(src, tgt)
            created_subdirs.add(os.path.dirname(src))
        print(f"  OK:   {out_basename}")

for d in sorted(created_subdirs, key=len, reverse=True):
    try:
        os.rmdir(d)
    except OSError:
        pass
PY
}

download_full_videos() {
    echo ""
    echo "=== Full Test Videos (500 from lmms-lab/VideoChatGPT, ~11 GB) ==="
    local videos_dir="$DATA_DIR/Test_Videos"
    mkdir -p "$videos_dir"

    if [[ "$(ls "$videos_dir"/*.mp4 2>/dev/null | wc -l)" -gt 100 ]]; then
        echo "  SKIP: Test_Videos already contains $(ls "$videos_dir"/*.mp4 | wc -l) videos"
        return 0
    fi

    local tmp_file="$DATA_DIR/.videos_download.zip"
    download_file "$HF_VIDEOS_ZIP_URL" "$tmp_file" "lmms-lab videos.zip (~11 GB)" || return 1

    echo "  Extracting videos (this may take a while)..."
    if unzip -qoj "$tmp_file" -d "$videos_dir" 2>/dev/null; then
        rm -f "$tmp_file"
        echo "  OK: $(ls "$videos_dir"/*.mp4 2>/dev/null | wc -l) video files"
    else
        echo "  WARN: unzip failed; archive left at $tmp_file"
        DOWNLOAD_FAILURES+=("Full test videos (unzip failed)")
        return 1
    fi
}

download_test_videos() {
    echo ""
    echo "=== Test Videos (10 ActivityNet + 3 Open-Source) ==="
    local test_dir="$DATA_DIR/evaluation/sample_videos"
    mkdir -p "$test_dir"

    local needed=()
    for yt_id in "${TEST_VIDEO_IDS[@]}"; do
        local stem="v_${yt_id}"
        local existing
        existing=$(compgen -G "$test_dir/${stem}.*" 2>/dev/null | head -1)
        if [[ -n "$existing" ]]; then
            echo "  SKIP: $(basename "$existing") (already exists)"
            continue
        fi
        needed+=("$stem")
    done

    if [[ ${#needed[@]} -gt 0 ]]; then
        echo "  Extracting ${#needed[@]} ActivityNet videos from HF videos.zip ..."
        if ! extract_from_hf_zip "$test_dir" "${needed[@]}"; then
            echo "  WARN: HF zip extraction failed"
            DOWNLOAD_FAILURES+=("ActivityNet test videos (HF extraction failed)")
        fi
        for stem in "${needed[@]}"; do
            if ! compgen -G "$test_dir/${stem}.*" >/dev/null 2>&1; then
                DOWNLOAD_FAILURES+=("$stem.* (not in HF videos.zip)")
            fi
        done
    fi

    download_file "$BIG_BUCK_BUNNY_URL"     "$test_dir/big_buck_bunny_clip.mp4"  "Big Buck Bunny (CC-BY 3.0)"
    download_file "$ELEPHANTS_DREAM_URL"    "$test_dir/elephant_dream_clip.mp4"  "Elephants Dream (CC-BY 2.5)"
    download_file "$FOR_BIGGER_BLAZES_URL"  "$test_dir/for_bigger_blazes.mp4"    "For Bigger Blazes"

    echo "  OK: $(ls "$test_dir"/*.mp4 2>/dev/null | wc -l) test videos in $test_dir"
}

print_summary() {
    echo ""
    echo "=== Done ==="
    echo "Dataset directory: $DATA_DIR"

    if [[ ${#DOWNLOAD_FAILURES[@]} -gt 0 ]]; then
        echo ""
        echo "Sources that failed or require manual action (${#DOWNLOAD_FAILURES[@]}):"
        for f in "${DOWNLOAD_FAILURES[@]}"; do
            echo "  - $f"
        done
    fi

    echo ""
    echo "Next steps:"
    echo "  1. Run ingestion: uv run python scripts/run_ingestion.py --video_dir $DATA_DIR/evaluation/sample_videos --backend vespa"
    echo "  2. Run evaluation: uv run python tests/comprehensive_video_query_test_v2.py"
}

# --- Main ---

DOWNLOAD_FULL_VIDEOS="true"
DOWNLOAD_TEST_VIDEOS="true"

case "${1:-}" in
    --test-only)
        DOWNLOAD_FULL_VIDEOS="false"
        echo "Mode: test videos only (10 ActivityNet + 3 open-source) + QA"
        ;;
    --no-videos)
        DOWNLOAD_FULL_VIDEOS="false"
        DOWNLOAD_TEST_VIDEOS="false"
        echo "Mode: QA files only (no video downloads)"
        ;;
    "")
        echo "Mode: full download (all data)"
        ;;
    *)
        echo "Usage: $0 [--test-only | --no-videos]"
        exit 1
        ;;
esac

echo "Target: $DATA_DIR"
check_deps
mkdir -p "$DATA_DIR"

download_qa_files
download_captions

if [[ "$DOWNLOAD_FULL_VIDEOS" == "true" ]]; then
    download_full_videos
fi

if [[ "$DOWNLOAD_TEST_VIDEOS" == "true" ]]; then
    download_test_videos
fi

print_summary
