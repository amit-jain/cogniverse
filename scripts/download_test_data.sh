#!/usr/bin/env bash
#
# Downloads evaluation dataset for Cogniverse.
#
# Sources:
#   - Video-ChatGPT benchmark (MBZUAI-Oryx): QA files, human captions, 500 test videos
#   - ActivityNet-200: 10 selected test videos via yt-dlp
#   - Blender Foundation: Big Buck Bunny (CC-BY 3.0), Elephants Dream (CC-BY 2.5)
#   - Google: For Bigger Blazes test video
#
# Usage:
#   ./scripts/download_test_data.sh              # Download everything
#   ./scripts/download_test_data.sh --test-only   # Download only 10+3 test videos
#   ./scripts/download_test_data.sh --no-videos   # Download QA/captions only (skip 500 videos)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/testset"

# Video-ChatGPT SharePoint URLs
VIDEOS_URL="https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EatOpE7j68tLm2XAd0u6b8ABGGdVAwLMN6rqlDGM_DwhVA?e=90WIuW&download=1"
QA_URL="https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EoS-mdm-KchDqCVbGv8v-9IB_ZZNXtcYAHtyvI06PqbF_A?e=1sNbaa"
CAPTIONS_URL="https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EYqblLdszspJkayPvVIm5s0BCvl0m6q6B-ipmrNg-pqn6A?e=QFzc1U&download=1"

# 10 ActivityNet test video YouTube IDs
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

# Open-source test videos
BIG_BUCK_BUNNY_URL="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
ELEPHANTS_DREAM_URL="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"
FOR_BIGGER_BLAZES_URL="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"

# --- Helpers ---

check_deps() {
    local missing=()
    command -v curl  >/dev/null 2>&1 || missing+=(curl)
    command -v unzip >/dev/null 2>&1 || missing+=(unzip)

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "ERROR: Missing required tools: ${missing[*]}"
        exit 1
    fi

    if [[ "$DOWNLOAD_TEST_VIDEOS" == "true" ]]; then
        if ! command -v yt-dlp >/dev/null 2>&1; then
            echo "ERROR: yt-dlp is required for downloading test videos."
            echo "Install: pip install yt-dlp  OR  brew install yt-dlp"
            exit 1
        fi
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
    curl -fSL -o "$output" "$url" || {
        echo "  WARN: Failed to download $description"
        echo "        URL: $url"
        echo "        You may need to download manually from the Video-ChatGPT repo."
        return 1
    }
}

# --- Download functions ---

download_qa_files() {
    echo ""
    echo "=== QA Files (Video-ChatGPT) ==="
    local qa_dir="$DATA_DIR/queries"
    mkdir -p "$qa_dir"

    echo "  NOTE: QA files are hosted on SharePoint. If automatic download fails,"
    echo "        download manually from the Video-ChatGPT repo:"
    echo "        https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/quantitative_evaluation/README.md"
    echo ""

    # The SharePoint folder link may not support direct download.
    # Clone the specific files from the GitHub repo instead.
    local gh_raw="https://raw.githubusercontent.com/mbzuai-oryx/Video-ChatGPT/main/quantitative_evaluation"
    for qa_file in generic_qa.json temporal_qa.json consistency_qa.json; do
        download_file \
            "$gh_raw/benchmarking/$qa_file" \
            "$qa_dir/$qa_file" \
            "$qa_file"
    done
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
    download_file "$CAPTIONS_URL" "$tmp_file" "Human annotated captions"
    if [[ -f "$tmp_file" ]]; then
        echo "  Extracting captions..."
        unzip -qo "$tmp_file" -d "$captions_dir" 2>/dev/null || {
            # May be a tar or other format
            echo "  NOTE: Could not unzip. File may need manual extraction."
            echo "        Downloaded to: $tmp_file"
            return 1
        }
        rm -f "$tmp_file"
        echo "  OK: $(ls "$captions_dir" | wc -l | tr -d ' ') caption files"
    fi
}

download_full_videos() {
    echo ""
    echo "=== Full Test Videos (500 ActivityNet) ==="
    local videos_dir="$DATA_DIR/Test_Videos"
    mkdir -p "$videos_dir"

    if [[ "$(ls "$videos_dir"/*.mp4 2>/dev/null | wc -l)" -gt 100 ]]; then
        echo "  SKIP: Test_Videos already contains $(ls "$videos_dir"/*.mp4 | wc -l) videos"
        return 0
    fi

    local tmp_file="$DATA_DIR/.videos_download.zip"
    download_file "$VIDEOS_URL" "$tmp_file" "500 ActivityNet test videos (~11GB)"
    if [[ -f "$tmp_file" ]]; then
        echo "  Extracting videos (this may take a while)..."
        unzip -qo "$tmp_file" -d "$videos_dir" 2>/dev/null || {
            echo "  NOTE: Could not unzip. File may need manual extraction."
            echo "        Downloaded to: $tmp_file"
            return 1
        }
        rm -f "$tmp_file"
        echo "  OK: $(ls "$videos_dir"/*.mp4 2>/dev/null | wc -l) video files"
    fi
}

download_test_videos() {
    echo ""
    echo "=== Test Videos (10 ActivityNet + 3 Open-Source) ==="
    local test_dir="$DATA_DIR/evaluation/test_videos"
    mkdir -p "$test_dir"

    # Download 10 ActivityNet videos via yt-dlp
    for yt_id in "${TEST_VIDEO_IDS[@]}"; do
        local output_name="v_${yt_id}"
        # Check if already downloaded (any extension)
        if ls "$test_dir/${output_name}".* >/dev/null 2>&1; then
            echo "  SKIP: $output_name (already exists)"
            continue
        fi

        echo "  Downloading: $output_name from YouTube ..."
        yt-dlp \
            --no-playlist \
            --format "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
            --output "$test_dir/${output_name}.%(ext)s" \
            "https://www.youtube.com/watch?v=${yt_id}" 2>/dev/null || {
            echo "  WARN: Failed to download $output_name (video may be unavailable)"
        }
    done

    # Download 3 open-source videos
    download_file "$BIG_BUCK_BUNNY_URL" "$test_dir/big_buck_bunny_clip.mp4" "Big Buck Bunny (CC-BY 3.0)"
    download_file "$ELEPHANTS_DREAM_URL" "$test_dir/elephant_dream_clip.mp4" "Elephants Dream (CC-BY 2.5)"
    download_file "$FOR_BIGGER_BLAZES_URL" "$test_dir/for_bigger_blazes.mp4" "For Bigger Blazes"

    echo "  OK: $(ls "$test_dir" | wc -l | tr -d ' ') test videos"
}

# --- Main ---

DOWNLOAD_FULL_VIDEOS="true"
DOWNLOAD_TEST_VIDEOS="true"

case "${1:-}" in
    --test-only)
        DOWNLOAD_FULL_VIDEOS="false"
        echo "Mode: test videos only (10 ActivityNet + 3 open-source)"
        ;;
    --no-videos)
        DOWNLOAD_FULL_VIDEOS="false"
        DOWNLOAD_TEST_VIDEOS="false"
        echo "Mode: QA files and captions only (no video downloads)"
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

echo ""
echo "=== Done ==="
echo "Dataset directory: $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Run ingestion: uv run python scripts/run_ingestion.py --video_dir $DATA_DIR/evaluation/test_videos --backend vespa"
echo "  2. Run evaluation: uv run python tests/comprehensive_video_query_test_v2.py"
