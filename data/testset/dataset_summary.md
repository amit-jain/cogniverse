# Evaluation Dataset

## Sources

### Video-ChatGPT Benchmark (MBZUAI-Oryx)

- **Repository**: https://github.com/mbzuai-oryx/Video-ChatGPT
- **Paper**: [Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models](https://arxiv.org/abs/2306.05424) (ACL 2024)
- **License**: Check repository for terms

Provides:
- **500 test videos** from ActivityNet-200 (YouTube-sourced)
- **Human-annotated captions** (499 text descriptions)
- **QA pairs** across 3 categories: generic, temporal, consistency

Download links (from Video-ChatGPT quantitative evaluation):
- Test videos: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EatOpE7j68tLm2XAd0u6b8ABGGdVAwLMN6rqlDGM_DwhVA?e=90WIuW
- QA files: https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EoS-mdm-KchDqCVbGv8v-9IB_ZZNXtcYAHtyvI06PqbF_A?e=1sNbaa
- Human captions: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EYqblLdszspJkayPvVIm5s0BCvl0m6q6B-ipmrNg-pqn6A?e=QFzc1U

### ActivityNet-200

- **Website**: http://activity-net.org/
- **Videos**: YouTube clips covering 200 activity categories
- **Video naming**: `v_` prefix + YouTube video ID (e.g., `v_-6dz6tBH77I` → youtube.com/watch?v=-6dz6tBH77I)

### Blender Foundation (Creative Commons)

- **Big Buck Bunny**: https://peach.blender.org/ — CC-BY 3.0
- **Elephants Dream**: https://orange.blender.org/ — CC-BY 2.5

### Google Test Videos

- **For Bigger Blazes**: https://www.youtube.com/watch?v=dMH0bHeiRNg

## Directory Structure

After running `scripts/download_test_data.sh`:

```
data/testset/
├── dataset_summary.md              # This file
├── Test_Videos/                     # 500 ActivityNet evaluation videos (downloaded)
├── Test_Human_Annotated_Captions/   # 499 human descriptions (downloaded)
├── queries/                         # Video-ChatGPT QA files (downloaded)
│   ├── generic_qa.json
│   ├── temporal_qa.json
│   └── consistency_qa.json
└── evaluation/
    ├── test_videos/                 # 10 selected ActivityNet + 3 open-source (downloaded)
    ├── extract_sample_video_queries.py  # Script to extract queries for test videos
    ├── sample_videos_retrieval_queries.json  # 125 extracted queries
    └── processed/                   # LLM-generated cache (committed)
        ├── descriptions/            # VLM frame descriptions
        ├── transcripts/             # Whisper transcripts
        └── metadata/               # Pipeline processing metadata
```

## Test Video IDs

The 10 ActivityNet videos selected for development testing:

| Filename | YouTube ID |
|----------|-----------|
| v_-6dz6tBH77I | -6dz6tBH77I |
| v_-D1gdv_gQyw | -D1gdv_gQyw |
| v_-HpCLXdtcas | -HpCLXdtcas |
| v_-IMXSEIabMM | -IMXSEIabMM |
| v_-MbZ-W0AbN0 | -MbZ-W0AbN0 |
| v_-cAcA8dO7kA | -cAcA8dO7kA |
| v_-nl4G-00PtA | -nl4G-00PtA |
| v_-pkfcMUIEMo | -pkfcMUIEMo |
| v_-uJnucdW6DY | -uJnucdW6DY |
| v_-vnSFKJNB94 | -vnSFKJNB94 |

## What's Committed vs Downloaded

| Path | In Git | Notes |
|------|--------|-------|
| `dataset_summary.md` | Yes | Attribution and setup docs |
| `evaluation/extract_sample_video_queries.py` | Yes | Our query extraction script |
| `evaluation/sample_videos_retrieval_queries.json` | Yes | Our 125 extracted queries |
| `evaluation/processed/descriptions/` | Yes | LLM-generated cache (saves ingestion time) |
| `evaluation/processed/transcripts/` | Yes | Whisper cache |
| `evaluation/processed/metadata/` | Yes | Pipeline metadata |
| `Test_Videos/` | No | Run `scripts/download_test_data.sh` |
| `Test_Human_Annotated_Captions/` | No | Run `scripts/download_test_data.sh` |
| `queries/` | No | Run `scripts/download_test_data.sh` |
| `evaluation/test_videos/` | No | Run `scripts/download_test_data.sh` |
