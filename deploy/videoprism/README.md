# VideoPrism inference sidecar

Custom JAX sidecar that serves Google DeepMind **VideoPrism** chunk-level
video embeddings. There is no upstream containerized release, so this image
is built locally and addressed via the `videoprism_jax` inference engine
(profiles in the `video_videoprism_*` family).

## Build

```bash
docker build -t cogniverse/videoprism:dev deploy/videoprism/
```

Heavy build (~5 GB): it installs the JAX / flax / tensorflow stack and the
upstream `videoprism` package from git
(`git+https://github.com/google-deepmind/videoprism.git`). See the
`Dockerfile` for the pinned versions.

## Run

```bash
docker run --rm -p 7999:7999 -e JAX_PLATFORM_NAME=cpu cogniverse/videoprism:dev
```

`server.py` exposes a single endpoint:

- `POST /v1/video/embeddings` — embeds sampled frames of a video chunk.
- `GET /health`

**Video embeddings only.** There is no text-embedding endpoint; text query
encoding for the LVT models runs in-process via the local `videoprism`
package (`cogniverse_core.common.models.videoprism_text_encoder`), not here.

## Models

The upstream `videoprism` package registers four checkpoints:

| Name (`vp.MODELS`) | Type | Text encoding |
|---|---|---|
| `videoprism_public_v1_base` / `_large` | patch (multi-vector) | no — video only |
| `videoprism_lvt_public_v1_base` / `_large` | LVT (single-vector) | yes |

The chart/config refers to these by their HuggingFace aliases
(`..._hf`); the server maps the alias back to the bare upstream name via
`_HF_SUFFIX_MAP`. The default model is `videoprism_public_v1_base_hf`.

## Tests

The ingestion integration suite spawns this sidecar automatically when the
image is present:

- `tests/ingestion/integration/conftest.py` boots it once per session and
  populates `INFERENCE_SERVICE_URLS["videoprism_jax"]`.
- Tests are gated on the image existing
  (`requires_videoprism_jax` / `_videoprism_image_built()`); they skip with
  the build command above if it is not built locally.
