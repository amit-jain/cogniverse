# VideoPrism inference sidecar

Custom JAX sidecar that serves Google DeepMind **VideoPrism** chunk-level
video embeddings. There is no upstream containerized release, so this image
is built locally and addressed via the `videoprism_jax` inference engine
(profiles in the `video_videoprism_*` family).

## Build

```bash
docker build -f deploy/videoprism/Dockerfile \
  -t cogniverse/videoprism:0.1.0-dev .
```

Heavy build (~5 GB): it installs the JAX / flax / tensorflow stack and the
upstream `videoprism` package from git
(`git+https://github.com/google-deepmind/videoprism.git`). See the
`Dockerfile` for the pinned versions.

## Run

```bash
docker run --rm -p 7999:7999 \
  -e JAX_PLATFORM_NAME=cpu \
  -e MODEL_NAME=videoprism_public_v1_base_hf \
  cogniverse/videoprism:0.1.0-dev
```

`cogniverse_cli.modal_inference.servers.videoprism` is copied into the image
as `server.py` and exposes two routes: one embedding endpoint plus health.

- `POST /v1/video/embeddings` — embeds sampled frames of a video chunk.
- `GET /health`

**Video embeddings only.** There is no text-embedding endpoint and this
service does not serve an LVT or large checkpoint.

## Models

The service accepts exactly `videoprism_public_v1_base_hf`, backed by upstream
model `videoprism_public_v1_base`. The Hugging Face checkpoint revision is
`be719a406d563b66f0ac969e7c94bab8e997c81a`; the pinned VideoPrism source
revision is `d481d91b9bf8c9d330d1e526e511a359c799bbe1`. `MODEL_NAME`, when set,
must equal that canonical service model. Requests naming another model fail
validation instead of selecting another checkpoint.

## Tests

Integration tests declare `@pytest.mark.requires_inference("videoprism_jax")`.
The shared session fixture resolves the exact pinned service from its enabled
provider. An explicit endpoint or enabled Modal lifecycle is authoritative, so
its failure does not trigger a local replacement. When neither is enabled and
no exact cluster endpoint exists, the fixture builds
`cogniverse/videoprism:0.1.0-dev`, starts a uniquely named test-owned container
on a free host port, validates `/health`, and publishes the resolved URL to the
production Cogniverse client. It removes that container during session
teardown. Missing images are built; service build, startup, health, or identity
failures fail setup and are never converted into skips.
