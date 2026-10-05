# Changelog

## 0.10.0

Initial public release.

Highlights:

- Distributed and Remote Only scheduling against stock InvokeAI v7 workers
- queue-safe worker claiming and offline-worker recovery
- live remote progress and preview relay
- image/video result import with source-node preservation
- image/video workflow input transfer and remapping
- terminal handling for clear remote GPU OOM failures
- single-file LAN model transfer through stock InvokeAI model installation
- Hugging Face directory/Diffusers installation through stock remote APIs
- exact HF subfolder and repo-variant preservation
- shared singleflight model installs for concurrent generations
- stale remote model-record rejection through stock `/api/v2/models/missing`
- safer remote cleanup and retained failed remote queue rows
- code split between invocation definition and backend-style worker-pool implementation
