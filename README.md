# Lreader Engine

Local-first multilingual OCR and translation service for
[lreader-extension](https://github.com/huni0i/lreader-extension).

## Supported MVP languages

- Source: Japanese, English, and Chinese
- Target: Korean, English, Japanese, and Chinese

The API schema also accepts Korean as a source language so translated pages can
be processed consistently.

## Development

ML dependencies require Python 3.11–3.13. Python 3.14 is intentionally excluded
until the OCR and PyTorch ecosystem supports it.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn lreader_engine.main:app --reload
```

Run tests:

```bash
pytest
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

## RTX 3090 deployment

The CUDA image uses Hy-MT2-7B, keeps one API worker, and stores downloaded
models outside the container.

```bash
docker compose build
docker compose up -d
docker compose logs -f engine
```

Verify GPU use and service health:

```bash
docker compose exec engine nvidia-smi
curl http://127.0.0.1:8765/health
```

The service binds to the server loopback interface by default. From the client
Mac, create a Tailscale SSH tunnel so the extension can keep using localhost:

```bash
ssh -fN -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 8765:127.0.0.1:8765 huni0i@100.107.63.5
```

When the tunnel drops too often, bind the service to the Tailscale interface
instead and let the extension reach the server directly. Create `.env` next to
`compose.yaml` with the server Tailscale address:

```bash
echo "LREADER_BIND_ADDRESS=100.107.63.5" > .env
docker compose up -d
```

The port stays private to the Tailscale network, so no LAN or public interface
is exposed.

Stop or update the service:

```bash
docker compose down
git pull
docker compose up -d --build
```

## Planned pipeline

1. Download and normalize chapter images.
2. Detect comic text regions and cleanup masks.
3. Route crops to multilingual or language-specialized OCR.
4. Translate ordered dialogue with chapter context.
5. Return text, confidence, orientation, and relative coordinates.

The browser extension owns DOM overlays and scrolling behavior. This service
owns image understanding and translation.
