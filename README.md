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
curl http://127.0.0.1:8000/health
```

## Planned pipeline

1. Download and normalize chapter images.
2. Detect comic text regions and cleanup masks.
3. Route crops to multilingual or language-specialized OCR.
4. Translate ordered dialogue with chapter context.
5. Return text, confidence, orientation, and relative coordinates.

The browser extension owns DOM overlays and scrolling behavior. This service
owns image understanding and translation.
