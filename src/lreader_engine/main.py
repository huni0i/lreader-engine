from fastapi import FastAPI, HTTPException

from lreader_engine.models import ChapterRequest


app = FastAPI(title="Lreader local engine", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chapters")
def translate_chapter(request: ChapterRequest) -> None:
    del request
    raise HTTPException(
        status_code=501,
        detail="The OCR pipeline is not connected yet.",
    )
