from pathlib import Path

import torch

from lreader_engine.eval import Box, EvalPage
from lreader_engine.eval_datasets import load_synthetic_pages
from lreader_engine.ocr_router import (
    ComicOcrRouter,
    RouterDataset,
    boxes_to_heatmap,
    router_loss,
    split_name,
    train_router,
)
from lreader_engine.synthetic_comics import generate_synthetic_split


def test_boxes_to_heatmap_fills_the_covered_cells() -> None:
    heatmap = boxes_to_heatmap(
        [Box(left=0, top=0, right=50, bottom=50)],
        image_width=100,
        image_height=100,
        map_size=4,
    )

    assert heatmap.shape == (1, 4, 4)
    assert heatmap[0, 0, 0] == 1.0
    assert heatmap[0, 3, 3] == 0.0


def test_router_forward_shapes() -> None:
    model = ComicOcrRouter()
    images = torch.zeros(2, 3, 128, 128)
    route, heatmap = model(images)

    assert route.shape == (2,)
    assert heatmap.shape == (2, 1, 16, 16)
    loss = router_loss(
        route,
        heatmap,
        torch.tensor([1.0, 0.0]),
        torch.zeros(2, 1, 16, 16),
    )
    assert torch.isfinite(loss)


def test_split_name_keeps_manifest_test_pages() -> None:
    page = EvalPage(
        id="ja_000",
        image_path="ja_000.png",
        language="ja",
        split="test",
        source="synthetic-ja-en",
        boxes=[],
        expect_white_bubbles=False,
    )
    assert split_name(page) == "test"


def test_train_router_runs_one_epoch_on_synthetic_pages(tmp_path: Path) -> None:
    generate_synthetic_split(tmp_path, pages_per_language=5)
    pages = load_synthetic_pages(tmp_path)
    assert len(RouterDataset(pages)) == 10

    report = train_router(
        pages,
        tmp_path / "run",
        epochs=1,
        batch_size=4,
        device=torch.device("cpu"),
    )

    assert report["counts"]["train"] >= 1
    assert report["counts"]["test"] >= 1
    assert 0.0 <= report["learned_test"]["route_accuracy"] <= 1.0
    assert (tmp_path / "run" / "best.pt").exists()
