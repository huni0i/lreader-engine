from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lreader_engine.eval import Box, EvalPage


IMAGE_SIZE = 128
HEATMAP_SIZE = 16
ROUTE_HEATMAP_WEIGHT = 0.5


class ComicOcrRouter(nn.Module):
    """Cheap appearance router: white-bubble page vs dark/vertical lettering.

    This is not a new backbone. The method is the cascade decision: predict
    whether EasyOCR is safe, and a coarse text heatmap as a spatial prior.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.route_head = nn.Linear(64, 1)
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        pooled = features.mean(dim=(2, 3))
        route_logit = self.route_head(pooled).squeeze(-1)
        heatmap_logit = self.heatmap_head(features)
        return route_logit, heatmap_logit


def boxes_to_heatmap(
    boxes: list[Box],
    image_width: int,
    image_height: int,
    map_size: int = HEATMAP_SIZE,
) -> torch.Tensor:
    heatmap = torch.zeros(1, map_size, map_size)
    for box in boxes:
        left = int(box.left / image_width * map_size)
        top = int(box.top / image_height * map_size)
        right = min(map_size - 1, max(left, int(box.right / image_width * map_size)))
        bottom = min(map_size - 1, max(top, int(box.bottom / image_height * map_size)))
        heatmap[0, top : bottom + 1, left : right + 1] = 1.0
    return heatmap


def split_name(page: EvalPage) -> str:
    if page.split == "test":
        return "test"
    digest = int(hashlib.md5(page.id.encode("utf-8")).hexdigest(), 16)
    return "val" if digest % 5 == 0 else "train"


class RouterDataset(Dataset):
    def __init__(self, pages: list[EvalPage], *, augment: bool = False) -> None:
        self.pages = pages
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        page = self.pages[index]
        image = Image.open(page.image_path).convert("RGB")
        width, height = image.size
        resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        if self.augment:
            array = np.clip(array * np.random.uniform(0.85, 1.15), 0.0, 1.0)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        route = torch.tensor(1.0 if page.expect_white_bubbles else 0.0)
        heatmap = boxes_to_heatmap(page.boxes, width, height)
        return {"image": tensor, "route": route, "heatmap": heatmap}


def router_loss(
    route_logit: torch.Tensor,
    heatmap_logit: torch.Tensor,
    route: torch.Tensor,
    heatmap: torch.Tensor,
) -> torch.Tensor:
    classification = nn.functional.binary_cross_entropy_with_logits(
        route_logit, route
    )
    localization = nn.functional.binary_cross_entropy_with_logits(
        heatmap_logit,
        heatmap,
        pos_weight=torch.tensor(28.0, device=heatmap_logit.device),
    )
    return classification + ROUTE_HEATMAP_WEIGHT * localization


def heatmap_iou(logit: torch.Tensor, target: torch.Tensor) -> float:
    predicted = logit.sigmoid() >= 0.5
    gold = target >= 0.5
    intersection = (predicted & gold).sum().item()
    union = (predicted | gold).sum().item()
    if union == 0:
        return 1.0
    return intersection / union


def run_epoch(
    model: ComicOcrRouter,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    route_hits = 0
    heatmap_scores: list[float] = []
    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        route = batch["route"].to(device)
        heatmap = batch["heatmap"].to(device)
        if training:
            optimizer.zero_grad()
        route_logit, heatmap_logit = model(images)
        loss = router_loss(route_logit, heatmap_logit, route, heatmap)
        if training:
            loss.backward()
            optimizer.step()
        predicted_route = route_logit.sigmoid() >= 0.5
        route_hits += int((predicted_route == route.bool()).sum().item())
        heatmap_scores.append(heatmap_iou(heatmap_logit, heatmap))
        total_loss += float(loss.item()) * images.size(0)
        count += images.size(0)
    return {
        "loss": total_loss / count if count else 0.0,
        "route_accuracy": route_hits / count if count else 0.0,
        "heatmap_iou": sum(heatmap_scores) / len(heatmap_scores) if heatmap_scores else 0.0,
    }


def opencv_route_accuracy(pages: list[EvalPage]) -> float:
    from lreader_engine.bubble_detector import BubbleDetector

    detector = BubbleDetector()
    labeled = [page for page in pages if page.expect_white_bubbles is not None]
    if not labeled:
        return 0.0
    hits = sum(
        int(detector.has_speech_bubbles(page.image_path) is page.expect_white_bubbles)
        for page in labeled
    )
    return hits / len(labeled)


def train_router(
    pages: list[EvalPage],
    output_dir: Path,
    *,
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    resolved_device = device or torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    grouped: dict[str, list[EvalPage]] = {"train": [], "val": [], "test": []}
    for page in pages:
        if page.expect_white_bubbles is None:
            continue
        grouped[split_name(page)].append(page)
    if not grouped["val"] and grouped["train"]:
        grouped["val"] = grouped["train"]

    model = ComicOcrRouter().to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loaders = {
        name: DataLoader(
            RouterDataset(group, augment=name == "train"),
            batch_size=batch_size,
            shuffle=name == "train",
        )
        for name, group in grouped.items()
        if group
    }
    history: list[dict[str, float]] = []
    best_val = -1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, loaders["train"], resolved_device, optimizer
        )
        val_metrics = run_epoch(model, loaders["val"], resolved_device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_route_accuracy": train_metrics["route_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_route_accuracy": val_metrics["route_accuracy"],
            "val_heatmap_iou": val_metrics["heatmap_iou"],
        }
        history.append(row)
        if val_metrics["route_accuracy"] >= best_val:
            best_val = val_metrics["route_accuracy"]
            torch.save(model.state_dict(), output_dir / "best.pt")

    model.load_state_dict(
        torch.load(output_dir / "best.pt", map_location=resolved_device, weights_only=True)
    )
    test_metrics = run_epoch(model, loaders["test"], resolved_device)
    report = {
        "method": "appearance-conditioned comic OCR router",
        "device": str(resolved_device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "counts": {name: len(group) for name, group in grouped.items()},
        "labels": "synthetic exact boxes and white-bubble flags only",
        "baseline_opencv_test_accuracy": opencv_route_accuracy(grouped["test"]),
        "learned_test": test_metrics,
        "history": history,
    }
    return report
