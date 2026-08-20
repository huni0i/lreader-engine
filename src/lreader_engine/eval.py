from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float
    text: str | None = None

    @property
    def area(self) -> float:
        return max(0.0, self.right - self.left) * max(0.0, self.bottom - self.top)

    def iou(self, other: Box) -> float:
        overlap_left = max(self.left, other.left)
        overlap_top = max(self.top, other.top)
        overlap_right = min(self.right, other.right)
        overlap_bottom = min(self.bottom, other.bottom)
        overlap = max(0.0, overlap_right - overlap_left) * max(
            0.0, overlap_bottom - overlap_top
        )
        union = self.area + other.area - overlap
        if union <= 0:
            return 0.0
        return overlap / union


@dataclass(frozen=True)
class EvalPage:
    id: str
    image_path: str
    language: str
    split: str
    source: str
    boxes: list[Box]
    expect_white_bubbles: bool | None = None


@dataclass(frozen=True)
class DetectionMetrics:
    predicted: int
    ground_truth: int
    matches: int
    precision: float
    recall: float
    mean_iou: float


def box_from_polygon(points: list[dict[str, float]], text: str | None = None) -> Box:
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return Box(left=min(xs), top=min(ys), right=max(xs), bottom=max(ys), text=text)


def match_boxes(
    predicted: list[Box],
    ground_truth: list[Box],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    pairs: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predicted):
        for gold_index, gold in enumerate(ground_truth):
            score = pred.iou(gold)
            if score >= iou_threshold:
                pairs.append((score, pred_index, gold_index))
    pairs.sort(reverse=True)

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matched_ious: list[float] = []
    for score, pred_index, gold_index in pairs:
        if pred_index in used_pred or gold_index in used_gold:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        matched_ious.append(score)

    predicted_count = len(predicted)
    gold_count = len(ground_truth)
    matches = len(matched_ious)
    return DetectionMetrics(
        predicted=predicted_count,
        ground_truth=gold_count,
        matches=matches,
        precision=matches / predicted_count if predicted_count else 1.0,
        recall=matches / gold_count if gold_count else 1.0,
        mean_iou=sum(matched_ious) / matches if matches else 0.0,
    )


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def character_error_rate(predicted: str, gold: str) -> float:
    if not gold:
        return 0.0 if not predicted else 1.0
    return levenshtein(predicted, gold) / len(gold)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
