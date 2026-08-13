from lreader_engine.fast_ocr import FastOcrEngine
from lreader_engine.models import OcrRegion, Point


class FakeReader:
    def readtext(self, *args, **kwargs):
        del args, kwargs
        return [
            (
                [[-4.5, -12.0], [40.0, -2.0], [40.0, 30.0], [-1.0, 30.0]],
                "text",
                0.9,
            )
        ]


def test_recognize_clamps_detector_coordinates_to_image_bounds() -> None:
    engine = FastOcrEngine("en")
    engine.__dict__["reader"] = FakeReader()

    region = engine.recognize("unused.png")[0]

    assert all(point.x >= 0 and point.y >= 0 for point in region.polygon)


def test_merge_block_orders_vertical_columns_right_to_left() -> None:
    right = OcrRegion(
        polygon=[
            Point(x=60, y=10),
            Point(x=80, y=10),
            Point(x=80, y=90),
            Point(x=60, y=90),
        ],
        text="右",
        confidence=0.9,
    )
    left = OcrRegion(
        polygon=[
            Point(x=30, y=10),
            Point(x=50, y=10),
            Point(x=50, y=90),
            Point(x=30, y=90),
        ],
        text="左",
        confidence=0.9,
    )

    assert FastOcrEngine._merge_block([left, right]).text == "右左"
