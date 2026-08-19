from lreader_engine.ocr import spotting_input_size


BUDGET = 2048 * 28 * 28


def test_small_pages_are_upscaled_within_the_pixel_budget() -> None:
    width, height = spotting_input_size(719, 1024, BUDGET)

    assert width > 719 and height > 1024
    assert width * height <= BUDGET


def test_large_pages_are_downscaled_to_the_pixel_budget() -> None:
    width, height = spotting_input_size(2000, 3000, BUDGET)

    assert width * height <= BUDGET


def test_resized_pages_keep_their_aspect_ratio() -> None:
    width, height = spotting_input_size(719, 1024, BUDGET)

    assert abs(width / height - 719 / 1024) < 0.01
