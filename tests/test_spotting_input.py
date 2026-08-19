from lreader_engine.ocr import spotting_input_size


BUDGET = 2048 * 28 * 28


def test_comic_pages_keep_their_original_size_when_under_budget() -> None:
    assert spotting_input_size(719, 1024, BUDGET) == (719, 1024)


def test_large_pages_are_downscaled_to_the_pixel_budget() -> None:
    width, height = spotting_input_size(2000, 3000, BUDGET)

    assert width * height <= BUDGET


def test_pages_under_budget_keep_their_aspect_ratio() -> None:
    width, height = spotting_input_size(719, 1024, BUDGET)

    assert (width, height) == (719, 1024)
