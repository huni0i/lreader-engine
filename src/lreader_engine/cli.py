import argparse

from lreader_engine.fast_ocr import FastOcrEngine
from lreader_engine.mlx_translator import MlxTranslationEngine
from lreader_engine.models import SourceLanguage, TargetLanguage
from lreader_engine.ocr import OcrEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lreader-engine")
    commands = parser.add_subparsers(dest="command", required=True)

    ocr = commands.add_parser("ocr", help="Detect and recognize text in an image")
    ocr.add_argument("image")

    fast_ocr = commands.add_parser(
        "fast-ocr",
        help="Run the low-latency OCR path",
    )
    fast_ocr.add_argument("image")
    fast_ocr.add_argument(
        "--source",
        choices=["ja", "en", "zh", "ko"],
        required=True,
    )

    translate = commands.add_parser("translate", help="Translate plain text")
    translate.add_argument("text")
    translate.add_argument(
        "--from",
        dest="source",
        choices=["ja", "en", "zh", "ko"],
        required=True,
    )
    translate.add_argument(
        "--to",
        choices=["ja", "en", "zh", "ko"],
        default="ko",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "ocr":
        print(OcrEngine().spot(args.image))
        return

    if args.command == "fast-ocr":
        source: SourceLanguage = args.source
        for region in FastOcrEngine(source).recognize(args.image):
            print(region.model_dump_json())
        return

    source: SourceLanguage = args.source
    target: TargetLanguage = args.to
    print(MlxTranslationEngine().translate(args.text, source, target))


if __name__ == "__main__":
    main()
