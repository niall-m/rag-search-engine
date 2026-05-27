import argparse

from lib.hybrid_search import normalize_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize some numbers")
    normalize_parser.add_argument("nums", type=float, nargs="*", help="Pass a list of numbers to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.nums)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
