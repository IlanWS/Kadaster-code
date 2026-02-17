"""CLI entry point."""
import sys

from kadaster_code.download import main as download_main
from kadaster_code.process import main as process_main
from kadaster_code.train import train


def print_usage() -> None:
    """Print usage information."""
    print("Usage: python -m kadaster_code <command>")
    print()
    print("Commands:")
    print("  download  Download images from WMS service")
    print("  process   Process images for training")
    print("  train     Train autoencoder model")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "download":
        download_main()
    elif command == "process":
        process_main()
    elif command == "train":
        train()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
