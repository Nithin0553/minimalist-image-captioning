from __future__ import annotations

import argparse
from pathlib import Path

from minimal_captioning.download import download_flickr8k


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and arrange the Flickr8k dataset")
    parser.add_argument("--target", type=Path, default=Path("data/flickr8k"))
    args = parser.parse_args()
    layout = download_flickr8k(args.target)
    print(f"Images: {layout.images_dir}")
    print(f"Captions: {layout.captions_file}")


if __name__ == "__main__":
    main()
