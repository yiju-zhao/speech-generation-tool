#!/usr/bin/env python3
"""
merge_mp3s.py

Merge all MP3 files in a given directory into a single MP3.
Usage: python merge_mp3s.py /path/to/mp3_folder output.mp3
"""

import sys
from glob import glob
from pathlib import Path
from pydub import AudioSegment

def merge_mp3_directory(input_dir: Path, output_file: Path):
    """
    Concatenates all .mp3 files in `input_dir` into one file at `output_file`.
    Files are merged in sorted (alphanumeric) order.
    """
    # Collect all MP3 files
    mp3_paths = sorted(glob(str(input_dir / "*.mp3")))
    if not mp3_paths:
        print(f"No MP3 files found in {input_dir}")
        return

    # Initialize an empty AudioSegment
    combined = AudioSegment.empty()  # type: AudioSegment

    # Append each MP3 to the combined segment
    for mp3 in mp3_paths:
        segment = AudioSegment.from_mp3(mp3)
        combined += segment  # simple concatenation :contentReference[oaicite:2]{index=2}

    # Export the result
    combined.export(output_file, format="mp3")
    print(f"Merged {len(mp3_paths)} files into {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_mp3s.py /path/to/input_dir output.mp3")
        sys.exit(1)

    inp_dir = Path(sys.argv[1])
    out_file = Path(sys.argv[2])
    merge_mp3_directory(inp_dir, out_file)
