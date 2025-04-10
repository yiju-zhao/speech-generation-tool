"""
Voice generation module for Digital Presenter.
"""

import time
import requests
from pathlib import Path

from .utils import load_config, ensure_directory, get_project_paths, load_json


def generate_audio(transcript, slide_number, output_dir, api_key=None):
    """Generate audio for a transcript using Minimax TTS API."""
    if not api_key:
        config = load_config()
        api_key = config["minimax_api_key"]

    # Minimax TTS API configuration
    api_url = "https://api.minimax.chat/v1/t2a_v2"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # Prepare the request payload
    payload = {
        "model": "speech-02-hd",
        "text": transcript,
        "stream": False,
        "language_boost": "auto",
        "output_format": "hex",
        "voice_setting": {
            "voice_id": "male-qn-qingse",
            "speed": 1,
            "vol": 1,
            "pitch": 0,
            "emotion": "happy",
        },
        "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3"},
    }

    # Make the API request
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the response
        response_data = response.json()

        # Check if the request was successful
        if response_data.get("base_resp", {}).get("status_code") == 0:
            # Get the audio data from the response
            audio_data = response_data.get("data", {}).get("audio")

            if audio_data:
                # Get additional information
                extra_info = response_data.get("extra_info", {})
                audio_length = extra_info.get("audio_length", 0)
                audio_size = extra_info.get("audio_size", 0)
                word_count = extra_info.get("word_count", 0)

                # Convert hex to binary and save to file
                audio_binary = bytes.fromhex(audio_data)
                output_file = output_dir / f"slide_{slide_number}.mp3"

                with open(output_file, "wb") as f:
                    f.write(audio_binary)

                print(f"Generated audio for slide {slide_number}")
                print(f"  - Audio length: {audio_length}ms")
                print(f"  - Audio size: {audio_size} bytes")
                print(f"  - Word count: {word_count}")

                return output_file
            else:
                print(f"No audio data received for slide {slide_number}")
                return None
        else:
            status_msg = response_data.get("base_resp", {}).get(
                "status_msg", "Unknown error"
            )
            print(f"API error for slide {slide_number}: {status_msg}")
            return None

    except Exception as e:
        print(f"Error generating audio for slide {slide_number}: {str(e)}")
        return None


def process_transcripts(transcripts_file, output_dir=None):
    """Process transcripts and generate audio for each slide."""
    # Load the transcripts
    transcripts = load_json(transcripts_file)

    # Set up output directory
    if not output_dir:
        paths = get_project_paths()
        output_dir = paths["output_dir"] / "audio_files"

    ensure_directory(output_dir)

    # Process each transcript
    for slide_data in transcripts:
        slide_number = slide_data["slide_number"]
        transcript = slide_data["transcript"]

        # Generate audio
        generate_audio(transcript, slide_number, output_dir)

        # Add a small delay to avoid rate limiting
        time.sleep(1)

    print("Audio generation complete!")
    return output_dir


def main():
    """Main function to run the voice generator."""
    paths = get_project_paths()
    ensure_directory(paths["output_dir"])

    # Find transcript files in the output directory
    transcript_files = list(paths["output_dir"].glob("*_transcripts.json"))

    if not transcript_files:
        print(f"No transcript files found in {paths['output_dir']}")
        return

    # Process each transcript file
    for transcript_file in transcript_files:
        print(f"Processing {transcript_file.name}...")
        output_dir = paths["output_dir"] / f"{transcript_file.stem}_audio"
        process_transcripts(transcript_file, output_dir)


if __name__ == "__main__":
    main()
