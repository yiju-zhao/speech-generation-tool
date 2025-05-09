"""
Voice generation module for Digital Presenter.
"""

import time
import requests
import openai
from pathlib import Path
from elevenlabs.client import ElevenLabs
from elevenlabs import play
elevenlabs_available = True


from .utils import (
    load_config,
    ensure_directory,
    get_project_paths,
    extract_transcripts_from_pptx,
)


def generate_audio_with_openai(transcript, slide_number, output_dir, api_key=None):
    """Generate audio for a transcript using OpenAI TTS API."""
    if not api_key:
        config = load_config()
        api_key = config["openai_api_key"]

    try:
        print(f"Generating audio for slide {slide_number} using OpenAI TTS...")
        client = openai.OpenAI(api_key=api_key)

        # Create output file path
        output_file = output_dir / f"slide_{slide_number}.mp3"

        # Call the OpenAI TTS API without streaming
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts", voice="alloy", input=transcript
        )

        # Save the audio file
        with open(output_file, "wb") as f:
            f.write(response.content)

        print(f"Successfully generated audio for slide {slide_number} using OpenAI TTS")
        return output_file

    except Exception as e:
        print(f"Error generating audio with OpenAI for slide {slide_number}: {str(e)}")
        return None


def generate_audio_with_minimax(transcript, slide_number, output_dir, api_key=None):
    """Generate audio for a transcript using Minimax TTS API."""
    if not api_key:
        config = load_config()
        api_key = config["minimax_api_key"]

    # Minimax TTS API configuration
    api_url = "https://api.minimax.chat/v1/t2a_v2"

    # According to Minimax documentation, the Authorization header should be "Bearer {api_key}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # Prepare the request payload exactly as specified in the documentation
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
        print(f"Generating audio for slide {slide_number} using Minimax...")
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

                print(
                    f"Successfully generated audio for slide {slide_number} using Minimax"
                )
                print(f"  - Audio length: {audio_length}ms")
                print(f"  - Audio size: {audio_size} bytes")
                print(f"  - Word count: {word_count}")

                return output_file
            else:
                print(f"No audio data received for slide {slide_number} from Minimax")
                return None
        else:
            status_msg = response_data.get("base_resp", {}).get(
                "status_msg", "Unknown error"
            )
            print(f"API error for slide {slide_number}: {status_msg}")
            return None

    except Exception as e:
        print(f"Error generating audio with Minimax for slide {slide_number}: {str(e)}")
        return None


def generate_audio_with_elevenlabs(transcript, slide_number, output_dir, api_key=None, voice_id="PZ3PfumfdpqvMhDFh6ea", model_id="eleven_multilingual_v2", output_format="mp3_44100_128"):
    """Generate audio for a transcript using ElevenLabs TTS API."""

    if not api_key:
        config = load_config()
        api_key = config.get("elevenlabs_api_key")
    if not api_key:
        print("No ElevenLabs API key found. Please set it in your config.toml.")
        return None

    try:
        print(f"Generating audio for slide {slide_number} using ElevenLabs...")
        client = ElevenLabs(api_key=api_key)
        audio = client.text_to_speech.convert(
            text=transcript,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )
        output_file = output_dir / f"slide_{slide_number}.mp3"
        with open(output_file, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        print(f"Successfully generated audio for slide {slide_number} using ElevenLabs")
        return output_file
    except Exception as e:
        print(f"Error generating audio with ElevenLabs for slide {slide_number}: {str(e)}")
        return None


def generate_audio(
    transcript, slide_number, output_dir, provider="minimax", api_key=None
):
    """
    Generate audio for a transcript using the specified provider.

    Args:
        transcript (str): The transcript to convert to audio
        slide_number (int): The slide number
        output_dir (Path): Directory to save the audio file
        provider (str): The TTS provider to use ('minimax', 'openai', or 'elevenlabs')
        api_key (str, optional): API key for the provider

    Returns:
        Path: Path to the generated audio file, or None if generation failed
    """
    provider = provider.lower()
    if provider == "openai":
        return generate_audio_with_openai(transcript, slide_number, output_dir, api_key)
    elif provider == "elevenlabs":
        return generate_audio_with_elevenlabs(transcript, slide_number, output_dir, api_key)
    else:  # Default to minimax
        return generate_audio_with_minimax(
            transcript, slide_number, output_dir, api_key
        )


def process_pptx_for_audio(pptx_path, output_dir=None, provider="minimax"):
    """
    Extract transcripts from a PowerPoint file and generate audio for each slide.

    Args:
        pptx_path (Path or str): Path to the PowerPoint file
        output_dir (Path or str, optional): Directory to save the audio files
        provider (str): The TTS provider to use ('minimax', 'openai', or 'elevenlabs')

    Returns:
        Path: Path to the directory containing the generated audio files
    """
    pptx_path = Path(pptx_path)
    paths = get_project_paths()
    audio_base_dir = paths["audio_dir"]
    pptx_stem = pptx_path.stem
    # Set up output directory: data/audio/<pptx_stem>
    if not output_dir:
        output_dir = Path(audio_base_dir) / pptx_stem
    else:
        output_dir = Path(output_dir)

    ensure_directory(output_dir)

    # Extract transcripts from PowerPoint
    print(f"Extracting transcripts from {pptx_path.name}...")
    transcripts = extract_transcripts_from_pptx(pptx_path)

    # Check if we have any transcripts
    valid_transcripts = [t for t in transcripts if t["transcript"]]
    if not valid_transcripts:
        print(f"No transcripts found in the notes section of {pptx_path.name}")
        print(
            "This might be because the PPTX file doesn't have transcripts in the notes section."
        )
        print(
            "Please run the transcript generation process first to add transcripts to the PPTX."
        )
        return None

    print(f"Found {len(valid_transcripts)} slide(s) with transcripts")
    print(f"Audio files will be saved to: {output_dir}")

    # Process each transcript
    for slide_data in transcripts:
        slide_number = slide_data["slide_number"]
        transcript = slide_data["transcript"]

        if not transcript:
            print(f"No transcript available for slide {slide_number}, skipping.")
            continue

        # Generate audio with the specified provider
        generate_audio(transcript, slide_number, output_dir, provider)

        # Add a small delay to avoid rate limiting
        time.sleep(1)

    print("Audio generation complete!")
    return output_dir


def main():
    """Main function to run the voice generator."""
    paths = get_project_paths()
    ensure_directory(paths["processed_dir"])

    # Ask user which TTS provider to use
    print("\nWhich Text-to-Speech provider would you like to use?")
    print("1. Minimax (default, better for Chinese)")
    print("2. OpenAI (better for English)")
    print("3. ElevenLabs (high quality, supports many languages)")

    provider_choice = input("Enter your choice (1, 2, or 3): ").strip()

    # Set the provider based on user choice
    if provider_choice == "2":
        provider = "openai"
        print("Using OpenAI TTS")
    elif provider_choice == "3":
        provider = "elevenlabs"
        print("Using ElevenLabs TTS")
    else:
        provider = "minimax"
        print("Using Minimax TTS")

    # Look for PowerPoint files in the processed directory
    processed_dir = paths["processed_dir"]
    pptx_files = list(processed_dir.glob("*_noted.pptx"))

    if not pptx_files:
        print(f"No PowerPoint files with transcripts found in {processed_dir}")
        return

    # Process each PowerPoint file
    for pptx_file in pptx_files:
        print(f"Processing {pptx_file.name}...")
        print(
            f"Audio files will be saved to the same directory as the PPTX file: {pptx_file.parent}"
        )
        process_pptx_for_audio(pptx_file, None, provider)


if __name__ == "__main__":
    main()
