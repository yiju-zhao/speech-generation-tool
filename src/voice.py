"""
Voice generation module for Digital Presenter.
"""
import os
import time
import requests
import openai
from pathlib import Path
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from dotenv import load_dotenv

load_dotenv()

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


def generate_audio_with_minimax(transcript: str, slide_number: int, output_dir, api_key: str = None, voice_id: str = "moss_audio_413921f7-327e-11f0-9505-4e9b7ef777f4", group_id: str = "1943724805874258267"):
    """
    Generate audio for a transcript using Minimax TTS API.

    Args:
        transcript (str): The text to convert to speech.
        slide_number (int): A number for naming the output file (e.g., slide number).
        output_dir (str or Path): The directory to save the generated audio file.
        group_id (str): Your Minimax Group ID.
        api_key (str, optional): Your Minimax API key. If None, attempts to load from
                                 the environment variable MINIMAX_API_KEY.
        voice_id (str, optional): The voice ID to use for generation.
                                  Defaults to "moss_audio_413921f7-327e-11f0-9505-4e9b7ef777f4".

    Returns:
        Path or None: The Path object to the generated audio file if successful, else None.
    """
    # Ensure output_dir is a Path object
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

    # Retrieve API key if not provided
    if not api_key:
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            print("Error: MINIMAX_API_KEY not provided and not found in environment variables.")
            return None
    
    if not group_id:
        print("Error: Minimax Group ID (group_id) must be provided.")
        return None

    # Minimax TTS API configuration - using "minimaxi.chat" and "GroupId" from the first example
    api_url = f"https://api.minimaxi.chat/v1/t2a_v2?GroupId={group_id}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Prepare the request payload
    # Merging details from both examples, ensuring "output_format": "hex" and "channel" are included
    payload = {
        "model": "speech-02-hd",
        "text": transcript,
        "stream": False, # As per both examples
        "output_format": "hex", # Crucial for bytes.fromhex(), explicitly in 2nd fn, implied by 1st
        "voice_setting": {
            "voice_id": "Chinese (Mandarin)_Reliable_Executive", # Using the more specific voice_id, can be parameterized
            "speed": 1.1, # Ensuring float for speed, vol, pitch if API expects it
            "vol": 1,
            "pitch": 0,
            "emotion": "neutral" # From the more detailed payload in the original function
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1 # From the first example's audio_setting
        },
        "subtitleEnable": True, # The parameter controls whether the subtitle service is enabled
        # "language_boost": "auto", # This was in the original function's payload.
                                  # Can be added if supported and needed. For now, keeping it closer to the first example's structure.
                                  # If you need it, uncomment the line above and ensure it's placed correctly in the payload.
    }

    print(f"Generating audio for slide {slide_number} using Minimax with voice {voice_id}...")
    print(f"API URL: {api_url}")
    # print(f"Payload: {json.dumps(payload, indent=2)}") # For debugging payload

    try:
        # Make the API request
        # Using requests.post with json=payload is generally preferred
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        # Parse the response
        response_data = response.json()

        # Check if the request was successful based on Minimax's response structure
        if response_data.get("base_resp", {}).get("status_code") == 0:
            # Get the audio data (hex string)
            audio_hex = response_data.get("data", {}).get("audio")

            if audio_hex:
                # Get additional information if available
                extra_info = response_data.get("extra_info", {})
                audio_length_ms = extra_info.get("audio_length", 0) # Assuming it's in ms
                # audio_size_bytes = extra_info.get("audio_size", 0) # Size of hex string, not binary
                word_count = extra_info.get("word_count", 0)

                # Convert hex string to binary audio data
                audio_binary = bytes.fromhex(audio_hex)
                
                output_file = output_dir_path / f"slide_{slide_number}.mp3"

                with open(output_file, "wb") as f:
                    f.write(audio_binary)

                # Get actual binary audio size
                audio_size_bytes_actual = len(audio_binary)

                print(f"Successfully generated audio for slide {slide_number} using Minimax.")
                print(f"  - Output file: {output_file}")
                print(f"  - Audio length: {audio_length_ms}ms")
                print(f"  - Audio size: {audio_size_bytes_actual} bytes")
                print(f"  - Word count: {word_count}")
                return output_file
            else:
                print(f"No audio data (hex string) received for slide {slide_number} from Minimax.")
                print(f"Full response data: {response_data}")
                return None
        else:
            status_msg = response_data.get("base_resp", {}).get("status_msg", "Unknown API error")
            error_code = response_data.get("base_resp", {}).get("status_code", "N/A")
            print(f"Minimax API error for slide {slide_number}: {status_msg} (Code: {error_code})")
            print(f"Full response data: {response_data}")
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while generating audio for slide {slide_number}: {http_err}")
        print(f"Response content: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred while generating audio for slide {slide_number}: {req_err}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response for slide {slide_number}.")
        print(f"Response content: {response.text if 'response' in locals() else 'No response object'}")
    except Exception as e:
        print(f"An unexpected error occurred generating audio with Minimax for slide {slide_number}: {str(e)}")
    
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


def process_pptx_for_audio(pptx_path, output_dir=None, provider="minimax", page_range=None):
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

        if page_range is not None and slide_number not in page_range:
            print(f"Slide {slide_number} is not in the specified range, skipping.")
            continue
        print(f"Generating audio for slide {slide_number}...")

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
