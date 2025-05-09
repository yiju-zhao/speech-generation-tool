"""
Command line argument parsing for Digital Presenter.
"""

import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Digital Presenter - Generate transcripts and audio for presentations"
    )
    parser.add_argument(
        "--pptx",
        type=str,
        help="Path to PowerPoint file (if not provided, will look in data/raw)",
    )
    parser.add_argument(
        "--skip-audio", action="store_true", help="Skip audio generation"
    )
    parser.add_argument(
        "--direct-audio",
        action="store_true",
        help="Generate audio directly from PowerPoint notes without generating separate transcript file",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["chinese", "english"],
        default="chinese",
        help="Language for transcript generation (default: chinese)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for transcript generation",
    )
    parser.add_argument(
        "--tts-provider",
        type=str,
        choices=["minimax", "openai", "elevenlabs"],
        default="minimax",
        help="Text-to-speech provider to use (default: minimax, which is better for Chinese; openai is better for English)",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["openai", "azure", "deepseek", "gemini"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--use-storm",
        action="store_true",
        help="Use the Storm-enhanced approach for reducing hallucinations",
    )
    parser.add_argument(
        "--disable-search",
        action="store_true",
        help="Disable web searches when using Storm-enhanced approach",
    )
    parser.add_argument('--slides', nargs='+', type=int,
                    help='List of slide numbers to process (1-based index)')
    return parser.parse_args()
