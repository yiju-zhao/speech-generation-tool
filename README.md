# Digital Presenter

A tool for generating presentation transcripts and audio narration from PowerPoint slides.

## Features

- Extract text from PowerPoint slides
- Generate natural, conversational transcripts using LLM
- Convert transcripts to audio narration
- Create a complete digital presentation experience

## Project Structure

```
digital_presenter/
├── config/                  # Configuration files
│   └── .env                 # Environment variables
├── data/                    # Data files
│   ├── input/               # Input files (e.g., PowerPoint presentations)
│   └── output/              # Output files (transcripts, audio)
├── src/                     # Source code
│   ├── __init__.py
│   ├── transcript.py        # Transcript generation module
│   ├── voice.py             # Voice generation module
│   └── utils.py             # Utility functions
├── notebooks/               # Jupyter notebooks
│   └── demo.ipynb           # Demo notebook
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the `config` directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   MINIMAX_API_KEY=your_minimax_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ```

## Usage

### Running the Complete Pipeline

The easiest way to use Digital Presenter is through the `run.py` script:

```bash
# Run the complete pipeline (will use PowerPoint files from data/input/)
python run.py

# Run with a specific PowerPoint file
python run.py --pptx path/to/presentation.pptx

# Skip transcript generation (use existing transcripts)
python run.py --skip-transcript

# Skip audio generation
python run.py --skip-audio
```

### Running Individual Components

You can also run the transcript and voice generation steps separately:

1. Place your PowerPoint presentation in the `data/input` directory
2. Run the transcript generator:
   ```
   python -m src.transcript
   ```
3. Run the voice generator:
   ```
   python -m src.voice
   ```
4. Check the generated files in the `data/output` directory:
   - Transcripts are saved as JSON files
   - Audio files are saved as MP3 files in the `audio_files` directory