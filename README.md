# EchoPersona Transcript Generator

EchoPersona is a powerful transcript generation tool for PowerPoint presentations. It extracts content from slides and generates high-quality, natural-sounding speech transcripts that can be used for Text-to-Speech (TTS) applications.

## Features

- **Content Extraction**: Automatically extracts text from slides, including tables and text frames
- **Context-Aware Generation**: Maintains context between slides for coherent narration
- **Technical Precision**: Preserves exact terminology and numerical data
- **Hallucination Reduction**: Optional Storm-enhanced workflow for fact verification
- **Web Search Integration**: Uses Tavily API to verify facts from slide content
- **Knowledge Base RAG**: Leverages local knowledge bases for domain-specific accuracy
- **Multilingual Support**: Generate transcripts in multiple languages
- **PowerPoint Integration**: Saves transcripts directly to PowerPoint notes sections
- **Audio Generation**: Converts transcripts to speech using various TTS providers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EchoPersona.git
cd EchoPersona

# Install dependencies
pip install -r requirements.txt

# Optional: Install Tavily for web search capabilities
pip install tavily-python

# Optional: Install sentence-transformers for knowledge base search
pip install sentence-transformers
```

## Configuration

Create a `config.toml` file in the `config` directory with your API keys:

```toml
openai_api_key = "your-openai-api-key"
minimax_api_key = "your-minimax-api-key"
deepseek_api_key = "your-deepseek-api-key"
tavily_api_key = "your-tavily-api-key"   # Optional, for web search verification
knowledge_base_dir = "data/knowledge_base"  # Optional, for knowledge base features
```

## Usage

### Quick Start with run.py

The easiest way to use EchoPersona is through the `run.py` script, which provides a complete pipeline from transcription to audio generation:

```bash
# Standard transcript generation
python run.py

# Using Storm-enhanced approach for better accuracy
python run.py --use-storm

# Using Storm with web search for fact verification
python run.py --use-storm

# Using Storm with knowledge base for accuracy
python run.py --use-storm --knowledge-base data/knowledge_base

# Using knowledge base without web search
python run.py --use-storm --disable-search --knowledge-base data/knowledge_base

# Specify language and model
python run.py --language english --model gpt-4o

# Skip audio generation
python run.py --skip-audio

# Generate audio from an existing noted presentation
python run.py --direct-audio --pptx /path/to/noted_presentation.pptx
```

### Using Individual Modules

You can also use the individual modules directly for more control:

1. Place your PowerPoint files in the `input` directory
2. Run the transcript generator:

```bash
# Standard transcript generation
python -m src.transcript

# Storm-enhanced approach for improved factual accuracy
python -m src.transcript --use-storm

# Storm-enhanced approach with knowledge base
python -m src.transcript --use-storm --knowledge-base data/knowledge_base

# Specify language
python -m src.transcript --language french

# Specify model
python -m src.transcript --model gpt-4
```

3. Find the transcribed PowerPoint files in the `output` directory

### Creating a Knowledge Base

Use the provided tool to create a knowledge base from your documents:

```bash
# Create knowledge base from documents
python tools/create_knowledge_base.py --input /path/to/your/documents --output-dir data/knowledge_base

# Merge multiple knowledge bases
python tools/create_knowledge_base.py --input data/knowledge_base --output-dir data/knowledge_base --merge
```

For more details, see the [Knowledge Base Documentation](docs/knowledge_base.md).

## How It Works

### Standard Workflow

1. **Content Extraction**: Extracts text from slides and any existing notes
2. **Transcript Generation**: Uses language models to create natural-sounding speech
3. **Output**: Saves transcripts to the notes section of the PowerPoint file
4. **Optional Audio Generation**: Converts transcripts to speech using TTS providers

### Storm-Enhanced Workflow

The Storm-enhanced workflow adds several steps to reduce hallucinations and improve accuracy:

1. **Knowledge Curation**: 
   - Generates search queries for key facts
   - Retrieves information from knowledge base (if enabled)
   - Performs web searches using Tavily API (if enabled)
   - Extracts verifiable facts from slide content and retrieved information
   - Synthesizes verified content

2. **Transcript Generation**:
   - Uses only verified content to create transcripts
   - Maintains context between slides

3. **Transcript Review**:
   - Checks transcripts against verified facts
   - Identifies and corrects any hallucinations
   - Produces final verified transcripts

4. **Detailed Logging**:
   - Saves full process data for review
   - Tracks verification steps for each slide
   - Stores relevant knowledge and search results

5. **Optional Audio Generation**:
   - Converts verified transcripts to speech

## Knowledge Base Feature

The knowledge base feature enhances transcript generation with domain-specific information:

1. **Local Knowledge Retrieval**: Uses a local knowledge base with information related to your presentation
2. **Semantic Search**: Finds the most relevant information for each slide
3. **Enhanced Verification**: Cross-references slide content with knowledge base to improve accuracy
4. **Complement to Web Search**: Can be used with or without web search

Benefits of using a knowledge base:
- Works offline without requiring internet access
- Provides domain-specific knowledge that might not be available online
- Ensures consistency in terminology and facts across presentations
- Gives you control over the information sources used

For more details, see the [Knowledge Base Documentation](docs/knowledge_base.md).

## Web Search Integration

The system can use the Tavily API for web search to verify facts in your slides:

1. **Search Query Generation**: For each slide, the system generates relevant search queries based on the content
2. **Web Search**: Uses Tavily API to search the web for related information
3. **Fact Extraction**: Cross-references slide content with search results to verify facts
4. **Fact-Based Content**: Creates transcripts using only verified information

To use this feature:
1. Add your Tavily API key to the config.toml file
2. Run with the `--use-storm` flag (search is enabled by default)
3. To disable web search, use the `--disable-search` flag

## Command-Line Options

### run.py Options

- `--pptx PATH`: Path to PowerPoint file (if not provided, will look in data/raw)
- `--use-storm`: Enable Storm-enhanced approach for reducing hallucinations
- `--disable-search`: Disable web searches for fact verification (only with --use-storm)
- `--knowledge-base DIR`: Directory containing knowledge base for RAG-enhanced generation
- `--language {chinese,english}`: Language for transcript generation (default: chinese)
- `--model {deepseek-chat,deepseek-reasoner,gpt-4o,gpt-4o-mini,gemini-2.0-flash,o3-mini}`: Model to use for generation
- `--skip-audio`: Skip audio generation step
- `--direct-audio`: Generate audio directly from PowerPoint notes
- `--tts-provider {minimax,openai}`: Text-to-speech provider to use

### transcript.py Options

- `--use-storm`: Enable Storm-enhanced approach
- `--disable-search`: Disable web searches for fact verification
- `--knowledge-base DIR`: Directory containing knowledge base files
- `--language LANGUAGE`: Target language for transcripts
- `--model MODEL`: LLM model to use for generation

## Customization

You can customize the behavior by adjusting the configuration in `config.toml`:

- LLM settings (model, temperature, etc.)
- Path configurations
- Default language settings
- TTS provider settings
- API keys for various services
- Knowledge base directory

## Requirements

- Python 3.8+
- python-pptx
- openai
- tavily-python (optional, for web search)
- sentence-transformers (optional, for knowledge base search)
- Other dependencies in requirements.txt

## License

MIT License