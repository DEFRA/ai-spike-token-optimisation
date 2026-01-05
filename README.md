# LLM Prompt Compression Experiment

This project compares different methods of compressing LLM prompts to reduce token counts while maintaining task performance.

## Why Compress Prompts?

Reducing input tokens provides multiple benefits:

- **💰 Cost Savings** - API providers charge per token. Fewer input tokens = lower costs, especially important for high-volume applications.
- **🌱 Environmental Impact** - Each token processed requires computation. Fewer tokens mean less energy consumption in data centers, reducing the carbon footprint of AI applications.
- **⚡ Speed** - Shorter prompts process faster, reducing latency and improving user experience.
- **📊 Efficiency** - More efficient prompts allow you to fit more context within token limits or process more requests with the same budget.

## Methods Compared

1. **No Compression (Baseline)** - Original prompts sent as-is
2. **Caveman Compression** - Drops unimportant words (articles, auxiliary verbs, etc.)
3. **LLMLingua-2** - Advanced selective compression using trained models

## Installation

### Quick Setup (Recommended)

```bash
# Run the setup script
./setup_venv.sh
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure .env file
cp .env.example .env
# Edit .env with your actual API keys
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```bash
# Choose one or both:
OPENAI_API_KEY=sk-your-actual-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here
```

## Usage

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the example demos
python example_usage.py

# Run the full experiment
python experiment.py
```

## Project Structure

- `llm_client.py` - Base LLM client with token counting
- `compressors.py` - Compression methods (Caveman, LLMLingua-2)
- `experiment.py` - Experiment runner comparing methods
- `example_usage.py` - Example usage demonstrations
- `.env` - API keys and configuration (create from .env.example)
