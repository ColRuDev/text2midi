## Overview

This repository implements a text-to-MIDI generation system that creates symbolic music from natural language descriptions.

**MSc Applied Artificial Intelligence - ICESI University**

## Academic Context

This project is part of the MSc in Applied Artificial Intelligence program at Universidad ICESI, Colombia. It focuses on symbolic music generation using deep learning techniques to bridge natural language and musical representation.

### Objectives

**Specific Objectives**
- Review the state-of-the-art related to symbolic music generation models from textual descriptions.
- Deepen the understanding and explanation of the training and generation process of the Transformer architecture model, detailing how it transforms text into symbolic music.
- Design a prototype for the conversion of natural language descriptions into symbolic music, evaluating its impact on music generation.

## Installation

This project requires Python 3.12+ and [uv](https://github.com/astral-sh/uv) package manager

```bash
# Clone the repository
git clone https://github.com/NickEsColR/symbolic-music-generation.git
cd symbolic-music-generation

# Install dependencies with uv
uv sync
```

### Environment Variables (.env)

If you plan to use natural language translation via LLM (e.g., Gemini) by passing `--translator-model` in the CLI, you must set up your environment variables:

1. Copy the example `.env` file or create a new one:
```bash
cp .env.example .env
```
2. Add your Google API Key to the `.env` file:
```ini
GOOGLE_API_KEY=your_api_key_here
```

## Hardware Requirements

To run the generation models effectively, please note the following hardware constraints:

- **CPU Evaluation**: The sequence evaluation phase (scoring generated MIDI with musical heuristics) is highly CPU-intensive.
- **VRAM Requirements (MidiLLM)**: The MidiLLM model is the heaviest in the system and requires a dedicated GPU with at least **4GB of VRAM** for stable execution. However, despite being heavier, MidiLLM generates music significantly faster due to its batch processing capabilities.
- **Execution Time (Text2Midi)**: Generation time for the Text2Midi model grows exponentially as the number of branches in beam search increases, because these searches are evaluated sequentially rather than in parallel.

## Usage

> **Note**: All notebooks can be run directly in Google Colab using their respective badges below.

### [text2midi training analysis](./notebooks/text2midi_train.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ColRuDev/text2midi/blob/main/notebooks/text2midi_train.ipynb)

This notebook addresses the first part of **Objective 2** by providing a deep dive into the encoder-decoder architecture and the training process of the model.

The analysis covers:
1. **Dataset Preparation**: Preprocess SymphonyNet dataset using music21 to extract tempo, key, BPM, and instruments
2. **Pseudo-Caption Generation**: Create template-based captions for pre-training
3. **Model Architecture**:
   - Encoder: FlanT5-base for text processing
   - Tokenizer: REMI+ for MIDI tokenization
   - Decoder: Processes encoded text and tokenized MIDI
4. **Training**: Complete training loop with the prepared datasets

### [text2midi generation exploration](./notebooks/text2midi_poc.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ColRuDev/text2midi/blob/main/notebooks/text2midi_poc.ipynb)

This notebook addresses the second part of **Objective 2** by focusing on the generation process. It explores how to use the pre-trained text2midi model to generate symbolic music from natural language prompts.

The analysis covers:
1. **Environment Setup & Model Loading**: Preparing the test lab and configuring the Hugging Face model for internal data extraction.
2. **Initial Processing**: Tokenization and Embeddings, observing how text translates to a latent space.
3. **Encoder Journey**: Analyzing self-attention and contextual understanding of the textual prompt.
4. **Decoder & MIDI Generation**: Deconstructing the autoregressive mechanism and cross-attention step-by-step.
5. **Raw Output & Final Reconstruction**: Decoding model logits back into playable MIDI files using REMI+.

### [Generation Workflows Evaluation](./notebooks/evaluacion_flujos.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ColRuDev/text2midi/blob/main/notebooks/evaluacion_flujos.ipynb)

This notebook addresses **Objective 3** by evaluating the complete MIDI generation pipeline from natural language descriptions. It compares two distinct generation strategies and assesses their impact on the final musical output.

The analysis covers:
1. **Pipeline & Environment Setup**: Configuration of translators and necessary dependencies to run the end-to-end flow.
2. **Base Prompt Definition**: Establishment of the natural language request ("Una melodía de piano melancólica...") to be processed.
3. **Flow 1: Text2Midi (Progressive Search)**: Generation using the `one-shot` profile to evaluate progressive search capabilities.
4. **Flow 2: MidiLLM (Best-of-N)**: Generation using the `midillm-fast` profile to evaluate batch generation and selection.
5. **Conclusions & Review**: Comparative analysis of the resulting technical prompts, instrumentation adherence, tempo biases, and harmonic progression accuracy.

### Code Versioning using jupytext

The project use jupytext to create jupyter cells using a .py file.

To syncronized the .ipynb with the .py run

```bash
uv run jupytext --set-formats ipynb,py:percent notebooks/notebook.ipynb
```

To syncronized the .py with the .ipynb run

```bash
uv run jupytext --sync notebooks/notebook.py
```

## Implementation

### [Source Code Architecture](./docs/implementation.md)

This document addresses the structural design of the project, focusing on how the MIDI generation pipeline is built for maintainability and extensibility.

The overview covers:
1. **Domain Layer**: Core business entities and interfaces (`src/domain/`).
2. **Use Cases Layer**: Orchestration logic for `ProgressiveSearch` and `BestOfNSearch` (`src/use_cases/`).
3. **Adapters Layer**: Integrations with external models, translators, and evaluators (`src/adapters/`).
4. **Configuration & Models**: Predefined profiles and neural network definitions.
5. **Orchestration**: The Dependency Injection pipeline that wires the architecture together (`pipeline.py`).

### [Command Line Interface (CLI)](./docs/cli_reference.md)

The project includes a robust CLI to generate MIDI files directly from the terminal. 

**Basic Usage:**
```bash
python -m src.cli --text "A peaceful piano melody" --output peaceful.mid
```

For a comprehensive guide on all available arguments (like `--profile`, `--translator-model`, and `--strict-instruments`) and advanced usage examples, refer to the [CLI Reference](./docs/cli_reference.md).

## Developer Setup

### Running Tests

The project includes a comprehensive test suite covering the domain, use cases, and adapters. To run the tests, use `pytest` via `uv`:

```bash
uv run pytest tests/
```

### Local Models Directory

When working locally, you should place downloaded model weights and vocabularies (e.g., the pre-trained `text2midi` model from Hugging Face) inside a `models/` directory at the root of the project.

```bash
mkdir -p models/
# Download hugging face model bin and vocab to this folder
```

## Datasets

### SymphonyNet
- **Size**: 46,359 MIDI files
- **Duration**: 3,284 hours of music
- **Content**: Symphonic music with multiple instrument tracks
- **Reference**: [SymphonyNet Dataset](https://symphonynet.github.io/)

### MidiCaps
- **Features**: MIDI files with rich text captions
- **Metadata**: Genre, mood, tempo, key, time signature, chord progressions, instrumentation
- **Split**: 90/10 train/test partition
- **Reference**: [MidiCaps on HuggingFace](https://huggingface.co/datasets/amaai-lab/MidiCaps)

## References

- **Text2midi Paper**: [Text2midi: Generating Symbolic Music from Captions](http://arxiv.org/abs/2412.16526)
- **Pre-trained Model**: [text2midi on HuggingFace](https://huggingface.co/amaai-lab/text2midi)
- **SymphonyNet Dataset**: [https://symphonynet.github.io/](https://symphonynet.github.io/)
- **MidiCaps Dataset**: [https://huggingface.co/datasets/amaai-lab/MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps)
