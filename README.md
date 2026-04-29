## Overview

This repository implements a text-to-MIDI generation system that creates symbolic music from natural language descriptions.

**MSc Applied Artificial Intelligence - ICESI University**

## Installation

This project requires Python 3.12+ and [uv](https://github.com/astral-sh/uv) package manager

```bash
# Clone the repository
git clone https://github.com/NickEsColR/symbolic-music-generation.git
cd symbolic-music-generation

# Install dependencies with uv
uv sync
```

## Usage

### [text2midi training analysis](./notebooks/text2midi_train.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ColRuDev/text2midi/blob/main/notebooks/text2midi_train.ipynb)

1. **Dataset Preparation**: Preprocess SymphonyNet dataset using music21 to extract tempo, key, BPM, and instruments
2. **Pseudo-Caption Generation**: Create template-based captions for pre-training
3. **Model Architecture**:
   - Encoder: FlanT5-base for text processing
   - Tokenizer: REMI+ for MIDI tokenization
   - Decoder: Processes encoded text and tokenized MIDI
4. **Training**: Complete training loop with the prepared datasets

The notebook can be run directly in Google Colab using the badge above.

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

## Academic Context

This project is part of the MSc in Applied Artificial Intelligence program at Universidad ICESI, Colombia. It focuses on symbolic music generation using deep learning techniques to bridge natural language and musical representation.
