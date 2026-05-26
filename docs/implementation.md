# Source Code Implementation Details

The `src/` directory is structured following **Clean Architecture** and **Hexagonal Architecture** principles. This design ensures that the core domain logic remains independent of external frameworks, heavy machine learning adapters, and UI details, enabling high maintainability and testability.

## Directory Structure Overview

### 1. Domain Layer (`src/domain/`)
The core of the application, defining business entities and ports (interfaces).
- **`entities.py`**: Defines core data structures like `GenerationProfile`, `GenerationResult`, `Intent`, and `MidiBytes`.
- **`interfaces.py`**: Declares abstract ports (e.g., `MidiGenerator`, `BatchMidiGenerator`, `Evaluator`) that adapters must implement.
- **`remi_vocab.py`**: Definitions for MIDI tokenization vocabularies.

### 2. Use Cases Layer (`src/use_cases/`)
Contains the application-specific business rules and orchestration logic.
- **`progressive_search.py`**: Orchestrates the *Text2Midi* generation strategy using a step-by-step progressive evaluation.
- **`best_of_n_search.py`**: Orchestrates the *MidiLLM* generation strategy using batch generation and Best-of-N scoring.
- **`token_heuristics.py`**: Defines heuristic-based evaluations for fast sequence scoring.

### 3. Adapters Layer (`src/adapters/`)
Contains the concrete implementations of the interfaces defined in the domain layer, bridging the gap to external systems and libraries.
- **`generators/`**: Specific integrations with generation models (`text2midi_generator.py` and `midillm_generator.py`).
- **`evaluators/`**: Implementations for sequence scoring (`clap_evaluator.py`, `composite.py`).
- **`translators/`**: NLP translation implementations to convert natural language to technical prompts (`google_ai_translator.py`, `pass_through_translator.py`).
- **`audio/`**: Audio rendering engines to transform symbolic MIDI into waveforms for evaluation (`fluidsynth_memory.py`).

### 4. Configuration & Models (`src/config/`, `src/models/`)
- **`profiles.py`**: Defines predefined generation profiles (`BALANCED`, `MIDILLM_FAST`, etc.) to easily switch between strategies.
- **`transformer_model.py`**: Contains the definition of the underlying neural network architectures (like the Transformer Encoder-Decoder).

### 5. Orchestration (`src/pipeline.py` & `src/cli.py`)
- **`pipeline.py`**: The main entry point acting as a **Dependency Injection Container**. It instantiates heavy adapters once, prevents GPU memory leaks, and selects the correct search strategy based on the profile.
- **`cli.py`**: The Command Line Interface exposing the pipeline features to end-users.

## Architectural Data Flow

1. **Input**: The user requests a generation through `cli.py`.
2. **Setup**: `pipeline.py` initializes the heavy models/adapters and injects them into the chosen use case strategy.
3. **Execution**: Based on the `GenerationProfile`, either `ProgressiveSearch` or `BestOfNSearch` takes control.
4. **Processing**: 
   - Translators transform natural language into structured technical prompts.
   - Generators create raw token sequences representing MIDI events.
   - Evaluators compute scores using heuristics or ML models (e.g., CLAP) to select the best output.
5. **Output**: The optimal `GenerationResult` (containing the final MIDI bytes) is returned and saved to disk.
