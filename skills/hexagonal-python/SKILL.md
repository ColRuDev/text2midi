---
name: hexagonal-python
description: "Trigger: Python architecture, hexagonal, clean architecture, ports adapters, domain layer, new module, new adapter. Enforce Clean/Hexagonal Architecture separation in Python projects."
license: Apache-2.0
metadata:
  author: gentleman-programming
  version: "1.0"
---

# Skill: Hexagonal Python Architecture

Enforce Clean/Hexagonal Architecture (Ports & Adapters) in Python projects.

## Activation Contract

Activate when:
- Creating new modules, adapters, or use cases
- Refactoring code between layers
- Writing tests that mock external dependencies
- Reviewing code for architectural compliance

## Hard Rules

| Layer | Allowed | Forbidden |
|-------|---------|-----------|
| `domain/` | dataclasses, abc.ABC, typing, stdlib | PyTorch, APIs, I/O, external libs |
| `use_cases/` | domain interfaces, dependency injection | direct adapter imports, PyTorch, APIs |
| `adapters/` | everything (PyTorch, APIs, I/O, domain) | business logic, mutating domain entities |

### Layer Separation

1. **domain/**: Pure business rules. NO external dependencies.
   - Entities: `@dataclass` for data containers
   - Interfaces: `abc.ABC` with `@abstractmethod`
   - No imports from `adapters/` or `use_cases/`

2. **use_cases/**: Orchestration logic only.
   - Import domain interfaces, NOT implementations
   - Receive adapters via dependency injection (constructor)
   - No PyTorch tensors, no API calls, no file I/O

3. **adapters/**: External world implementation.
   - Implement domain interfaces
   - All PyTorch, API calls, file I/O goes here
   - Wrap third-party libs, don't leak them to use_cases

### Dependency Injection

```python
# use_cases/generate.py
class GenerateMidiUseCase:
    def __init__(
        self,
        translator: LLMTranslator,      # interface from domain
        generator: MidiGenerator,       # interface from domain
        evaluator: Evaluator,           # interface from domain
    ):
        self._translator = translator
        self._generator = generator
        self._evaluator = evaluator
```

### Testability Contract

- Every adapter MUST be mockable
- Use `unittest.mock.Mock` or `pytest-mock` in tests
- Tests for use_cases MUST NOT load models or call APIs
- GPU-dependent code stays in adapters, mocked in unit tests

## Decision Gates

| Need | Layer | Example |
|------|-------|---------|
| New data structure | domain/ | `@dataclass class Intent` |
| New contract/interface | domain/ | `class LLMTranslator(ABC)` |
| Business orchestration | use_cases/ | `class SearchUseCase` |
| PyTorch model wrapper | adapters/ | `class Text2MidiGenerator` |
| API client (OpenAI, Anthropic) | adapters/ | `class ClaudeTranslator` |
| File I/O, audio conversion | adapters/ | `class FluidSynthAdapter` |

## Output Contract

When creating or reviewing code:

1. Verify no forbidden imports in each layer
2. Confirm interfaces are in domain/
3. Confirm implementations are in adapters/
4. Confirm use_cases only use interfaces
5. Confirm dependency injection is used

## References

- `ARCHITECTURE.md` — Full architectural decision document with trade-offs
