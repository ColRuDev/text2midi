# Arquitectura del Proyecto: (Pipeline de Generación Guiado por Intenciones)

Este documento detalla la decisión arquitectónica para la implementación del pipeline de inferencia alineada (traducción de intenciones a MIDI mediante búsqueda guiada por recompensas), descartando frameworks de alto nivel en favor de una Arquitectura Limpia (Clean Architecture / Hexagonal) en Python puro.

## ¿Por qué usar Python Puro con Arquitectura Limpia?

Cuando se construyen flujos que involucran LLMs, el instinto inicial suele ser adoptar frameworks como LangChain o LlamaIndex. Sin embargo, para este caso de uso específico, dicha decisión representa un error arquitectónico grave.

Lo que estamos construyendo **no es un simple chatbot ni un sistema RAG**, sino un **bucle de inferencia de Machine Learning altamente personalizado** (un *Reward-Guided Beam Search*) que mezcla procesamiento de lenguaje natural, tensores de PyTorch, síntesis de audio en memoria y heurísticas musicales complejas.

Las razones fundamentales para elegir una Arquitectura Limpia en Python puro son:

### 1. Eliminación de "Magia" y Acoplamiento Innecesario

Frameworks como LangChain añaden capas de abstracción opacas y cambian sus APIs frecuentemente. Acoplar un pipeline de inferencia a estas librerías dificulta el debugging. El día que ocurra un cuello de botella en la memoria de la GPU o un error de renderizado en un nodo específico del *beam search*, rastrear el error a través de las abstracciones del framework será una pesadilla. Es preferible usar los SDKs oficiales (`openai`, `anthropic`) directamente.

### 2. Testabilidad Absoluta (Sin quemar la GPU)

En un script monolítico, probar si la lógica del bucle funciona requiere cargar modelos de gigabytes en la GPU y esperar minutos por cada test. Con Arquitectura Limpia, inyectamos *Mocks* (dobles de prueba) en los tests unitarios:

- Un traductor falso que responde instantáneamente.
- Un generador falso que devuelve arrays de NumPy aleatorios.
- Un evaluador falso que devuelve puntajes fijos.
Esto permite probar la lógica central (poda de ramas, selección, iteración) en milisegundos dentro de un entorno CI/CD.

### 3. Auditabilidad y Observabilidad

Al separar el bucle principal (Caso de Uso) de las implementaciones, podemos inyectar un sistema de *Logging Estructurado* (ej. `structlog`). En lugar de `print()` sueltos, emitimos eventos JSON con telemetría exacta por cada iteración, rama evaluada y recompensas asignadas. Esto permite auditar exactamente **por qué** el modelo eligió una rama sobre otra y enviar estos datos a plataformas de monitoreo.

### 4. Intercambiabilidad (Pluggability) y Trade-offs (Calidad vs Velocidad)

Los usuarios requerirán distintos perfiles (rápido vs. alta calidad). Con esta arquitectura, cambiar el modelo generador o el LLM traductor no requiere tocar la lógica de búsqueda. Solo se crea un nuevo "Adaptador" que cumpla con la interfaz del "Puerto" definido.

---

## Estructura de Directorios

El proyecto debe organizarse separando estrictamente el *Dominio* (reglas de negocio e interfaces) de los *Adaptadores* (PyTorch, APIs externas, I/O).

Se recomienda la siguiente estructura a partir de la carpeta `src/`:

### `src/domain/`

**Reglas puras, sin dependencias externas (Sin PyTorch, sin APIs).**
Aquí se define el "qué" hace el sistema, no el "cómo".

- **Entidades/Data Classes**: Clases que representan datos puros. (ej. `Intent`, `MidiSequence`, `GenerationProfile`).
- **Interfaces (Puertos)**: Clases base abstractas (`abc.ABC`) que definen los contratos que deben cumplir los componentes externos.
  - `LLMTranslator`: Interfaz para traducir intención a prompt técnico.
  - `MidiGenerator`: Interfaz para generar tokens paso a paso.
  - `Evaluator`: Interfaz para calcular la recompensa de una secuencia.

### `src/adapters/`

**El código que toca el mundo exterior (El "cómo").**
Aquí van las implementaciones específicas que cumplen con las interfaces del dominio.

- `translators/`: Implementaciones de `LLMTranslator` (ej. script que llama a la API de Anthropic/Claude, script que usa OpenAI, o Mocks para testing).
- `generators/`: Implementaciones de `MidiGenerator` (ej. el wrapper que carga el Transformer de Text2MIDI, maneja tensores y GPU).
- `evaluators/`: Implementaciones de `Evaluator` (ej. script que carga el modelo CLAP, scripts con heurísticas musicales para penalizar disonancias).
- `audio/`: Manejo de I/O y conversiones (ej. wrapper para llamar a FluidSynth, decodificar MIDI a WAV en memoria o disco).

### `src/use_cases/`

**La lógica orquestadora.**

- Aquí vive el bucle principal (ej. `progressive_search.py`).
- Este código **solo conoce las interfaces del dominio**. Instancia el flujo: pide al traductor las variaciones, le dice al generador que avance un bloque, le pide al evaluador los puntajes, y poda las secuencias. No sabe si por detrás hay una GPU de NVIDIA o un Mock.

### `src/config/`

**Configuración y Perfiles de Ejecución.**

- Archivos que definen los hiperparámetros según la necesidad del usuario.
- Ejemplos de perfiles:
  - *One-Shot*: `batch_size` alto, 1 beam (rápido, baja calidad garantizada).
  - *Balanced*: `batch_size` medio, 3 beams (punto dulce).
  - *Deep-Search*: `batch_size` pequeño, múltiples beams (lento, alta calidad musical).

