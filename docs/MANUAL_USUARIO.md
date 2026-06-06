# Manual de Usuario — text2midi

- [1. Introducción](#1. Introducción)
- [2. Requisitos](#2-requisitos)
    - [2.1 Software](#2-1-software)
    - [2.2 Hardware](#2-2-hardware)
    - [2.3 Cuenta de Google AI Studio](#2-3-cuenta-google)
- [3. Instalación](#3-instalacion)
    - [3.1 Clonar el repositorio](#3-1-clonar)
    - [3.2 Sincronizar dependencias](#3-2-sincronizar)
    - [3.3 Configurar variables de entorno](#3-3-variables)
    - [3.4 Descargar los pesos de los modelos](#3-4-modelos)
- [4. Primera Ejecución](#4-primera-ejecucion)
- [5. Resultados Generados](#5-resultados)
- [6. Flujos de Trabajo por Rol](#6-flujos-rol)
- [7. Uso de la Traducción LLM](#7-traduccion)
- [8. Perfiles de Generación](#8-perfiles)
- [9. Control de Calidad y Parámetros](#9-calidad)
- [10. Gestión de Modelos y Pesos](#10-modelos)
- [11. Configuración de FluidSynth](#11-fluidsynth)
- [12. VRAM y Rendimiento](#12-vram)
- [13. Troubleshooting](#13-troubleshooting)
- [14. Recursos Adicionales](#14-recursos)

## 1. Introducción

El sistema text2midi es una herramienta de interfaz de línea de comandos diseñada para la síntesis de música simbólica en formato MIDI a partir de descripciones textuales. Su propósito es resolver la brecha entre la conceptualización lingüística de una obra y su representación técnica, permitiendo que la intención musical se traduzca en datos estructurados sin requerir conocimientos profundos de programación.

La herramienta está orientada a tres perfiles de usuario: músicos que buscan prototipado rápido, investigadores de inteligencia artificial interesados en la generación musical y desarrolladores que requieran integrar capacidades de síntesis MIDI. Para un análisis detallado sobre el diseño del pipeline de inferencia y el uso de *Reward-Guided Beam Search* (búsqueda por haz guiada por recompensas), se recomienda consultar el documento `docs/ARCHITECTURE.md`.

## 2. Requisitos {#2-requisitos}

### 2.1 Software {#2-1-software}
| Componente | Versión Mínima | Observación |
|---|---|---|
| Python | `3.12+` | Requerido para compatibilidad con tipado moderno |
| `uv` | Última estable | Gestor de dependencias y entornos recomendado |
| FluidSynth | v2.0+ | Necesario para el renderizado de audio |
| `GOOGLE_API_KEY`| N/A | Opcional (solo para el modo de traducción LLM) |

### 2.2 Hardware {#2-2-hardware}
| Recurso | Mínimo (Evaluación/CPU) | Recomendado (MidiLLM/GPU) |
|---|---|---|
| GPU | No requerida | NVIDIA con  >4 GB VRAM |
| CPU | Quad-Core 2.5 GHz | Octa-Core 3.0 GHz+ |
| RAM | 8 GB | 16 GB |
| Disco | 5 GB libres | 10 GB libres (para pesos de modelos) |

### 2.3 Cuenta de Google AI Studio {#2-3-cuenta-google}
El uso de una `GOOGLE_API_KEY` es opcional. Solo es necesaria cuando se utiliza un modelo de traducción (ej. `gemma-4-31b`) para convertir lenguaje natural en prompts técnicos. En el modo *pass-through* (paso directo), el sistema no requiere conectividad externa.

> ⚠️ **Advertencia:** La ausencia de FluidSynth en el `PATH` del sistema es la causa más común de errores durante la ejecución de la síntesis.

## 3. Instalación {#3-instalacion}

### 3.1 Clonar el repositorio {#3-1-clonar}
Se debe obtener una copia local del código fuente mediante el comando `git clone`:
```bash
git clone https://github.com/colrudev/text2midi.git
cd text2midi
```

### 3.2 Sincronizar dependencias {#3-2-sincronizar}
El proyecto utiliza `uv` para garantizar la reproducibilidad del entorno. Ejecute el siguiente comando para instalar todas las dependencias declaradas:
```bash
uv sync
```

### 3.3 Configurar variables de entorno {#3-3-variables}
El sistema requiere un archivo `.env` para gestionar claves de API y configuraciones sensibles. Cree el archivo a partir de la plantilla proporcionada:
```bash
cp .env.example .env
```
El archivo resultante debe seguir este formato:
```ini
# Clave para Google AI Studio (opcional)
GOOGLE_API_KEY=tu_clave_aqui
```

### 3.4 Descargar los pesos de los modelos {#3-4-modelos}
Los pesos de los modelos generadores deben ubicarse en el directorio `models/` en la raíz del proyecto. 
> **Nota:** La descarga de estos pesos es un prerrequisito manual del usuario; el sistema no los descarga automáticamente durante la instalación.

Para una descripción detallada de los argumentos y banderas disponibles, consulte la `docs/cli_reference.md`.

Continúa con 4 para ejecutar la primera generación.

## 4. Primera Ejecución {#4-primera-ejecucion}

El sistema permite la generación de música MIDI mediante la interfaz de línea de comandos. Para asegurar la correcta ejecución, se debe utilizar el módulo de CLI en lugar del script de entrada.

> ⚠️ **Advertencia:** El archivo `main.py` es un stub de compatibilidad. El comando canónico de ejecución es `python -m src.cli`.

### 4.1 Ejecución básica (Modo Pass-through)
En este modo, la descripción textual se utiliza directamente como instrucción técnica, omitiendo la traducción por LLM.

```bash
python -m src.cli --text "Piano acoustic grand, 4/4, 120 bpm" --output simple.mid
```

### 4.2 Ejecución con Traductor LLM
Este modo traduce una descripción en lenguaje natural a un prompt técnico estructurado antes de la generación. Requiere la configuración de la variable `GOOGLE_API_KEY` en el archivo `.env`.

```bash
python -m src.cli --text "Una pieza de piano atmosférica y cinematográfica" --translator-model "gemini-2.5-flash" --profile balanced --output cinematic.mid
```

El argumento `--profile` define la estrategia de búsqueda y calidad; por defecto se utiliza el perfil `balanced`. Para una lista exhaustiva de banderas y argumentos, consulte la `docs/cli_reference.md`.

## 5. Resultados Generados {#5-resultados}

Por cada proceso de generación exitoso, el sistema escribe dos archivos en la ruta especificada:

1. **Archivo MIDI** (`.mid`): Contiene la secuencia musical simbólica generada.
2. **Prompt Técnico** (`.txt`): Un archivo compañero que almacena la instrucción técnica final utilizada para guiar al modelo. Este archivo es fundamental para la auditoría de los resultados del traductor.

Ejemplo de salida: `piano.mid` y `piano.txt`.

## 6. Flujos de Trabajo por Rol {#6-flujos-rol}

El uso de `text2midi` varía según los objetivos del usuario y su perfil técnico.

### 6.1 Para músicos
> **Dirigido a:** Músicos

Se recomienda el uso de descripciones en lenguaje natural detalladas para aprovechar el modo de traducción. Para iteraciones rápidas, el perfil `one-shot` es adecuado, mientras que para resultados finales se sugiere `deep-search`. La escucha de los resultados se realiza mediante la reproducción del archivo `.mid` en cualquier estación de trabajo de audio digital (DAW) o mediante FluidSynth.

### 6.2 Para investigadores de inteligencia artificial
> **Dirigido a:** Investigadores de IA

El sistema implementa dos estrategias de búsqueda guiadas por recompensas basadas en CLAP y heurísticas musicales:
- *Progressive search* (búsqueda progresiva): Estrategia secuencial utilizada en los perfiles `balanced` y `deep-search`.
- *Best-of-N* (el mejor de N): Generación en lote utilizada por el perfil `midillm-fast`.

Para asegurar la reproducibilidad, se debe considerar que el *beam search* (búsqueda por haz) es secuencial y el tiempo de computación crece exponencialmente con el número de haces.

### 6.3 Para desarrolladores
> **Dirigido a:** Desarrolladores

Para tareas de depuración y optimización, se recomienda el uso de las banderas `--print-prompt` y `--verbose`. Asimismo, la bandera `--strict-instruments` permite penalizar la aparición de instrumentos no solicitados en la generación. Para detalles sobre la implementación de puertos y adaptadores, consulte la `ARCHITECTURE.md`, y para la referencia de comandos, la `docs/cli_reference.md`.

## 7. Uso de la Traducción LLM {#7-traduccion}

El sistema opera en dos modalidades de procesamiento de entrada:

1. **Modo pass-through** (sin `--translator-model`): El texto proporcionado en `--text` se envía directamente al generador como un prompt técnico.
2. **Modo traductor** (con `--translator-model`): El texto es procesado por un LLM (como `gemini-2.5-flash`), que devuelve un prompt técnico estructurado que posteriormente alimenta al generador.

Comparativa de ejecución:

```bash
# Modo pass-through (Precisión técnica)
python -m src.cli --text "Acoustic Grand Piano, 4/4, 120bpm" --output direct.mid

# Modo traductor (Creatividad y lenguaje natural)
python -m src.cli --text "Un piano triste y lento" --translator-model "gemini-2.5-flash" --output translated.mid
```

El prompt resultante de la traducción es el contenido que se persiste en el archivo compañero `.txt` descrito en la 5.

## 8. Perfiles de Generación {#8-perfiles}

El sistema expone distintos perfiles de generación que permiten equilibrar velocidad, consumo de VRAM y calidad musical. Cada perfil activa una combinación específica de estrategia de búsqueda y número de candidatos, controlable mediante el flag `--profile` (valor por defecto: `balanced`). La siguiente tabla resume las cuatro opciones disponibles, cuyos argumentos detallados se documentan en la `cli_reference.md`.

| Perfil | Estrategia de búsqueda | VRAM mínimo | Velocidad relativa | Calidad esperada | Caso de uso recomendado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `one-shot` | *Beam search* (búsqueda por haz) con 2 haces | 2 GB | Muy alta (segundos) | Aceptable | Pruebas rápidas, validación de prompts, entornos sin GPU dedicada. |
| `balanced` (default) | *Beam search* con 5 haces | 4 GB | Media (1–3 min) | Alta | Producción general y experimentación iterativa. |
| `deep-search` | *Progressive search* (búsqueda progresiva) con 8 haces | 6 GB | Baja (10–30 min) | Máxima | Composiciones complejas, evaluación exhaustiva, generación final. |
| `midillm-fast` | *Best-of-N* (el mejor de N) con *batch processing* (procesamiento por lotes) | 4 GB | Alta (minutos) | Alta | Hardware con VRAM limitada que prioriza velocidad. |

El perfil `midillm-fast` aplica procesamiento por lotes sobre el modelo MidiLLM, lo cual lo hace significativamente más rápido que `balanced` a pesar de basarse en un modelo más pesado. Por su parte, `deep-search` aplica una *progressive search* cuyo tiempo de generación crece de forma exponencial con el número de haces configurados, por lo que se recomienda reservar este perfil para iteraciones de validación final.

### 8.1 Invocación por línea de comandos

La selección del perfil se realiza en el momento de la invocación mediante el flag `--profile`. El siguiente ejemplo ilustra el uso de `deep-search` combinado con un traductor LLM para una composición de validación:

```bash
python -m src.cli \
  --text "Una orquesta cinematográfica melancólica en compás 3/4" \
  --profile deep-search \
  --translator-model "gemini-2.5-flash" \
  --output orchestra.mid
```

Para una iteración rápida sin GPU dedicada, el perfil `one-shot` ofrece resultados aceptables en segundos, ideal durante la fase de diseño de prompts.

### 8.2 Compatibilidad entre perfiles y hardware

No todos los perfiles son ejecutables en cualquier hardware. `one-shot` y `balanced` funcionan tanto en CPU como en GPU; `deep-search` exige una GPU con al menos 6 GB de VRAM para evitar paginación a memoria del sistema, y `midillm-fast` requiere estrictamente ≥ 4 GB de VRAM. Si la invocación excede la VRAM disponible, el sistema emite un error de tipo CUDA OOM y la ejecución se interrumpe.

## 9. Control de Calidad y Parámetros {#9-calidad}

El pipeline evalúa cada MIDI candidato mediante dos capas complementarias de *scoring* que se combinan en una puntuación final:

1. **CLAP** (*Contrastive Language-Audio Pretraining*): modelo neuronal preentrenado que asigna una puntuación de alineación audio-texto entre el prompt técnico y la representación sonora inferida del MIDI.
2. **Heurísticas musicales**: conjunto de reglas deterministas que penalizan disonancias, notas fuera de tonalidad, inconsistencias de tempo y otras violaciones a las convenciones de la notación MIDI.

La etapa de evaluación se ejecuta predominantemente en CPU y constituye el principal cuello de botella del perfil `deep-search`, donde se evalúan múltiples candidatos en cada iteración del bucle de búsqueda.

El flag `--strict-instruments` añade una penalización adicional cuando el MIDI generado contiene instrumentos no solicitados en el prompt técnico, lo que permite reforzar el cumplimiento del descriptor musical. El flag `--print-prompt`, por su parte, imprime el prompt técnico resultante por la etapa de traducción para facilitar la auditoría del proceso. La lista completa de parámetros y sus valores por defecto se documenta en la `cli_reference.md`.

### 9.1 Interpretación de las puntuaciones

La puntuación combinada devuelta por la etapa de evaluación se expresa como un valor escalar en el rango `[0, 1]`, donde valores cercanos a `1.0` indican alta alineación con el prompt técnico y baja presencia de violaciones heurísticas. Esta métrica es relativa entre ejecuciones y no debe interpretarse como una medida absoluta de calidad musical. Se recomienda comparar generaciones de un mismo prompt para identificar diferencias atribuibles al perfil o al hardware subyacente.

### 9.2 Reproducibilidad y semillas

El pipeline admite la configuración de semillas aleatorias mediante variables de entorno, lo que permite reproducir resultados entre invocaciones. Esta funcionalidad es especialmente relevante para investigadores que requieran comparar perfiles bajo condiciones controladas. El detalle de las variables disponibles se documenta en la `cli_reference.md`.

## 10. Gestión de Modelos y Pesos {#10-modelos}

El sistema soporta dos generadores principales, cada uno con requisitos y características distintas:

- **Text2Midi**: arquitectura *Transformer* encoder-decoder construida sobre `FlanT5-base` con tokenización `REMI+`. Constituye el generador por defecto para los perfiles `one-shot`, `balanced` y `deep-search`, y aplica *progressive search* (búsqueda progresiva) con evaluación iterativa. Su tiempo de generación crece de forma exponencial con el número de haces.
- **MidiLLM**: modelo más pesado que procesa la generación mediante *batch processing* (procesamiento por lotes). Es utilizado exclusivamente por el perfil `midillm-fast` y requiere al menos 4 GB de VRAM. Resulta significativamente más rápido que `Text2Midi` gracias al paralelismo del procesamiento por lotes.

Los pesos de ambos modelos (*model weights*) deben ubicarse en el directorio `models/` descrito en la 3.4, y se obtienen desde los repositorios públicos de Hugging Face indicados en el `README.md`. Si los archivos no están presentes, el pipeline abortará durante la carga inicial con un error de tipo `FileNotFoundError`.

### 10.1 Formatos de pesos soportados

El sistema acepta archivos en formato `.safetensors` y `.bin` (PyTorch). Se recomienda el uso de `safetensors` por su carga más rápida y por evitar la ejecución de código arbitrario durante la deserialización. Cada modelo publica un manifiesto con la lista exacta de archivos requeridos, que debe replicarse dentro de `models/` respetando la estructura de subdirectorios original.

### 10.2 Actualización y versionado

Las versiones de los pesos publicadas en Hugging Face están etiquetadas mediante *tags* semánticos. Se recomienda fijar una versión específica al descargar los archivos para garantizar la reproducibilidad de los resultados a lo largo del tiempo. La actualización a una versión mayor puede requerir cambios en el código del pipeline; se sugiere revisar el `CHANGELOG.md` del repositorio antes de actualizar.

## 12. VRAM y Rendimiento {#12-vram}

La selección del perfil de generación debe considerar la VRAM disponible y el tiempo máximo aceptable para una iteración. La siguiente tabla ofrece un árbol de decisión en función del hardware del usuario:

| Hardware disponible | Perfil recomendado | Justificación |
| :--- | :--- | :--- |
| Sin GPU dedicada (solo CPU) | `one-shot` | Ejecución viable en CPU, aunque lenta; produce resultados en segundos. |
| GPU con 4 GB de VRAM | `midillm-fast` | Mejor relación velocidad/calidad para hardware con VRAM limitada. |
| GPU con ≥ 8 GB de VRAM | `balanced` o `deep-search` | Permite exploración progresiva de mayor calidad. |

En términos de costo temporal, el perfil `deep-search` con 5 o más haces puede demandar entre 10 y 30 minutos por generación, dependiendo de la longitud del prompt y la disponibilidad de GPU. En contraste, `one-shot` se completa en segundos y `midillm-fast` se sitúa en el orden de minutos.

> **Nota:** Si la generación finaliza con un error de falta de memoria (OOM, *Out Of Memory*), se recomienda reducir el tamaño de lote o cambiar a un perfil más liviano como `one-shot` o `balanced`. El flag `--verbose` permite inspeccionar el consumo de memoria y la telemetría de cada iteración durante la ejecución.

### 12.1 Tiempos de generación esperados

La siguiente tabla resume los tiempos de generación orientativos para un prompt técnico de longitud media (≤ 200 tokens) en hardware representativo:

| Perfil | CPU (sin GPU) | GPU 4 GB | GPU ≥ 8 GB |
| :--- | :--- | :--- | :--- |
| `one-shot` | 5–15 s | 2–5 s | 1–3 s |
| `balanced` | 5–10 min | 1–3 min | 30–60 s |
| `deep-search` | No recomendado | 10–30 min | 5–15 min |
| `midillm-fast` | No soportado | 1–3 min | 30–90 s |

Los valores anteriores son aproximados y dependen de la longitud del prompt, el número de instrumentos solicitados y la temperatura del muestreo.

### 12.2 Verificación del entorno CUDA

Antes de invocar perfiles que requieran GPU, se recomienda verificar que el runtime de CUDA sea detectable por PyTorch. El siguiente comando, ejecutado en un *shell* de Python dentro del entorno virtual, confirma la disponibilidad de la GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Si la salida indica `No GPU`, se debe revisar la instalación de los drivers NVIDIA y de la versión de CUDA compatible con la build de PyTorch declarada en el `pyproject.toml`.

### 12.3 Escalamiento incremental

Se sugiere adoptar una estrategia de escalamiento incremental: iniciar la experimentación con `one-shot` para validar el prompt técnico, ascender a `balanced` para evaluar la calidad base, y reservar `deep-search` o `midillm-fast` para la generación definitiva. Esta progresión minimiza el tiempo total invertido en iteraciones fallidas y permite aislar la influencia del perfil en la calidad musical resultante.

## 11. Configuración de FluidSynth {#11-fluidsynth}

FluidSynth es un sintetizador de MIDI en tiempo real que el sistema utiliza como adaptador de salida para el renderizado de audio. Actúa como puente entre las secuencias simbólicas producidas por el modelo y la generación audible mediante *soundfonts* (colecciones de muestras instrumentales). Su instalación correcta se anticipó en 2.1 como requisito de sistema operativo y constituye el punto de fallo más frecuente en la fase de puesta en marcha.

> ⚠️ **Advertencia:** La ausencia del binario `fluid-synth` o de su biblioteca nativa es la causa más habitual de fallo durante la primera ejecución. Ningún perfil de generación puede producir audio si esta dependencia no se encuentra correctamente instalada y accesible en el `PATH` del sistema.

### 11.1 Instalación en macOS

En sistemas macOS con Homebrew como gestor de paquetes, la instalación se reduce a una única instrucción:

```bash
brew install fluid-synth
```

### 11.2 Instalación en Linux (Debian/Ubuntu)

En distribuciones basadas en Debian, el paquete se encuentra disponible en los repositorios oficiales:

```bash
sudo apt-get install fluidsynth
```

### 11.3 Instalación en Linux (Fedora/RHEL)

En distribuciones de la familia Red Hat, el equivalente se obtiene mediante el gestor `dnf`:

```bash
sudo dnf install fluidsynth
```

### 11.4 Instalación en Windows

La vía recomendada consiste en descargar el *release* oficial de FluidSynth desde su repositorio, extraer el contenido y agregar el subdirectorio `bin` a la variable de entorno `PATH` del sistema. Como alternativa avanzada, MSYS2 ofrece el paquete `mingw-w64-x86_64-fluidsynth` para usuarios familiarizados con dicho entorno.

La instalación puede verificarse en cualquier sistema operativo mediante la siguiente consulta en una terminal:

```bash
fluid-synth --version
```

## 13. Troubleshooting {#13-troubleshooting}

La siguiente tabla compendia los fallos observados con mayor frecuencia durante la operación del sistema. Cada fila asocia un síntoma reconocible a su causa probable y a la acción correctiva documentada en otra sección de este manual.

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'pyfluidsynth'` | FluidSynth no instalado a nivel de sistema, o `uv sync` no ejecutado tras la clonación del repositorio. | Ejecutar `uv sync`; si el error persiste, instalar la biblioteca nativa de FluidSynth conforme a 11. |
| `OSError: cannot open shared object file: libfluidsynth.so.3` (Linux) | Biblioteca nativa de FluidSynth ausente en las rutas de carga del enlazador dinámico. | Instalar el paquete `fluidsynth` mediante el gestor de paquetes del sistema operativo (11.2 o 11.3). |
| `KeyError: 'GOOGLE_API_KEY'` o error de la API de Gemini | Variable de entorno ausente o inválida en el archivo `.env`. | Editar el archivo `.env` y proporcionar una clave válida, o invocar el CLI en modo *pass-through* omitiendo la bandera `--translator-model` (7). |
| `torch.cuda.OutOfMemoryError` con `midillm-fast` o `deep-search` | Memoria de vídeo (VRAM) insuficiente para alojar los tensores del modelo seleccionado. | Cambiar al perfil `balanced` o `one-shot` según la guía de 12, o liberar memoria cerrando otros procesos que consuman GPU. |
| `FileNotFoundError: models/text2midi/...` | Pesos del modelo no descargados en el directorio `models/` del repositorio. | Revisar 10 y descargar los *checkpoints* (puntos de control del modelo) desde Hugging Face hacia el directorio `models/`. |
| La generación se prolonga durante horas sin progresar visiblemente | Perfil `deep-search` con un `beam_width` excesivamente alto sobre hardware limitado. | Reducir el `beam_width` mediante los parámetros del perfil o sustituir el perfil por `balanced` (8). |
| El MIDI resultante suena fuera de tonalidad o con disonancias severas | Heurísticas musicales del evaluador penalizando combinaciones legítimas. | Revisar el *prompt* técnico con `--print-prompt` (9) y activar `--strict-instruments` para forzar el cumplimiento estricto de los instrumentos solicitados. |
| `python: command not found` al invocar el comando canónico | Python 3.12 no instalado o no presente en el `PATH` del sistema. | Instalar Python 3.12 desde el sitio oficial `python.org` y verificar la disponibilidad con `python --version`. |

Para errores no contemplados en la tabla precedente, se recomienda consultar la referencia exhaustiva de banderas en `docs/cli_reference.md` y reproducir la ejecución con la bandera `--verbose` para obtener trazas a nivel `DEBUG` que faciliten el diagnóstico.

## 14. Recursos Adicionales {#14-recursos}

Los cuadernos de análisis y experimentación están disponibles en el directorio `notebooks/` y pueden ejecutarse en Google Colab mediante las insignias dispuestas en el `README.md` del repositorio.

Para profundizar en aspectos técnicos del proyecto, se recomienda la lectura complementaria de los siguientes documentos, cuyos contenidos no se duplican en este manual. La [Arquitectura del proyecto](ARCHITECTURE.md) detalla las decisiones de Clean/Hexagonal, la separación entre puertos y adaptadores, y la organización del dominio y los casos de uso. La [Referencia de CLI](cli_reference.md) ofrece la descripción exhaustiva de cada bandera, sus valores por defecto y ejemplos avanzados de invocación.

Para reportar incidencias, solicitar soporte o participar en discusiones de diseño, se pueden utilizar los *issues* del repositorio en GitHub, así como los canales de comunicación dispuestos por la MSc.
