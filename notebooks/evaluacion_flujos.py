# %% [markdown]
# # Evaluación de Flujos Text2Midi y MidiLLM
#
# Este notebook es parte de la investigación académica de maestría. Su propósito es evaluar el flujo completo de generación de MIDI desde descripciones naturales, comparando las dos estrategias principales:
# 1. **Text2Midi** (Progressive Search / Step-by-Step) usando el perfil `BALANCED`.
# 2. **MidiLLM** (Batch Generation / Best-of-N Search) usando el perfil `MIDILLM_FAST`.
#
# El objetivo es documentar la traducción (technical prompt) resultante y el MIDI generado final para un objetivo específico.

# %% [markdown]
# ## 1. Configuración de dependencias y entorno

# %%
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Obtener el directorio raíz del proyecto detectando la carpeta 'src'
current_dir = Path.cwd()
if (current_dir / "src").exists():
    project_root = current_dir
elif (current_dir.parent / "src").exists():
    project_root = current_dir.parent
else:
    # Si se ejecuta como script .py
    project_root = (
        Path(__file__).resolve().parent.parent
        if "__file__" in globals()
        else current_dir
    )

# Cambiar el directorio de trabajo a la raíz para que las rutas relativas (como models/) funcionen correctamente
if Path.cwd() != project_root:
    os.chdir(project_root)
    print(f"Directorio de trabajo cambiado a: {project_root}")

# Asegurar que importamos desde src
sys.path.append(str(project_root / "src"))

load_dotenv()  # Carga de GOOGLE_API_KEY desde .env

# Verificar la clave
if not os.getenv("GOOGLE_API_KEY"):
    print(
        "⚠️ ADVERTENCIA: No se encontró GOOGLE_API_KEY en el entorno. La traducción de LLM no funcionará correctamente."
    )

# %% [markdown]
# ## 2. Importar componentes del Pipeline

# %%
# Configurar logs para ver el proceso
import logging

from adapters.translators.google_ai_translator import GoogleAIConfig
from config.profiles import get_profile
from pipeline import Text2MidiPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# %% [markdown]
# ## 3. Inicializar Pipeline
#
# Instanciamos el pipeline inyectando la configuración para el traductor. Los modelos generadores (Text2Midi y MidiLLM) se inicializan bajo demanda según el perfil elegido.

# %%
# Usar Gemini 2.5 Flash u otro modelo para la traducción
translator_config = GoogleAIConfig(model_name="gemma-4-26b-a4b-it", temperature=0.9)

print("Inicializando pipeline...")
pipeline = Text2MidiPipeline(translator_config=translator_config)
print("✅ Pipeline inicializado.")

# %% [markdown]
# ## 4. Definición de la Prueba Base
#
# Definimos el prompt natural que servirá como base para ambas pruebas.

# %%
# Prompt natural del usuario / investigador
user_prompt = "Una melodía de piano melancólica y lenta, con un acompañamiento suave de cuerdas de fondo."

# Directorio de salida
output_dir = Path("outputs/evaluacion-notebook")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"📝 Prompt de prueba: '{user_prompt}'")

# %% [markdown]
# ## 5. Prueba 1: Flujo Text2Midi (Progressive Search)
#
# Utilizaremos el perfil `balanced`, que usa beam search y evaluación progresiva con heurísticas y CLAP.

# %%
print("\n" + "=" * 50)
print("🎹 INICIANDO PRUEBA 1: Text2Midi (Progressive Search)")
print("=" * 50)

# Cargar el perfil más rápido
profile_t2m = get_profile("one-shot")

# Generar MIDI
result_t2m = pipeline.generate(text=user_prompt, profile=profile_t2m)

print("\n✨ Resultados Text2Midi:")
print(
    f"- Traducción (Technical Prompt) usada por el generador:\n{result_t2m.technical_prompt}"
)
print(f"- Tamaño del MIDI generado: {len(result_t2m.midi_bytes)} bytes")

# Guardar
out_path_t2m = output_dir / "text2midi_resultado.mid"
with open(out_path_t2m, "wb") as f:
    f.write(result_t2m.midi_bytes)
print(f"💾 Guardado en: {out_path_t2m}")


# %% [markdown]
# ## 6. Prueba 2: Flujo MidiLLM (Batch / Best-of-N Search)
#
# Utilizaremos el perfil `midillm-fast`, que genera múltiples secuencias candidatas en bloque (batch) y luego usa el evaluador para elegir la mejor (Best-of-N).

# %%
print("\n" + "=" * 50)
print("🎹 INICIANDO PRUEBA 2: MidiLLM (Best-of-N)")
print("=" * 50)

# Cargar el perfil midillm
profile_mllm = get_profile("midillm-fast")

# Generar MIDI
result_mllm = pipeline.generate(text=user_prompt, profile=profile_mllm)

print("\n✨ Resultados MidiLLM:")
print(
    f"- Traducción (Technical Prompt) usada por el generador:\n{result_mllm.technical_prompt}"
)
print(f"- Tamaño del MIDI generado: {len(result_mllm.midi_bytes)} bytes")

# Guardar
out_path_mllm = output_dir / "midillm_resultado.mid"
with open(out_path_mllm, "wb") as f:
    f.write(result_mllm.midi_bytes)
print(f"💾 Guardado en: {out_path_mllm}")

# %% [markdown]
# ## 7. Conclusiones y Revisión
#
# * Revisa las traducciones (technical_prompts) para asegurar que el traductor interpretó correctamente la intención.
# * Escucha los archivos MIDI generados (`outputs/evaluacion-notebook/*.mid`) y compáralos según los criterios de la investigación.
#
# ### Resultados Observados
# 
# Al evaluar la generación de ambos modelos, el traductor generó el siguiente prompt técnico:
# 
# > "A cinematic classical track with a melancholic and somber vibe, featuring a delicate felt piano melody accompanied by a soft, legato string ensemble. The song is in the key of A minor with a 4/4 time signature and a slow Adagio tempo. The 8-measure chord progression Am - Dm - G - E, followed by Am - Dm - F - E7 - Am, guides the listener through a journey of quiet sadness, moving from a state of unresolved longing toward a gentle, final resolution."
# 
# - Tamaño del MIDI generado: 2105 bytes
#
# **Análisis Comparativo:**
# 
# **Text2Midi (Progressive Search):**
# Generó demasiados instrumentos con poco sentido musical y un número reducido de notas en general, desviándose de la instrumentación solicitada.
# 
# **MidiLLM (Batch / Best-of-N Search):**
# - **Instrumentación:** Se restringió exclusivamente a los 2 instrumentos indicados, aunque invirtió los roles (le dio la melodía a las cuerdas/string y el legato al piano).
# - **Métrica y Tempo:** Cambió la métrica internamente (aunque al oído se percibe como 4/4) y persistieron los sesgos de generar un tempo rápido, probablemente heredados de los datasets de entrenamiento del modelo.
# - **Armonía y Progresión:** No siguió la progresión exacta. Mantuvo la cantidad de acordes, pero modificó algunos: respetó los 3 primeros y en el cierre (F - E - Am) le agregó un G al final. El cambio más grande fue en la mitad de la frase, cambiando el V (quinto grado) por el iii (tercer grado) y usando el vi (sexto grado) para transicionar al cierre, en lugar de repetir el comienzo de la frase anterior como indicaba el prompt.
