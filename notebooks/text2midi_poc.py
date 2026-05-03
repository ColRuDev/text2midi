# %% [markdown]
# # Análisis Profundo de Text2MIDI
#
# ##🎯 Objetivo del Notebook
# El propósito de este análisis es desglosar y visualizar el flujo de datos dentro de un modelo Transformer Encoder-Decoder especializado en la conversión de lenguaje natural a secuencias MIDI. Buscamos entender cómo el modelo "comprende" una descripción textual y la traduce matemáticamente en eventos musicales, analizando cada bloque funcional: desde la tokenización hasta la reconstrucción del archivo final.
#
# Autor:
#
# [Nicolas Colmenares]
#
# Fecha de creación: 03/06/26
#
# **Resumen:**
# Este notebook realiza una "autopsia" técnica de un modelo de Hugging Face. Pasaremos por la preparación del entorno, la inspección de los Embeddings, el mapeo de los mecanismos de Self-Attention en el Encoder, la dinámica de la Cross-Attention en el Decoder y, finalmente, la interpretación de los Logits de salida que conforman la estructura de un archivo MIDI.
#
# # [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ColRuDev/text2midi/blob/main/notebooks/text2midi_poc.ipynb)
#

# %% [markdown]
# ## Fase 1: Configuración del Entorno y Carga del Modelo
#
# En esta fase inicial, se prepara el laboratorio de pruebas. Se importa las dependencias necesarias para el procesamiento de audio y tensores, y cargamos el modelo de Hugging Face utilizando configuraciones específicas (output_attentions=True) que nos permitirán extraer los datos internos que normalmente están ocultos durante una inferencia estándar.

# %%
# !pip install jsonlines st-moe-pytorch miditok --quiet

# %%
# imports
import pickle
import torch
import torch.nn as nn
import sys
import os
from transformers import T5Tokenizer
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
import plotly.express as px
import torch
import torch.nn.functional as F


# %%
if not os.path.exists('text2midi'):
    # !git clone https://github.com/amaai-lab/text2midi.git

from text2midi.model.transformer_model import Transformer

repo_id = "amaai-lab/text2midi"
model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Using device: {device}")

# Load the tokenizer dictionary
with open(tokenizer_path, "rb") as f:
    r_tokenizer = pickle.load(f)

vocab_size = len(r_tokenizer)
print("Vocab size: ", vocab_size)

# Initialize and load the custom Transformer model
model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# text2midi uses T5Tokenizer for the text encoder part
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

print('Model loaded successfully.')

# %%
print(f"Tipo de modelo: {type(model).__name__}")

# Encoder: T5EncoderModel
# En la versión de HF, las capas suelen estar en .encoder.block si es el modelo completo
# o en .block si es solo el encoder. Verificamos la ruta correcta.
if hasattr(model.encoder, 'block'):
    n_layers_encoder = len(model.encoder.block)
elif hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'block'):
    n_layers_encoder = len(model.encoder.encoder.block)
else:
    # Conteo por inspección de módulos
    n_layers_encoder = len([m for m in model.encoder.modules() if type(m).__name__ == 'T5Block'])

# Decoder: TransformerDecoder personalizado
n_layers_decoder = len(model.decoder.layers) if hasattr(model.decoder, 'layers') else 0

print(f"Capas del Encoder: {n_layers_encoder}")
print(f"Capas del Decoder: {n_layers_decoder}")

# %%
PROMPT = "A melodic electronic song with ambient elements, featuring piano, acoustic guitar, alto saxophone, string ensemble, and electric bass. Set in G minor with a 4/4 time signature, it moves at a lively Presto tempo. The composition evokes a blend of relaxation and darkness, with hints of happiness and a meditative quality."
print('Generating for prompt: ' + PROMPT)

# %%
model

# %% [markdown]
# ### ⚙ Análisis de la Arquitectura del Modelo
#
# Basándonos en la salida anterior, podemos identificar los tres pilares del modelo **Text2MIDI**:
#
# 1.  **Encoder (T5EncoderModel):** Utiliza 12 capas de un transformador T5 estándar. Su trabajo es procesar el texto del prompt y generar una representación matemática (contexto) que capture la intención musical.
#
# 2.  **Decoder (TransformerDecoder):** Es la parte más profunda con **18 capas**. A diferencia del encoder, este es autorregresivo: genera los eventos MIDI uno por uno, consultando constantemente la información del encoder mediante el mecanismo de **Multihead Attention**.
#
# 3.  **Embeddings y Proyecci#n:**
#     *   `input_emb`: Una matriz de 524 x 768 que mapea los tokens musicales.
#     *   `projection`: Una capa lineal final que decide, de entre las 524 posibilidades del vocabulario MIDI, cu!l es la próxima nota o evento m!s probable.

# %% [markdown]
# ## Fase 2: Procesamiento Inicial (Tokenización y Embeddings)
#
# Aquí se analiza el primer contacto del texto con el modelo. Observando cómo la cadena de texto se fragmenta en tokens y cómo estos se transforman en vectores numéricos dentro de un espacio latente (Embeddings). Además, explorar el Positional Encoding, la técnica que le permite al modelo entender el orden de las palabras sin tener una estructura recurrente.

# %%
inputs = tokenizer(PROMPT, return_tensors='pt', padding=True, truncation=True)
print(inputs)

# %%
# Obtenemos los IDs de los tokens
token_ids = inputs['input_ids'][0].tolist()

# Convertimos cada ID individualmente de vuelta a su texto (token)
tokens = [tokenizer.decode([tid]) for tid in token_ids]

# Creamos un DataFrame para una visualización tabular limpia
df_tokens = pd.DataFrame({
    'Token': tokens,
    'ID Numérico': token_ids
})

print("Mapeo de Tokenización (Texto -> ID):")
display(df_tokens.T) # Transpuesto para que sea más fácil de leer horizontalmente

# %% [markdown]
# ### Explicación de la Conversión
# En la tabla anterior, puedes ver cómo el **T5Tokenizer** fragmenta el lenguaje natural:
# 1.  **Sub-palabras:** Palabras complejas se dividen en fragmentos (ej. 'acoustic' podría dividirse si no está en el vocabulario base).
# 2.  **IDs:** Cada fragmento tiene un índice único en la matriz de embeddings del modelo.
# 3.  **Tokens Especiales:** El ID `1` al final suele representar el token `</s>` (fin de secuencia).

# %%
df_tokens.Token.unique()

# %%
input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=0)
input_ids = input_ids.to(device)
attention_mask =nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=0)
attention_mask = attention_mask.to(device)

# %% [markdown]
# Preparación de los tokens para que queden justo como los espera el modelo

# %% [markdown]
# ## Fase 3: El Viaje por el Encoder (Comprensión Contextual)
#
# El Encoder es el encargado de extraer el significado de nuestra petición. En esta fase, se visualiza las capas internas para ver cómo la información fluye y se transforma. Utilizaremos mapas de calor para inspeccionar la Self-Attention, identificando a qué palabras el modelo presta más atención para construir una representación rica del contexto musical solicitado.

# %%
# Obtenemos las salidas de todas las capas usando el método oficial del modelo
with torch.no_grad():
    # Pasamos el input por el encoder solicitando los estados ocultos
    encoder_outputs = model.encoder.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )

    # hidden_states es una tupla: (embeddings, capa_1, ..., capa_12)
    all_hidden_states = encoder_outputs.hidden_states

    print(f"Estado inicial (Embeddings): {all_hidden_states[0].shape}")
    print("-" * 50)

    # Iteramos sobre los estados de las capas (empezando desde el índice 1)
    for i in range(1, len(all_hidden_states)):
        layer_output = all_hidden_states[i]
        print(f"Capa Encoder {i:02d}: Output Shape -> {layer_output.shape}")

    print("-" * 50)
    last_hidden_state = encoder_outputs.last_hidden_state
    print(f"Output final del Encoder (Contexto): {last_hidden_state.shape}")

# %%
encoder_embeddings_np = last_hidden_state.squeeze(0).detach().cpu().numpy()
print(f"Shape of encoder_embeddings_np: {encoder_embeddings_np.shape}")

# Inicializamos UMAP con 2 componentes para visualización
reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_results = reducer.fit_transform(encoder_embeddings_np)
print(f"Shape of umap_results: {umap_results.shape}")

# %%
# 1. Preparar los datos filtrados
skip_tokens = ['</s>', '', ',', '.', ' ']
filtered_data = []

for i, token in enumerate(tokens):
    if token not in skip_tokens:
        filtered_data.append({
            'Token': token,
            'UMAP1': umap_results[i, 0],
            'UMAP2': umap_results[i, 1]
        })

df_umap = pd.DataFrame(filtered_data)

# 2. Crear la gráfica interactiva con Plotly
fig = px.scatter(
    df_umap,
    x='UMAP1',
    y='UMAP2',
    text='Token',
    title='Exploración Interactiva de Embeddings (UMAP)',
    template='plotly_white',
    hover_data={'UMAP2': ':.4f', 'UMAP2': ':.4f'}
)

# 3. Ajustar estilo de los puntos y etiquetas
fig.update_traces(
    textposition='top center',
    marker=dict(size=10, color='royalblue', opacity=0.6),
    mode='markers+text'
)

fig.update_layout(
    dragmode='pan', # Permite arrastrar por defecto
    width=900,
    height=700,
    xaxis_title="UMAP Dimensión 1",
    yaxis_title="UMAP Dimensión 2"
)

fig.show()

# %% [markdown]
# ## Fase 4: El Decoder y la Generación de MIDI
#
# El Decoder es la unidad creativa que genera la música token por token. Analizando el mecanismo de Cross-Attention, donde el modelo "mira" hacia atrás (al Encoder) para asegurarse de que la nota que está generando coincide con la descripción original. Desglosando el proceso autoregresivo paso a paso, observando cómo la probabilidad de cada nota cambia en cada iteración.

# %%
# 1. Preparar las variables de entrada
current_tokens = torch.tensor([[0]]).to(device)
tgt = current_tokens.to(device)
memory = last_hidden_state.to(device)

# 2. inicializar el embedding y el positional encoding con el tensor MIDI vacio
decoder_hidden_state = model.input_emb(tgt) * (768 ** 0.5)
decoder_hidden_state = model.pos_encoder(decoder_hidden_state)

decoder_layer_outputs = []
current_hs = decoder_hidden_state
print(f"Initial Decoder Embedding Shape: {current_hs.shape}")
print("-" * 30)

# 3. Iteramos sobre las capas del decoder
for i, layer in enumerate(model.decoder.layers):
    current_hs = layer(current_hs, memory)
    decoder_layer_outputs.append(current_hs)
    print(f"Layer {i+1:02d} Hidden State Shape: {current_hs.shape}")

# 7. Final Layer Norm
final_decoder_hs = model.decoder.norm(current_hs)
print("-" * 30)
print(f"Final Decoder Output (Post-Norm) Shape: {final_decoder_hs.shape}")

# %%
print(f"--- Iniciando Generación Paso a Paso ---\n")

# El modelo requiere src_mask.
src_mask = attention_mask.to(device)

# Obtenemos el mapeo inverso de vocabulario para visualización
# r_tokenizer.vocab es un diccionario {event_str: id}
inv_vocab = {v: k for k, v in r_tokenizer.vocab.items()}

max_steps = 10
with torch.no_grad():
    for step in range(max_steps):
        # 2. Inferencia
        logits = model(src=input_ids, tgt=current_tokens, src_mask=src_mask)

        # 3. Tomamos los logits del último token
        next_token_logits = logits[:, -1, :]

        # 4. Greedy search
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        # 5. Concatenamos
        current_tokens = torch.cat([current_tokens, next_token], dim=1)

        # Visualización
        token_id = next_token.item()
        token_name = inv_vocab.get(token_id, "Unknown")

        print(f"Paso {step+1:02d}: ID {token_id:3} -> Evento: {token_name}")

print(f"\nSecuencia final generada (primeros 10): {current_tokens.tolist()}")

# %%
# 1. Preparar las variables de entrada
tgt = current_tokens.to(device)
memory = last_hidden_state.to(device)

# 2. inicializar el embedding y el positional encoding con el tensor MIDI vacio
decoder_hidden_state = model.input_emb(tgt) * (768 ** 0.5)
decoder_hidden_state = model.pos_encoder(decoder_hidden_state)

decoder_layer_outputs = []
current_hs = decoder_hidden_state
print(f"Initial Decoder Embedding Shape: {current_hs.shape}")
print("-" * 30)

# 3. Iteramos sobre las capas del decoder
for i, layer in enumerate(model.decoder.layers):
    current_hs = layer(current_hs, memory)
    decoder_layer_outputs.append(current_hs)
    print(f"Layer {i+1:02d} Hidden State Shape: {current_hs.shape}")

# 7. Final Layer Norm
final_decoder_hs = model.decoder.norm(current_hs)
print("-" * 30)
print(f"Final Decoder Output (Post-Norm) Shape: {final_decoder_hs.shape}")

# %% [markdown]
# El decoder genera un token Midi+ por cada paso de forma consecutiva y los va añadiendo al tensor de salida. Es decir, cada nuevo token considera los tokens previos

# %% [markdown]
# ## Fase 5: Salida Cruda y Reconstrucción Final
#
# En la etapa final, se examina los datos "crudos" que emite el modelo antes de ser convertidos en música. Analizando la tabla de vocabulario MIDI para entender qué representan los IDs de salida (notas, velocidades, tiempos) y utilizando herramientas de visualización (como Piano Rolls) para comparar la intención inicial del texto con el resultado sonoro obtenido.

# %% [markdown]
# > Temperatura inferior a 0.5 resulta en una generación de pocas notas

# %%
output = model.generate(input_ids, attention_mask, max_len=2000,temperature=0.5)
output_list = output[0].tolist()

# %%
# Ajuste de tipos para compatibilidad con el motor de decodificación de REMI
if not hasattr(r_tokenizer.config, 'additional_tokens'):
    r_tokenizer.config.additional_tokens = []

attrs_to_fix = {
    'use_velocities': True,
    'use_note_duration_programs': [],
    'use_programs': True,
    'use_chords': False,
    'use_rests': False,
    'use_tempos': True,
    'use_time_signatures': True,
    'use_sustain_pedals': False,
    'use_pitch_bends': False,
    'use_pitch_intervals': False,
    'program_changes': False,
    'default_note_duration': 0.5
}

for attr, value in attrs_to_fix.items():
    setattr(r_tokenizer.config, attr, value)

try:
    # Decodificación
    generated_midi = r_tokenizer.decode(output_list)
    print("¡Éxito! Música MIDI decodificada.")

    # symusic usa 'tracks' en lugar de 'instruments'
    if hasattr(generated_midi, 'tracks'):
        num_tracks = len(generated_midi.tracks)
    elif hasattr(generated_midi, 'instruments'):
        num_tracks = len(generated_midi.instruments)
    else:
        num_tracks = "desconocido"

    print(f"Instrumentos/Tracks detectados: {num_tracks}")

    # symusic utiliza dump_midi para guardar archivos
    generated_midi.dump_midi('resultado_text2midi.mid')
    print("Archivo 'resultado_text2midi.mid' guardado correctamente.")
except Exception as e:
    print(f"Error crítico al decodificar: {e}")
    import traceback
    traceback.print_exc()
