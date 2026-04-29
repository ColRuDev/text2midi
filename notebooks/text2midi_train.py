# %% [markdown]
# # Replica del entrenamiento del modelo text2midi
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ColRuDev/text2midi/blob/main/notebooks/text2midi_train.ipynb)
#
# En este notebook se replica todo el proceso de entrenamiento presentado en el paper [Text2midi: Generating Symbolic Music from Captions](http://arxiv.org/abs/2412.16526).
#
# El modelo se puede encontrar perfectamente en hugging face en el [siguiente enlace](https://huggingface.co/amaai-lab/text2midi)

# %%
# Global Variables
DEVICE = "cuda" # Choose "cuda" or "cpu" based on your hardware capabilities
DATASET_PATH = "../datasets"
MAX_POSITION_EMBEDDINGS = 1024
DECODER_LAYERS = 12
DECODER_HEADS = 8
DECODER_HIDDEN_SIZE = 512
DECODER_INTERMEDIATE_SIZE = 1024
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

# %% [markdown]
# ## Preparación del dataset SymphonyNet
#
# Primero se realiza el preprocesamiento del dataset usado para el pre entrenamiento. Se utiliza la librería music 21 para extraer los atributos de tempo, tonalidad, bpm e instrumentos de los archivos midi. Estos atributos sirven como placeholder de 10 frases diferentes, para simplificar usaremos una única plantilla.

# %% [markdown]
# ### Prueba de music 21
#
# Primero se prueba que music 21 lea correctamente uno de los archivos midi antes de crear pseudo captions.

# %%
# Importar librería music21
from music21 import converter, tempo, key
import os

# Definir la ruta del archivo MIDI
midi_path = "../datasets/SymphonyNet_Dataset/classical/altnikol_befiehl_du_deine_wege_(c)icking-archive.mid"

# Verificar que el archivo existe
if os.path.exists(midi_path):
    print(f"Archivo encontrado: {midi_path}")
else:
    print(f"Archivo NO encontrado: {midi_path}")

# Cargar el archivo MIDI
score = converter.parse(midi_path)
print("\nArchivo MIDI cargado correctamente")
print(f"Tipo de objeto: {type(score)}")

# %% [markdown]
# Ahora que sabemos que el archivo se leyó correctamente. Extraemos los atributos requeridos para el pre entrenamiento.

# %%
# Mostrar información básica del score
print("=" * 60)
print("INFORMACIÓN BÁSICA DEL ARCHIVO MIDI")
print("=" * 60)
print(f"\nNúmero de partes (tracks): {len(score.parts)}")
print(f"Duración total: {score.quarterLength} quarter notes")
print(f"Duración en segundos: {score.seconds:.2f} segundos")

# %%
# Crear un resumen estructurado de los atributos extraídos
print("=" * 60)
print("RESUMEN DE ATRIBUTOS EXTRAÍDOS")
print("=" * 60)

# Función auxiliar para extraer atributos de forma robusta
def extract_musical_attributes(score):
    attributes = {
        'tempo_bpm': None,
        'key': None,
        'key_mode': None,
        'time_signature': None,
        'instruments': set(),
        'num_tracks': len(score.parts),
        'duration_seconds': round(score.seconds, 2),
        'num_notes': len(score.flatten().notes)
    }

    # Tempo
    tempo_marks = score.flatten().getElementsByClass(tempo.MetronomeMark)
    if tempo_marks:
        attributes['tempo_bpm'] = tempo_marks[0].number
    else:
        try:
            estimated_tempo = score.flatten().metronomeMarkBoundaries()[0][-1]
            attributes['tempo_bpm'] = estimated_tempo.number
        except:
            pass

    # Tonalidad
    key_sigs = score.flatten().getElementsByClass(key.Key)
    if key_sigs:
        attributes['key'] = key_sigs[0].tonic.name
        attributes['key_mode'] = key_sigs[0].mode
    else:
        try:
            analyzed_key = score.analyze('key')
            attributes['key'] = analyzed_key.tonic.name
            attributes['key_mode'] = analyzed_key.mode
        except:
            pass

    # Compás
    time_sigs = score.flatten().getElementsByClass('TimeSignature')
    if time_sigs:
        attributes['time_signature'] = f"{time_sigs[0].numerator}/{time_sigs[0].denominator}"

    # Instrumentos
    for instrument in score.getInstruments() :
        if instrument:
            attributes['instruments'].add(instrument.bestName())

    attributes['instruments'] = ', '.join(attributes['instruments']) if attributes['instruments'] else None

    return attributes

# Extraer y mostrar atributos
attributes = extract_musical_attributes(score)

print("\nAtributos extraídos:")
for k, value in attributes.items():
    print(f"  {k}: {value}")

print("\n" + "=" * 60)

# %%
PSEUDO_TEMPLATE = f"A musical piece in {{key}} {{key_mode}} key with a tempo of {{tempo_bpm}} BPM, time signature of {{time_signature}}, featuring instruments: {{instruments}}."

# %%
print(PSEUDO_TEMPLATE.format(**attributes))

# %% [markdown]
# Vemos que usando *music21* se obtiene los atributos musicales necesarios para generar las pseudo captions. Esta función se usara más adelante tras iterar por todos los archivos y generar un dataframe con los atributos **location** y **caption** para mantener los nombres del MidiCaps dataset.

# %% [markdown]
# ### Generar el Dataframe para SymphonyNetDataset
#
# En esta sección se genera el dataframe que sera usado como pre entrenamiento del modelo

# %%
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def generate_symphonynet_dataset():
    """
    Recorre todos los archivos MIDI en el dataset SymphonyNet, extrae atributos musicales
    y genera un DataFrame con ubicaciones y captions.

    Returns:
        pd.DataFrame: DataFrame con columnas 'location' y 'caption'
    """
    # Inicializar DataFrame
    df = pd.DataFrame(columns=['location', 'caption'])

    # Ruta base del dataset
    dataset_path = Path("../datasets/SymphonyNet_Dataset")

    # Buscar todos los archivos MIDI
    midi_files = list(dataset_path.rglob("*.mid")) + list(dataset_path.rglob("*.midi"))

    percentage = 0.01

    print(f"Se encontraron {len(midi_files)} archivos MIDI")
    print(f"usando {percentage*100}% de los archivos")
    percentage_midi_files = midi_files[:int(len(midi_files)*percentage)]
    print("Procesando archivos...")

    # Lista para almacenar los datos temporalmente
    data_rows = []
    errors = []

    # Procesar cada archivo MIDI
    for midi_file in tqdm(percentage_midi_files, desc="Procesando archivos MIDI"):
        try:
            # Cargar el archivo MIDI
            score = converter.parse(str(midi_file))

            # Extraer atributos musicales
            attributes = extract_musical_attributes(score)

            # Generar caption usando la plantilla
            # Manejar valores None en los atributos
            safe_attributes = {
                'key': attributes.get('key') or 'Unknown',
                'key_mode': attributes.get('key_mode') or 'unknown',
                'tempo_bpm': attributes.get('tempo_bpm') or 'Unknown',
                'time_signature': attributes.get('time_signature') or 'Unknown',
                'instruments': attributes.get('instruments') or 'Unknown'
            }

            caption = PSEUDO_TEMPLATE.format(**safe_attributes)

            # Agregar fila al DataFrame
            data_rows.append({
                'location': str(midi_file.relative_to(dataset_path.parent)),
                'caption': caption
            })

        except Exception as e:
            errors.append((str(midi_file), str(e)))
            print(f"\nError procesando {midi_file.name}: {str(e)}")
            continue

    # Crear DataFrame a partir de la lista de filas
    df = pd.DataFrame(data_rows)

    # Guardar el DataFrame como CSV
    output_path = dataset_path / "symphonynet_captions.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n{'='*60}")
    print(f"Procesamiento completado!")
    print(f"{'='*60}")
    print(f"Total de archivos procesados exitosamente: {len(df)}")
    print(f"Total de errores: {len(errors)}")
    print(f"Archivo CSV guardado en: {output_path}")
    print(f"{'='*60}")

    if errors:
        print(f"\nPrimeros 5 errores:")
        for i, (file, error) in enumerate(errors[:5]):
            print(f"  {i+1}. {Path(file).name}: {error}")

    return df

# Ejecutar la función
df_symphonynet = generate_symphonynet_dataset()


# %%
# Mostrar las primeras filas del DataFrame generado
print("Primeras 5 filas del dataset:")
display(df_symphonynet.head())

print("\nÚltimas 5 filas del dataset:")
display(df_symphonynet.tail())

print("\nInformación del dataset:")
display(df_symphonynet.info())


# %% [markdown]
# En este caso se toma el **1%** del dataset para crear el dataframe. Esto con el objetivo de agilizar el experimento.
#
# El dataframe se guarda en un archivo csv para poder usar más adelante sin tener que ejecutar nuevamente esta sección.

# %% [markdown]
# ## Preparación del modelo
#
# En esta sección se prueba todo el código requerido para completar una epoca de entrenamiento del modelo.

# %% [markdown]
# ### Crear el codificador
#
# Ahora se crea el codificador usado en el modelo. Siguiendo la arquitectura presentado en el paper de Text2Midi, se usa [FlanT5](https://huggingface.co/google/flan-t5-base).

# %%
from transformers import T5Tokenizer, T5EncoderModel

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

## Load the pre-trained FLAN T5 encoder and freeze its parameters
flan_t5_encoder = T5EncoderModel.from_pretrained("google/flan-t5-small", device_map="auto")
for param in flan_t5_encoder.parameters():
    param.requires_grad = False

input_text = "A musical piece in C major key with a tempo of 120 BPM"
input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(DEVICE)

encoder_outputs = flan_t5_encoder(input_ids)
print(f"Encoder working properly! Output shape: {encoder_outputs.last_hidden_state.shape} (batch_size, seq_len, hidden_dim)")

# %% [markdown]
# ### Crear el tokenizador midi
#
# Ahora se crea y valida el tokenizador REMI+. Con esto, el modelo puede procesar correctamente el archivo y las captions.

# %%
from miditok import REMI, TokenizerConfig

config = TokenizerConfig(
    use_programs=True,
    use_time_signatures=True,
    one_token_stream_for_programs=True
)

remi_tokenizer = REMI(config)

midi_file_path = "../datasets/SymphonyNet_Dataset/classical/altnikol_befiehl_du_deine_wege_(c)icking-archive.mid"

midi_tokens = remi_tokenizer(midi_file_path)

print(f"Tokenización correcta! Número de tokens: {len(midi_tokens)}")

# %% [markdown]
# ### Prueba del encoder y los tokenizers
#
# Ahora se prueba que tanto el codificador como el tokenizador se integren adecuadamente con el dataset. De esta forma nos aseguramos que se extraiga correctamente la ubicación del archivo y las captions

# %%
import pandas as pd

pretrain_df = pd.read_csv("../datasets/SymphonyNet_Dataset/symphonynet_captions.csv")
pretrain_df.__len__()

# %% [markdown]
# Ahora que se cargo el dataframe correctamente, se crea una función que retorne la ubicación del archivo midi y su caption respectivo

# %%
import os

def get_midi_and_caption(index):
    midi_path = os.path.join(DATASET_PATH, pretrain_df.iloc[index]['location'])
    caption = pretrain_df.iloc[index]['caption']
    return midi_path, caption

midi_path, caption = get_midi_and_caption(0)
print(f"MIDI Path: {midi_path}")
print(f"Caption: {caption}")

# %% [markdown]
# Ahora se crea la función que recibe la ubicación del archivo midi y las captions, genera los tokens y los procesa con el encoder

# %%
from torch import tensor, int64
from torch.nn import functional as F

def encode_midi_and_caption(midi_path, caption):
    # Tokenizar y codificar la caption
    input_ids = tokenizer(caption, return_tensors="pt", padding=True, truncation=True).input_ids.to(DEVICE)
    encoder_outputs = flan_t5_encoder(input_ids)

    # Tokenizar el archivo MIDI
    midi_tokens = remi_tokenizer(midi_path)

    # Convertir los tokens MIDI a tensores
    # ![Importante] asegurarse de que el tensor no sobrepase el limite de tamaño del decoder
    if len(midi_tokens) < 1024:
        labels = F.pad(tensor(midi_tokens), (0, 1024 - len(midi_tokens))).to(int64).unsqueeze(0).to(DEVICE)
    else:
        labels = tensor(midi_tokens[:1024]).to(int64).unsqueeze(0).to(DEVICE)

    return encoder_outputs.last_hidden_state, labels

encoder_outputs, labels = encode_midi_and_caption(midi_path, caption)
print(f"Encoder Output Shape: {encoder_outputs.shape} (batch_size, seq_len, hidden_dim)")
print(f"Labels Shape: {labels.shape} (batch_size, seq_len)")


# %% [markdown]
# ### Crear el decoder
#
# Ahora que verificamos que el dataset se tokeniza y codifica correctamente, se crea el decoder que procesa el resultado del encoder y el midi tokenizado.

# %%
from transformers import BertConfig, BertLMHeadModel

config_decoder = BertConfig()
config_decoder.vocab_size = remi_tokenizer.vocab_size + 1
config_decoder.max_position_embeddings = 1024 # 2048 in the paper
config_decoder.max_length = 1024
config_decoder.bos_token_id = remi_tokenizer["BOS_None"] # type: ignore
config_decoder.eos_token_id = remi_tokenizer["EOS_None"] # type: ignore
config_decoder.pad_token_id = 0
config_decoder.num_hidden_layers = 12 # 18 in the paper
config_decoder.num_attention_heads = 8 # 8 in the paper
config_decoder.hidden_size = 512 # 768 in the paper
config_decoder.intermediate_size = 1024 # 1024 in the paper
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True
config_decoder.tie_encoder_decoder = False
config_decoder.tie_word_embeddings = False

decoder_model = BertLMHeadModel(config_decoder).to(DEVICE)


# %%
def decode_midi_and_caption(encoder_outputs, remi_tokens):
    outputs = decoder_model(
        input_ids = remi_tokens,
        encoder_hidden_states = encoder_outputs,
        labels = remi_tokens
    )
    loss = outputs.loss
    logits = outputs.logits
    return {"loss": loss, "logits": logits}

output_epoch = decode_midi_and_caption(encoder_outputs, labels)
print(f"Output Epoch Loss: {output_epoch['loss']}")
print(f"Output Epoch Logits Shape: {output_epoch['logits'].shape} (batch_size, seq_len, vocab_size)")

# %% [markdown]
# ## Entrenamiento
#
# Ahora que sabemos que cada bloque del modelo funciona correctamente, creamos todo el código necesario para el entrenamiento completo del modelo

# %% [markdown]
# ### Crear el dataloader
#
# Aquí se crea el dataloader que va a extraer el archivo de la ruta en el dataset y los captions. Luego los va a tokenizar y devolver el tensor listo para que el modelo lo reciba.

# %%
import os
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch import tensor, int64




class Text2MIDI_Dataset(Dataset):
    def __init__(self, dataframe, caption_tokenizer, remi_tokenizer, decoder_max_sequence_length: int=1024):
        self.dataframe = dataframe
        self.caption_tokenizer = caption_tokenizer
        self.remi_tokenizer = remi_tokenizer
        self.decoder_max_sequence_length = decoder_max_sequence_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        midi_path = os.path.join(DATASET_PATH, self.dataframe.iloc[idx]['location'])
        caption = self.dataframe.iloc[idx]['caption']

        tokens = self.remi_tokenizer(midi_path)

        if len(tokens.ids) == 0: # type: ignore
            midi_tokens = [self.remi_tokenizer["BOS_None"], self.remi_tokenizer["EOS_None"]]
        else:
            midi_tokens = [self.remi_tokenizer["BOS_None"]] + tokens.ids + [self.remi_tokenizer["EOS_None"]] # type: ignore

        caption_tokens = self.caption_tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        caption_input_ids = caption_tokens["input_ids"]
        caption_attention_mask = caption_tokens["attention_mask"]

        if len(midi_tokens) < self.decoder_max_sequence_length:
            labels = F.pad(tensor(midi_tokens), (0, self.decoder_max_sequence_length - len(midi_tokens))).to(int64).unsqueeze(0).to(DEVICE)
        else:
            labels = tensor(midi_tokens[:self.decoder_max_sequence_length]).to(int64).unsqueeze(0).to(DEVICE)

        return caption_input_ids, caption_attention_mask, labels


# %%
from transformers import T5Tokenizer
from miditok import REMI, TokenizerConfig

config = TokenizerConfig(
    use_programs=True,
    use_time_signatures=True,
    one_token_stream_for_programs=True
)

caption_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
remi_tokenizer = REMI(config)

# %%
import pandas as pd

pretrain_df = pd.read_csv("../datasets/SymphonyNet_Dataset/symphonynet_captions.csv")
pretrain_df.__len__()

# %%
pretrain_dataset = Text2MIDI_Dataset(pretrain_df, caption_tokenizer, remi_tokenizer,MAX_POSITION_EMBEDDINGS)
print(f"Dataset creado correctamente! Número de muestras: {len(pretrain_dataset)})")
a,b,c = pretrain_dataset[0]
print(type(a), a.shape)
print(type(b), b.shape)
print(type(c), c.shape)

# %% [markdown]
# Ya comprobamos que el dataset se crea correctamente y retorna los tensores esperados.

# %% [markdown]
# ### Crear arquitectura del modelo
#
# Continuamos creando nuestro modelo codificador decodificador personalizado.

# %%
from transformers import PreTrainedModel

class Text2MIDI_Model(PreTrainedModel):
    def __init__(self, encoder, decoder, encoder_config, decoder_config):
        super().__init__(encoder_config)
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

    def forward(self, encoder_input_ids, decoder_input_ids, encoder_attention_mask=None, decoder_attention_mask=None, labels=None, **kwargs):
        encoder_outputs = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels
        )

        logits = decoder_outputs.logits
        loss = decoder_outputs.loss

        return {"loss": loss, "logits": logits}


# %%
from transformers import T5EncoderModel, T5Config, BertConfig, BertLMHeadModel

flan_t5_encoder = T5EncoderModel.from_pretrained("google/flan-t5-small", device_map="auto")
for param in flan_t5_encoder.parameters():
    param.requires_grad = False

encoder_config = T5Config.from_pretrained("google/flan-t5-small")

decoder_config = BertConfig()
decoder_config.vocab_size = remi_tokenizer.vocab_size + 1
decoder_config.max_position_embeddings = MAX_POSITION_EMBEDDINGS
decoder_config.max_length = MAX_POSITION_EMBEDDINGS
decoder_config.bos_token_id = remi_tokenizer["BOS_None"] # type: ignore
decoder_config.eos_token_id = remi_tokenizer["EOS_None"] # type: ignore
decoder_config.pad_token_id = 0
decoder_config.num_hidden_layers = DECODER_LAYERS
decoder_config.num_attention_heads = DECODER_HEADS
decoder_config.hidden_size = DECODER_HIDDEN_SIZE
decoder_config.intermediate_size = DECODER_INTERMEDIATE_SIZE
decoder_config.is_decoder = True
decoder_config.add_cross_attention = True
decoder_config.tie_encoder_decoder = False
decoder_config.tie_word_embeddings = False
decoder_model = BertLMHeadModel(decoder_config)

# %%
custom_model = Text2MIDI_Model(
    encoder=flan_t5_encoder,
    decoder=decoder_model,
    encoder_config=encoder_config,
    decoder_config=decoder_config
)

# %% [markdown]
# ### Crear las funciones para procesar las metricas y los logits

# %%
from evaluate import load as load_metric
from torch import Tensor, argmax

metrics = {metric: load_metric(metric) for metric in ["accuracy"]}

def compute_metrics(eval_pred):
    """
    Compute metrics for pretraining.

    Must use preprocess_logits function that converts logits to predictions (argmax or sampling).

    :param eval_pred: EvalPrediction containing predictions and labels
    :return: metrics
    """
    predictions, labels = eval_pred
    not_pad_mask = labels != 0
    labels, predictions = labels[not_pad_mask], predictions[not_pad_mask]
    return metrics["accuracy"].compute(predictions=predictions.flatten(), references=labels.flatten())

def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
    """
    Preprocess the logits before accumulating them during evaluation.

    This allows to significantly reduce the memory usage and make the training tractable.
    """
    pred_ids = argmax(logits, dim=-1)  # long dtype
    return pred_ids


# %%
from torch import nn
def collate_fn(batch):
    """
    Collate function for the DataLoader
    :param batch: The batch
    :return: The collated batch
    """
    input_ids = [item[0].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = [item[1].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = [item[2].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    decoder_input_ids = labels[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    # return input_ids, attention_mask, labels
    return {
        'encoder_input_ids': input_ids,
        'encoder_attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'labels': labels
    }


# %% [markdown]
# ### Crear el trainer

# %%
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    def get_eval_dataloader(self, eval_dataset):
        return DataLoader(eval_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    def get_test_dataloader(self, test_dataset):
        return DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

training_args = TrainingArguments(
    output_dir=os.path.join("artifacts", "text2midi_model"),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_strategy="epoch",  # "steps" or "epoch"
    save_total_limit=1,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.3,
    max_grad_norm=3.0,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    # gradient_accumulation_steps=configs['training']['text2midi_model']['gradient_accumulation_steps'],
    # gradient_checkpointing=True,
    optim="adafactor",
    seed=444,
    logging_strategy="steps",
    logging_steps=10,
    load_best_model_at_end=True,
    # metric_for_best_model="loss",
    greater_is_better=False,
    report_to="tensorboard",
    run_name="text2midi_model",
    push_to_hub=False,
    dataloader_num_workers=5
)

# %%
trainer = CustomTrainer(
    model=custom_model,
    args=training_args,
    train_dataset=pretrain_dataset,
    eval_dataset=pretrain_dataset,
    compute_metrics=compute_metrics, # type: ignore
    preprocess_logits_for_metrics=preprocess_logits
)

# %%
# Train and save the model
train_result = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

# %% [markdown]
# Con esto validamos que el entrenamiento funciona correctamente. y una época tarda casi 4 minutos usando el 1% del dataset de pre entrenamiento. Este experimento se realizo sobre una **RTX 3050 4GB**.
#
# ### Datos de interés
#
# El modelo usa 3GB de VRAM para entrenar el modelo con las siguientes configuraciones:
#
# - **MAX_POSITION_EMBEDDINGS** de 1024
# - **DECODER_LAYERS** de 12
# - **DECODER_HEADS** de 8
# - **DECODER_HIDDEN_SIZE** de 512
# - **DECODER_INTERMEDIATE_SIZE** de 1024
# - **BATCH_SIZE** de 4
# - **EPOCHS** de 1
# - **LEARNING_RATE** de 5e-5
# - **WEIGHT_DECAY** de 0.01
