# Evaluación de Descripción Natural

## Prompts Utilizados

### Intención natural

A solitary old man captures stars from the firmament while resting upon the curve of a great crescent moon. With his fishing rod and a bucket overflowing with light, he seems to gather fragments of hope amidst the vastness of the sky.

The ambiance consists of a deep, dark sky where the scene is illuminated by the glints of stars and the warm, dim light of a lonely hanging lantern.

The perceived emotion is one of soft nostalgia and a serene melancholy that transforms into a feeling of pure optimism before the magic of the impossible.

### Formato basado en midicaps

A melodic cinematic ambient track with a magical and nostalgic vibe, featuring a solitary mellow piano, a deep string ensemble, and shimmering celesta representing starlight. The song is in the key of B minor with a 4/4 time signature and a slow Adagio tempo. The chord progression revolves around Bm, G, D, and A, creating a serene, melancholy atmosphere that gradually transforms into a feeling of pure optimism and impossible magic

## Lógica de evaluación

Partiendo de la metodología de Yamaha para enseñar composición, partimos de una imagen que se procede a describir en palabras naturales para extraer la intencionalidad de la composición. Esta descripción empieza por presentar la narrativa de la escena, luego la ambientación que rodea la escena y finalmente la emoción que se busca transmitir.

Lo anterior, proporciona la intención natural que queremos sea la base de la generación. Esta se usa directamente como entrada para los modelos Text2Midi y MidiLLM, evaluando la coherencia y la calidad de la salida generada. Adicionalmente, creamos un segundo prompt basado en la primera descripción, siguiendo la estructura presentada en el dataset Midicaps con el que se entrenó Text2Midi. De esta forma evaluamos si mejora el rendimiento de los modelos al usar el segundo prompt en lugar del primero.

Las composiciones generadas por los modelos se evaluaron en conjunto con una profesora de Yamaha para determinar su calidad y coherencia.

## Resultados

### Individuales

#### Text2Midi

- Se desempeña mejor con el prompt basado en Midicaps.
- Tiende a añadir instrumentos y le da todo el protagonismo a 1 o 2 instrumentos unicamente.
- En lugar de B menor, utiliza B bemol mayor.
- Usando solo la intención natural no sabe bien que hacer y termina usando muchos instrumentos.
- No parece seguir la progresión de acordes sugerida

#### MidiLLM

- Se desempeña mejor con la descripción natural.
- Tiende a usar notación muy compleja y replicar las mismas notas con los diferentes instrumentos. Que en algunos casos no es acorde con la forma de tocar dichos instrumentos.
- Composición más cerca a la intención emocional deseada.
- Hace buena elección de instrumentos y progresiones según la descripción natural. Con el error de usar incorrectamente los instrumentos.

### Gloables

- Formato extraño
- Se sugiere hacer una agrupación por familias de instrumentos (p.e. voces, percusión, cuerdas, etc.) y usar los registros respectivos de cada familia. ya sea mezclando o usando solo los registros de una familia en particular.
- Se advierte sobre los instrumentos transpositores que no suenan en la misma tonalidad en la que se escriben.
- Se recomienda mantenerse debajo de los 3 sostenidos o bemoles en la armaura de clave para facilitar la lectura.

## Conclusiones

Ambas técnicas presentan ventajas y desventajas, sin embargo, al traducir la descripción natural a teorica hay una mayor interpretabilidad de la composición y se le da menos libertad al modelo. Esto favorece tanto al entendimiento de la composición como a servir de paso para afinar resultados.

Resulta importante determinar que modelos son adecuados para esta estrategia de generación, de igual forma, a como usar un modelo de traducción intermedio que permite llevar una intención natural a una descripción teórica más precisa, la cual se puede usar como entrada para el modelo de generación musical.

## Referencias

- [Composición de Text2Midi base](../outputs/evaluación-intencion/text2midi_base.mid)
- [Composición de Text2Midi con formato midicaps](../outputs/evaluación-intencion/text2midi_midicaps_format.mid)
- [Composición de MidiLLM base](./outputs/evaluación-intencion/midillm_base.mid)
- [Composición de MidiLLM con formato midicaps](../outputs/evaluación-intencion/midillm_midicaps_format.mid)
