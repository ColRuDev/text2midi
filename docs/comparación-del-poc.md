# Comparación de resultados

## Prompt

A melodic electronic song with ambient elements, featuring piano, acoustic guitar, alto saxophone, string ensemble, and electric bass. Set in G minor with a 4/4 time signature, it moves at a lively Presto tempo. The composition evokes a blend of relaxation and darkness, with hints of happiness and a meditative quality.

## Componentes musicales esperados

- Velocidad rápida, al menos 168 bpms
- Compás 4/4
- Tonalidad de Sol menor
- Ambiente relajante y oscuro, con toques de felicidad pero sin sonar alegre.

### Intrumentos

- Piano
- Guitarra acustica
- Bajo electrico
- Saxo alto
- Cuerdas

## Resultados del modelo Text2Midi

- Velocidad de 121 bpms
- Compás 4/4
- Tono de Fa mayor
- Ambiente alegre y movido

### Instrumentos

- Piano
- bajo electrico
- sintetizador y clarinete con una nota cada uno
- Faltaron la guitarra, saxo alto y cuerdas

## Resultados del modelo Midi LLM

- Velocidad de 120 bpms
- Compás 4/4
- Tonalidad de Re Mayor
- Ambiente muy alegre, sin relajación, se siente el bucle

### Instrumentos

- Piano
- Guitarra
- Bajo electrico
- Saxo alto
- Agrega guitarra electrica y percusión
- Faltaron las cuerdas

## Resultados de Suno

Regenera el prompt en este caso usando “folk, Melodic electronic with ambient wash; pulsing side-chained pads under a steady four-on-the-floor kick and warm electric bass. Piano carries the main motif in G minor, doubled by soft string ensemble; acoustic guitar adds bright arpeggios and off-beat chords. Alto sax weaves lyrical top-lines with tasteful reverb tails. Lively Presto tempo but relaxed phrasing; subtle risers and filtered noise for transitions, with a meditative breakdown in the middle before a final, uplifting reprise., electric, saxophone, electronic, ambient, acoustic”

No me queda fácil saber el tiempo y la métrica aunque sí parece ser 4/4. La tonalidad tampoco se muestra.

Detectó piano, saxo, guitarra, bajo y no estoy seguro de si hay cuerdas. Percibo más instrumentos

El ambiente se siente relajante y  meditativo, con algo de felicidad y los toques de oscuridad son bastante ligeros. En términos de ambiente es el que más se acerca

## Conclusiones

1. Los modelos de generación simbólicos no pueden respetar tempo, tonalidad e instrumentación total indicada en la instrucción inicial
2. Midi LLM es el que más se acerca a los instrumentos solicitados. Ambos modelos agregan instrumentos adicionales
3. Los modelos simbólicos le dan más prioridad a sentirse alegre. Un ambiente complejo como el solicitado con relajación toques oscuros y alegres resultan en omisión de parte del ambiente.
4. La precisión técnica es difícil de determinar para Suno.
5. Suno regenera el prompt para ser más descriptivo bajo sus criterios
6. Suno es muy bueno para cumplir con el ambiente solicitado.
