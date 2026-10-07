# La CNN del proyecto, explicada paso a paso

Este documento explica **qué hace la red, por qué está armada así y cómo se entrena**,
siguiendo el código de [`kenken/cnn.py`](../kenken/cnn.py) y [`kenken/ocr.py`](../kenken/ocr.py).
La idea es que puedan defenderla en el informe y en el video.

---

## 1. ¿Qué problema resuelve?

Cada jaula tiene una etiqueta como `12×`, `3−`, `2÷` o `5`. La visión (hito 3) ya sabe
**dónde** está cada jaula; falta **leer** la etiqueta. Se divide en dos partes:

1. **Segmentar**: cortar la etiqueta en caracteres sueltos (`1`, `2`, `×`). Esto es visión
   clásica con OpenCV: no hace falta aprender nada.
2. **Clasificar** cada carácter en una de **14 clases**: `0 1 2 3 4 5 6 7 8 9 + - × ÷`.
   Esto es lo que hace la CNN.

`x` y `×` son la **misma clase** (multiplicación), igual que `/` y `÷` (división), porque
distintos KenKen usan distintos símbolos para la misma operación.

## 2. La entrada: un glifo de 32×32

`ocr.normalize_glyph()` convierte cada carácter en una imagen de **32×32 píxeles**:

- Valores entre 0 y 1, donde **1 = tinta y 0 = papel**. No se usa el gris crudo: se usa la
  "oscuridad relativa al papel" de ese recorte, para que una foto oscura o con sombra dé lo
  mismo que una captura limpia.
- El carácter se escala para que su **lado mayor mida 24 px**, **manteniendo la proporción**,
  y se centra. Así un `-` sigue siendo ancho y bajo y un `1` sigue siendo alto y angosto.
  Esa forma ayuda a distinguirlos.

## 3. La arquitectura

```
entrada                                   1 × 32 × 32
─ bloque 1 ─ conv3×3(32) → BN → ReLU
             conv3×3(32) → BN → ReLU
             maxpool 2×2                   32 × 16 × 16
─ bloque 2 ─ conv3×3(64) → BN → ReLU
             conv3×3(64) → BN → ReLU
             maxpool 2×2                   64 ×  8 ×  8
─ bloque 3 ─ conv3×3(128) → BN → ReLU
             maxpool 2×2                  128 ×  4 ×  4  = 2048 números
─ clasificador ─ dropout(0.3) → lineal 2048→128 → ReLU → dropout(0.3) → lineal 128→14
salida                                    14 puntajes (logits)
```

Tiene unos **403 000 parámetros** (pesos), de los que unos 262 000 están en la primera capa
lineal. Es una red pequeña: entrena en minutos en CPU.

### Qué hace cada pieza

| Pieza | Qué hace | Por qué la usamos |
|---|---|---|
| **Convolución 3×3** | Un filtro de 3×3 pesos recorre la imagen y en cada posición calcula una suma ponderada de los 9 píxeles vecinos. Un filtro "responde" fuerte donde encuentra su patrón (un borde vertical, una curva...). Una capa con 32 filtros produce 32 "mapas" de respuesta. | Los dígitos se reconocen por patrones **locales**, y el mismo patrón puede aparecer en cualquier lugar: el filtro se comparte en toda la imagen, así que aprende con pocos pesos. |
| **Apilar convoluciones** | La 2.ª capa combina los mapas de la 1.ª, la 3.ª los de la 2.ª... | Jerarquía: bordes → trazos → partes de símbolos (el lazo del 6, el cruce de la ×, los puntos del ÷). |
| **BatchNorm (BN)** | Normaliza cada canal (media 0 y varianza 1 dentro del lote), y luego aprende una escala y un desplazamiento. | El entrenamiento es más estable y rápido, y tolera tasas de aprendizaje más altas. |
| **ReLU** | `max(0, x)`. | Es la no linealidad: sin ella, toda la red sería una sola transformación lineal. |
| **MaxPool 2×2** | De cada bloque de 2×2 se queda con el máximo; el tamaño se reduce a la mitad. | Menos cómputo y **tolerancia a pequeños desplazamientos**: si el trazo se corre 1 px, el máximo casi no cambia. |
| **Dropout(0.3)** | Durante el entrenamiento apaga al azar el 30 % de las neuronas en cada paso. En evaluación no apaga nada. | Evita que la red dependa de unas pocas neuronas, lo que **reduce el sobreajuste**. |
| **Lineal final** | Combina las 2048 características en 14 números (logits). | Es el clasificador propiamente dicho. |

### De puntajes a probabilidades: softmax

Los 14 logits `z₁ … z₁₄` se convierten en probabilidades con

```
p_k = exp(z_k) / Σ_j exp(z_j)
```

Todas quedan entre 0 y 1 y suman 1. En el código usamos directamente
`log_softmax` (log-probabilidades), porque sumar logaritmos es más estable que multiplicar
probabilidades pequeñas.

## 4. ¿Por qué nos importan las probabilidades y no solo "la clase ganadora"?

Ahí está la gracia del proyecto (la idea del ejemplo de KU Leuven):

1. **Lectura de una etiqueta completa** (`ocr.decode_readings`): la probabilidad de leer
   `12×` es `p(1) · p(2) · p(×)` (suma de log-probs). Además se aplica la **gramática** de
   KenKen:
   - jaula de 1 celda → solo dígitos;
   - jaula de 2 celdas → dígitos y una operación de `+ − × ÷`;
   - jaula de 3 o más celdas → la operación solo puede ser `+` o `×`;
   - sin ceros a la izquierda, y el objetivo tiene que ser alcanzable (un `÷` de un 6×6 no
     puede valer 8).

   Con eso se obtienen las **k = 5 lecturas más probables** de cada etiqueta, no solo la mejor.
   Como la segmentación a veces falla (por ejemplo, un `1-` impreso pegado llega como un
   solo trozo), también se prueban **segmentaciones alternativas** con una sola corrección:
   partir un trozo en dos o unir dos vecinos. Cada alternativa paga una penalización de 3 en
   log-probabilidad (`ocr.SPLIT_PENALTY`), así que solo gana si la CNN y la gramática la
   prefieren con claridad. En el set normal, esto subió los tableros leídos completos del
   71 % al 81 %.
2. **Inferencia conjunta** (hito 6): el modelo CP recibe esas 5 alternativas por jaula y
   elige la combinación **más probable que además tenga solución**. Si la CNN confundió
   un `3` con un `8`, la lectura con `8` probablemente deja el puzzle sin solución y el
   solver elige la siguiente: la de `3`.

Por eso en la evaluación medimos también el **top-k**: si la lectura correcta está entre
las 5 candidatas, el modelo CP la puede recuperar.

## 5. El entrenamiento

### Los datos (todos sintéticos y con etiqueta conocida)

| Fuente | Cómo se genera | Para qué sirve |
|---|---|---|
| (a) Etiquetas sueltas | `render.render_label_crop()`: texto al azar (las 5 operaciones salen con la misma frecuencia), con una de las 15 fuentes, baja resolución, desenfoque, ruido, JPEG, contraste e iluminación variables, y una leve rotación o cizalla. | Mucha variedad y **clases balanceadas** (en los tableros, `÷` y `−` son raros). |
| (b) Recortes de tableros | Se renderizan 400 tableros completos (normales y difíciles) y se pasan por **toda** la visión: detección, rectificación, grilla y recorte. | Los glifos se ven **exactamente** como los verá el pipeline. |

Las dos fuentes usan la **misma** función de recorte y segmentación que el pipeline
(`label_from_gray` + `segment_glyphs` + `normalize_glyph`). Solo se guardan las etiquetas
en que la segmentación dio tantos trozos como caracteres tiene el texto, así cada glifo
tiene su clase segura.

La evaluación se hace sobre `dataset/synthetic` y `dataset/synthetic_hard`, que usan
**otras semillas**: la red nunca vio esas imágenes.

### El ciclo de entrenamiento (`cnn.train`)

```
para cada época:
    mezclar los índices de entrenamiento
    para cada lote de 128 glifos:
        x = aumentar(lote)                 # rotación ±8°, escala, traslación, ruido
        logits = modelo(x)                 # pasada hacia adelante
        pérdida = entropía_cruzada(logits, clases)
        pérdida.backward()                 # gradientes por retropropagación
        optimizador.step()                 # Adam ajusta los pesos
    medir exactitud en validación (10 % separado) y guardar el mejor modelo
```

- **Entropía cruzada**: `pérdida = −log(p de la clase correcta)`. Si la red da 0.9 a la
  clase correcta, la pérdida es 0.105; si da 0.01, es 4.6. Castiga mucho estar seguro
  de algo incorrecto.
- **Retropropagación**: calcula cuánto cambia la pérdida si se mueve un poco cada peso
  (el gradiente). PyTorch lo hace solo con `loss.backward()`.
- **Adam**: descenso de gradiente con un paso adaptado a cada peso. La tasa de aprendizaje
  sube y luego baja durante el entrenamiento (OneCycle): pasos grandes al principio y
  ajuste fino al final.
- **Aumentos en línea**: cada época ve una versión distinta de cada glifo. La red aprende
  que un `7` un poco girado o corrido sigue siendo un `7`.
- **Validación**: el 10 % de los glifos no se usa para entrenar. Si la exactitud de
  entrenamiento sube pero la de validación no, hay sobreajuste. Se guarda la época con
  mejor validación.

## 6. Cómo repetirlo

```bash
python -m kenken.cnn                                   # genera los datos, entrena y guarda models/ocr_cnn.pt
python -m kenken.evaluate --ocr dataset/synthetic results/ocr_synthetic.csv
python -m kenken.evaluate --ocr dataset/synthetic_hard results/ocr_synthetic_hard.csv
```

## 7. Preguntas que les pueden hacer (y la respuesta corta)

- **¿Por qué no MNIST?** MNIST son dígitos escritos a mano, sin `+ − × ÷`, y en otro estilo.
  Nuestras etiquetas son tipografía impresa, así que generamos datos con fuentes reales.
- **¿Por qué una CNN y no un perceptrón multicapa?** La convolución comparte pesos y
  aprovecha que el patrón es local e invariante a la posición: con menos parámetros
  generaliza mejor.
- **¿Por qué 32×32?** Es lo bastante grande para distinguir `6/8/9` o `+/×`, y lo bastante
  pequeño para entrenar rápido en CPU.
- **¿Qué limita la exactitud?** Más la segmentación que la CNN: si dos caracteres vienen
  pegados, la red recibe un glifo "imposible". Por eso el decodificador y la inferencia
  conjunta trabajan con alternativas.
- **¿Qué es Tesseract y por qué no lo usamos?** Es un OCR general de documentos. Nuestro
  caso son muy pocos caracteres, pequeños y en una gramática muy restringida: una red
  propia, entrenada para esto y que devuelve probabilidades, encaja mejor con la inferencia
  conjunta.
