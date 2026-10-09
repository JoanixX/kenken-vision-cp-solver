# Análisis Profundo de Fallos en OCR, Vulnerabilidad de Caracteres y Comportamiento de la Inferencia Conjunta (MAP)

---

## 1. Resumen Ejecutivo y Hallazgos Principales

A partir de los experimentos cruzados sobre los tres conjuntos de datos del proyecto (**Sintético Normal** [300 imágenes], **Sintético Difícil** [150 imágenes] y **Challenge Stress** [50 imágenes]), se llevó a cabo una auditoría cuantitativa y teórica para responder tres preguntas fundamentales:
1. **¿Dónde y por qué falla el flujo de percepción visual?**
2. **¿Con qué caracteres y pares de clases ocurren las confusiones más críticas?**
3. **¿Cómo influye la Inferencia Conjunta (MAP) ante un OCR defectuoso: cuándo salva el problema y cuándo "empeora" la situación generando soluciones divergentes?**

### Resumen de Métricas Clave de la Auditoría:
* **El carácter más vulnerable:** El operador de división `/` es el cuello de botella del OCR, con un F1 de apenas **$70.87\%$** en el dataset de estrés y **$76.13\%$** en el dataset difícil (Recall de solo $66.18\%$). El $46\%$ de las divisiones no reconocidas se confunden con `+`.
* **Impacto Positivo de MAP:** En el dataset de estrés, donde la lectura directa Top-1 solo resuelve el **$4.0\%$** de los tableros, la Inferencia Conjunta eleva la tasa de resolución exacta al Ground Truth al **$58.0\%$** (un incremento neto de $+54.0$ puntos porcentuales).
* **Impacto Negativo de MAP (El Fenómeno de Divergencia):** En un **$32.0\%$** de los tableros de estrés, MAP encuentra una solución matemáticamente válida (un Cuadrado Latino perfecto), pero **divergente de la imagen original impresa** (pseudo-solución), debido a que la etiqueta real no figuraba en el Top-$k$ y el solver combinó hipótesis alternativas de bajo rango para satisfacer la grilla.

---

## 2. Anatomía de los Fallos del Flujo de Percepción (¿Dónde Falla?)

El pipeline de extracción y resolución presenta tres puntos de fricción secuenciales:

```mermaid
flowchart TD
    A["1. Binarización & Fondo<br/>(Otsu / Sombras)"] -->|Ink > 18% o colapso| B["Pérdida de trazos delgados o<br/>fusión de dígitos con bordes"]
    B --> C["2. Segmentación de Glifos<br/>(Bounding Boxes)"]
    C -->|Fusión horizontal o corte espurio| D["Número erróneo de glifos o<br/>clasificación de fragmentos"]
    D --> E["3. Clasificación CNN<br/>(Logits / Softmax)"]
    E -->|Confusión morfológica| F["Top-1 erróneo<br/>(o verdadero ausente de Top-k)"]
    F --> G{"4. Solver CP-SAT<br/>(Variante A vs C)"}
    G -->|Variante A (Top-1)| H["INFEASIBLE"]
    G -->|Variante C (MAP)| I{"¿Está el Ground Truth<br/>en Top-k?"}
    I -- "Sí" --> J["Rescate Exacto (GT)"]
    I -- "No" --> K["Divergencia (Pseudo-solución)<br/>o Infeasible"]
```

### 2.1. Fallo 1: Iluminación No Uniforme y Colapso de Otsu
* **Mecanismo:** En fotos con sombras proyectadas o viñeteado, el histograma bimodal global se distorsiona. El umbral de Otsu convencional tiende a fijarse demasiado alto en regiones oscuras, interpretando áreas de sombra como tinta (área de tinta $> 18\%$).
* **Efecto:** Los dígitos se engrosan artificialmente y se unen a las líneas divisorias de la cuadrícula, impidiendo calcular la altura de referencia `ref_h`.
* **Mitigación actual:** La normalización adaptativa por desenfoque gaussiano local (`gray / GaussianBlur(gray, 31) * 255`) rescató tableros como `syn_00060.jpg`, pero ante sombras extremas con contraste menor a 15 niveles de gris, la relación señal/ruido sigue siendo crítica.

### 2.2. Fallo 2: Fusión Horizontal y Fragmentación de Trazos
* **Mecanismo:** Cuando un número consta de dos dígitos (e.g. `14+`, `48*`) y la tipografía tiene un *kerning* estrecho, la proyección vertical de tinta no presenta valles con valor 0.
* **Efecto:** La jaula se segmenta como 2 componentes en lugar de 3 (e.g., `14` como un único bloque ancho y `+` por separado). La gramática no puede decodificar una etiqueta de dos caracteres para una jaula que requiere operador, provocando que la jaula sea marcada como `unread`.

### 2.3. Fallo 3: Supresión de Bordes y Mutilación de Operadores
* **Mecanismo:** La función `clean_border_lines` elimina componentes conectados pegados a los márgenes ($x \le 2$ o $x \ge w-4$).
* **Efecto colateral:** En fuentes donde el operador aritmético está pegado al margen derecho de la celda (especialmente el brazo derecho del `+` o la barra diagonal `/`), el filtro de borde puede podar parcialmente el operador, transformando un `+` en un trazo vertical similar al dígito `1`.

---

## 3. Análisis de Caracteres: Vulnerabilidad y Confusión de Clases

Evaluando el rendimiento sobre las matrices de confusión reales de los tres entornos:

### 3.1. Tabla Comparativa de Rendimiento por Clase (F1-Score %)

| Glifo | Rol | F1 Normal (300 imgs) | F1 Difícil (150 imgs) | F1 Stress (50 imgs) | Vulnerabilidad |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **`/`** | Operador | **95.21%** | **76.13%** | **70.87%** | 🔴 **Crítica (Peor clase)** |
| **`-`** | Operador | 98.93% | 85.60% | 85.29% | 🟠 **Alta** |
| **`+`** | Operador | 98.90% | 87.95% | 91.24% | 🟡 **Moderada** |
| **`*`** | Operador | 99.91% | 92.34% | 98.55% | 🟢 **Baja** |
| **`0`** | Dígito | 99.66% | 89.11% | 94.17% | 🟡 **Moderada** |
| **`1`** | Dígito | 99.85% | 91.96% | 95.58% | 🟢 **Baja** (Suele absorber falsos positivos) |
| **`2`** | Dígito | 99.71% | 94.21% | 97.14% | 🟢 **Baja** |
| **`3`** | Dígito | 100.00% | 93.56% | 93.72% | 🟢 **Baja** |
| **`4`** | Dígito | 99.76% | 94.39% | 98.35% | 🟢 **Baja** |
| **`5`** | Dígito | 100.00% | 90.09% | 97.65% | 🟡 **Moderada** |
| **`6`** | Dígito | 99.92% | 92.84% | 96.12% | 🟢 **Baja** |
| **`7`** | Dígito | 99.74% | 96.18% | 96.70% | 🟢 **Baja** |
| **`8`** | Dígito | 100.00% | 92.34% | 98.01% | 🟢 **Baja** |
| **`9`** | Dígito | 100.00% | 87.00% | 97.50% | 🟡 **Moderada** |

---

### 3.2. Los 5 Pares Más Confundidos y su Causa Física

```
1.  /  ➔  +  (52 ocurrencias totales acumuladas)
    Causa: La barra oblicua '/' al cruzarse con ruido o remanentes de cuadrícula
    desarrolla un pseudotrazo horizontal, haciendo que la CNN active los filtros ortogonales de '+'.

2.  +  ➔  /  (40 ocurrencias totales acumuladas)
    Causa: En recortes con iluminación deficiente o erosión morfológica, uno de los brazos
    del '+' se atenúa y la inclinación de la cámara proyecta la cruz como un trazo oblicuo.

3.  -  ➔  +  (26 ocurrencias totales acumuladas)
    Causa: Un remanente de línea vertical de la cuadrícula que toca el centro del guion '-'
    convierte la línea horizontal en una cruz '+'.

4.  2  ➔  1  (24 ocurrencias totales acumuladas)
    Causa: En celdas de alta densidad (8x8, 9x9), la resolución del dígito baja de 16x16 px.
    Las curvas superior e inferior del '2' pierden contraste y solo el trazo diagonal/vertical
    queda activo, activando la neurona de '1'.

5.  *  ➔  +  (18 ocurrencias totales acumuladas)
    Causa: El glifo '×' de multiplicación difiere de '+' únicamente por una rotación de 45°.
    Ante pequeñas rotaciones de perspectiva del tablero (rotación ±15°), los filtros convolucionales
    se solapan fuertemente.
```

---

## 4. Impacto de la Inferencia Conjunta (MAP): ¿Cuándo Mejora y Cuándo Empeora?

La Inferencia Conjunta (Variante C) formula la resolución como un problema de optimización biobjetivo:

$$\max \sum_{i \in \text{jaulas}} \sum_{k=1}^K z_{i,k} \cdot \log P(c_{i,k}) \quad \text{sujeto a restricciones de Cuadrado Latino y jaulas activas}$$

donde $z_{i,k} \in \{0, 1\}$ es una variable booleana reificada (`OnlyEnforceIf`) que decide qué candidato visual $k$ se asigna a la jaula $i$.

### 4.1. Estudio Empírico Cuantitativo de Comportamiento

Al auditar individualmente los resultados en los entornos degradados:

| Conjunto de Datos | Muestras | Éxito Top-1 Directo | Éxito MAP Auto | **Rescate Exacto (GT)** | **Divergencia (Pseudo-Solución)** | Insoluble |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`challenge_stress`** | 50 | 4.0% (2) | **90.0% (45)** | **58.0% (29)** | **32.0% (16)** | 10.0% (5) |
| **`synthetic_hard`** | 30 | 30.0% (9) | **53.3% (16)** | **43.3% (13)** | **10.0% (3)** | 46.7% (14) |

---

### 4.2. ¿Cómo Mejora la Situación? (El Rol de Rescate Neuro-Simbólico)
* **Filtrado Lógico de Alucinaciones:** Si la CNN predice en Top-1 una jaula inconsistente (por ejemplo, una jaula de dos celdas horizontales con etiqueta `8+` en un tablero $4\times4$, donde los números máximos son $4$ y la suma no puede superar $4+3=7$ sin repetir), la Variante A falla inmediatamente con `INFEASIBLE`.
* **Deducción de la Verdad:** CP-SAT invalida el candidato Top-1 erróneo y evalúa los candidatos alternativos en Top-$k$. Si el candidato de rango 2 era `3-` o `7+`, el solver verifica que satisface las restricciones del Cuadrado Latino y lo selecciona.
* **Ganancia Neta:** En `challenge_stress`, **27 tableros** que estaban completamente perdidos por el OCR fueron recuperados con la solución matemática original exacta (elevando el éxito de $4\%$ a $58\%$).

---

### 4.3. ¿Cómo Empeora la Situación? (El Fenómeno de la Pseudo-Solución Divergente)
Este es el hallazgo conceptual más relevante de la investigación:

> **Definición de Divergencia:** Ocurre cuando el Ground Truth real de una o más jaulas **NO se encuentra dentro del Top-$k$ de candidatos visuales**, pero el espacio combinatorio de candidatos alternativos contiene combinaciones espurias que admiten un Cuadrado Latino válido.

* **Caso Real Observado en `stress_00000.jpg` ($8\times8$):**
  - La jaula real impresa es `12*` con celdas `[(0,0), (0,1)]`.
  - Debido a una sombra local, la CNN generó los candidatos: `['1-', '1/', '2-', '2/']`. El valor real `12*` quedó fuera del Top-5.
  - El solver CP-SAT, obligado a encontrar una solución factible que maximice la probabilidad acumulada, escogió `1-` (asignando los dígitos `3` y `2`).
  - Para compensar esta asignación en la fila 0, el solver alteró otras 7 jaulas adyacentes seleccionando sus candidatos de rango 3 y 4.
  - **Resultado:** El solver terminó con estado `OPTIMAL`, pero el tablero resuelto resolvió un **puzzle matemático diferente** al que estaba impreso en el papel (32% de los casos en estrés).
* **Riesgo:** Para el usuario final, el sistema reporta que el KenKen fue resuelto con éxito, cuando en realidad se inventó un acertijo alternativo.

---

## 5. Opciones y Propuestas de Solución (Roadmap Técnico)

A partir del diagnóstico, se identifican 5 líneas de mejora concretas para mitigar estos fallos:

### 5.1. Regla Sintáctica Dura por Tamaño de Jaula (Pruning Semántico A Priori)
* **Principio:** Las reglas canónicas de KenKen imponen restricciones topológicas sobre los operadores:
  1. Jaulas de $1$ celda: **Únicamente dígito** (sin operador).
  2. Jaulas de $2$ celdas: admiten `+`, `-`, `*`, `/`.
  3. Jaulas de $\ge 3$ celdas: **NUNCA pueden ser `/` ni `-`**. La resta y división solo están definidas para pares de celdas.
* **Propuesta:** En `decode_readings()`, si `len(cells) >= 3`, filtrar a priori cualquier hipótesis con operador `/` o `-`. Esto eliminará automáticamente las alucinaciones `+ -> /` y `+ -> -` en jaulas grandes.

---

### 5.2. Regularización de Penalización por Desvío ($\ell_0$ Change Penalty)
* **Problema actual:** El solver MAP trata el coste de cambiar del candidato Top-1 al Top-2 únicamente por la diferencia de log-probabilidad $(\log p_1 - \log p_2)$. Si la red tenía baja confianza en ambos, el coste de cambiar es casi cero.
* **Propuesta:** Incorporar un término de penalización fija por cada jaula modificada:

$$\max \sum_{i} \sum_{k} z_{i,k} \log P(c_{i,k}) - \lambda \sum_i (1 - z_{i, 1})$$

donde $\lambda \approx 1.5$. Esto desincentiva cascadas masivas de cambios espurios: el solver solo modificará una jaula si es estrictamente indispensable para evitar la infactibilidad.

---

### 5.3. Detector de Confianza y Límite de Divergencia
* **Propuesta:** Establecer un umbral máximo de jaulas modificadas:

$$M_{\text{max}} = \max\left(2, \lfloor 0.15 \times |\text{jaulas}| \rfloor\right)$$

Si el solver CP-SAT requiere modificar más de $M_{\text{max}}$ jaulas para encontrar solución, en lugar de retornar la pseudo-solución al usuario, debe emitir el estado:
`STATUS = DIVERGENT_UNRELIABLE`
advirtiendo que la calidad de la imagen no garantiza fidelidad con el original.

---

### 5.4. Aprendizaje Profundo de Métrica para Pares Críticos (Supervised Contrastive Loss)
* **Problema:** Cross-Entropy y Focal Loss optimizan la separación hiperplano por clase, pero no maximizan el margen geométrico entre representaciones de clases con morfología casi idéntica como `/` y `+`.
* **Propuesta:** Añadir una pérdida contrastiva (SupCon) sobre el vector de características de 128 dimensiones previo al clasificador denso:
  - Forzar que los embeddings de recortes con `/` se repelan fuertemente de los embeddings de `+` y `-`.

---

### 5.5. Beam Search Adaptativo ($k$ Dinámico)
* **Problema:** Fijar $k=5$ para todas las jaulas es subóptimo: para jaulas claras es redundante, y para jaulas degradadas 5 candidatos pueden excluir la lectura real.
* **Propuesta:** Calcular la entropía de Shannon del vector softmax $H(p) = -\sum p_j \log p_j$:
  - Si $H(p) < 0.2$ (alta certeza): mantener $k=2$.
  - Si $H(p) \ge 0.8$ (alta incertidumbre): expandir dinámicamente $k=8$ o $k=10$.
