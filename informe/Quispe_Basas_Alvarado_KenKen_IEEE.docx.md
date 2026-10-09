# **Resolución automática de KenKen mediante visión computacional y programación por restricciones**

**Autores:**  
Joaquín Basas, Joaquín Alvarado, Johan Quispe  
*Tópicos en Ciencias de la Computación (CC58)*  
*Universidad Peruana de Ciencias Aplicadas (UPC)*  
Lima, Perú  
**Repositorio GitHub:** https://github.com/JoanixX/kenken-vision-cp-solver  
**Demostración en Vivo (Hugging Face Spaces):** https://huggingface.co/spaces/joako2202/kenken-solver  

---

### **Resumen**
Este artículo presenta una arquitectura integral neuro-simbólica para la extracción, interpretación y resolución automatizada de acertijos aritmético-lógicos KenKen a partir de imágenes sin procesar. El sistema articula una fase sensorial de visión computacional (rectificación homográfica $H$, clasificación de bordes mediante Otsu y *Union-Find*, binarización adaptativa ante sombras y reconocimiento óptico de caracteres con una red `GlyphCNN` adaptada mediante *Fine-Tuning* con *Focal Loss* y minería de ejemplos difíciles) con un motor de Programación por Restricciones (CP) sustentado en Google OR-Tools CP-SAT. Formulamos el problema rigurosamente como un CSP $\langle X, D, C \rangle$, contrastando empíricamente: (1) modelado aritmético intensional con descomposición de productos y división reificada, (2) restricciones globales extensionales de tabla con garantía de Consistencia de Arco Generalizada (GAC), y (3) inferencia conjunta neuro-simbólica (MAP) con regularización $\ell_0$ para recuperación robusta ante fallos de lectura visual sin incurrir en divergencia. Evaluaciones sistemáticas sobre tres corpus (Sintético Normal de 300 imágenes, Sintético Difícil de 150 imágenes y un dataset de estrés de 50 tableros patológicos) demuestran una exactitud por glifo del $99.51\%$ y una tasa de resolución récord del $90.33\%$ en imágenes normales. En análisis de complejidad sobre órdenes $n \in [3, 9]$, el solver resuelve tableros de hasta $9 \times 9$ ($1.96 \times 10^{77}$ estados brutos) en menos de $13\text{ ms}$ en el nodo raíz con cero ramas de *backtracking*, logrando una reducción del $39.5\%$ en tiempo de cálculo mediante restricciones de tabla en $n=9$. Finalmente, se implementa una política de clasificación selectiva con opción de rechazo basada en un semáforo de fiabilidad gradual y una interfaz interactiva en Gradio con un catálogo representativo de 21 muestras operacionales.

**Palabras clave:**  
KenKen, visión computacional, GlyphCNN, OCR, programación por restricciones, CP-SAT, inferencia conjunta MAP, restricciones de tabla, GAC, regularización $\ell_0$, semáforo de fiabilidad.

---

## **I. INTRODUCCIÓN**

KenKen es un rompecabezas aritmético-lógico inventado en 2004 por el educador japonés Tetsuya Miyamoto. A diferencia del Sudoku tradicional, donde la cuadrícula está rígidamente subdividida en regiones uniformes de $3 \times 3$, KenKen generaliza el concepto de Cuadrado Latino al imponer restricciones aritméticas sobre subconjuntos conexos e irregulares de celdas denominados *jaulas* (*cages*). Una solución factible debe asignar números enteros del conjunto $\{1, 2, \dots, n\}$ a una cuadrícula de dimensión $n \times n$ garantizando que ningún dígito se repita en ninguna fila o columna, y satisfaciendo exactamente la meta numérica $T_c$ y el operador asignado $\text{op}_c \in \{=, +, -, \times, \div\}$ de cada jaula.

Esta topología induce una explosión combinatoria no restringida que escala como $\mathcal{O}(n^{n^2})$. El reto se incrementa drásticamente cuando el puzzle no proviene de una matriz estructurada, sino de fotografías físicas o capturas digitales tomadas bajo iluminación heterogénea, sombras locales y distorsión geométrica de perspectiva. En un pipeline estrictamente determinista, un único error de lectura en un dígito u operador corrompe las restricciones matemáticas, volviendo el CSP insatisfactible (`INFEASIBLE`) o forzando soluciones falsas.

Para superar este dilema, desarrollamos un sistema híbrido neuro-simbólico que combina:
1. Procesamiento visual en OpenCV con homografía proyectiva $H$ de $3 \times 3$, inferencia del orden $n$, segmentación morfológica y binarización adaptativa ante sombras.
2. Reconocimiento de caracteres mediante una red neuronal convolucional especializada (`GlyphCNN`, 403k parámetros), optimizada con *Fine-Tuning* focalizado, *MultiClass Focal Loss* y *Hard Example Mining*.
3. Modelado matemático formal en Google OR-Tools CP-SAT, comparando formulaciones intensionales, restricciones globales extensionales de tabla con Consistencia de Arco Generalizada (GAC), y un modelo de Inferencia Conjunta (MAP) asistido por regularización $\ell_0$ para rescate de ambigüedades.
4. Una política de fidedignidad basada en la teoría de clasificación selectiva con opción de rechazo (*Reject Option*), desplegada en un semáforo de fiabilidad y una aplicación interactiva en Gradio con 21 casos de prueba.

---

## **II. PIPELINE DE VISIÓN COMPUTACIONAL Y RECONOCIMIENTO DE CARACTERES**

### **A. Arquitectura General y Rectificación Geométrica**
El pipeline sensorial recibe una imagen en formato raster (`.jpg`, `.png`) y la convierte en una estructura simbólica estructurada. La secuencia operativa consta de:
1. **Localización Cuadrangular y Homografía:** Se identifican los contornos convexos dominantes y se calcula la matriz de transformación homográfica proyectiva $H \in \mathbb{R}^{3 \times 3}$, corrigiendo la perspectiva angular hacia un plano frontal canónico:
   $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
2. **Inferencia del Orden $n$:** Se determinan las frecuencias espaciales dominantes mediante perfiles de proyección morfológica de gradientes horizontales y verticales en $n \in [3, 9]$.
3. **Clasificación de Fronteras y Reconstrucción Topológica:** Para distinguir entre líneas internas tenues y bordes gruesos de jaulas, se aplica el método de umbralización de Otsu sobre perfiles de intensidad unidimensionales a lo largo de cada frontera entre celdas. Con la matriz de conectividad resultante, las celdas de cada jaula se agrupan mediante el algoritmo *Union-Find* con compresión de caminos, operando con complejidad casi lineal $\mathcal{O}(n^2 \alpha(n^2))$.

### **B. Mejoras Algorítmicas de Preprocesamiento y Segmentación**
Durante la experimentación sobre escenarios de degradación visual severa, se incorporaron cuatro mecanismos de robustez:
1. **Supresión Periférica de Cuadrícula (`clean_border_lines`):** Evita que remanentes de líneas exteriores distorsionen la altura de referencia tipográfica $ref\_h$. Suprime componentes conectados verticales ($h \ge 0.35 \times \text{cell\_h}$, $w \le 3\text{ px}$) u horizontales adheridos a los márgenes ($x \le 2$ o $x \ge w-4$).
2. **Binarización Resiliente ante Sombras Locales:** Cuando el umbral de Otsu detecta colapso por gradientes de iluminación (área de tinta $> 18\%$), el sistema aplica normalización por desenfoque gaussiano de baja frecuencia:
   $$I_{\text{norm}} = \text{clip}\left(\frac{I}{\mathcal{G}_{\sigma=31}(I)} \times 255, \; 0, \; 255\right)$$
   rescatando el contraste de los dígitos en sombras intensas.
3. **Hipótesis Combinadas de Segundo Orden:** En números compuestos de dos dígitos (e.g. `14+`, `48*`) con *kerning* estrecho, se evalúan cortes y uniones verticales concurrentes ponderadas log-probabilísticamente.
4. **Poda Sintáctica Canónica a Priori:** En las reglas canónicas de KenKen, las operaciones de resta ($-$) y división ($\div$) son binarias y no asociativas, restringidas exclusivamente a jaulas de exactamente 2 celdas. Toda hipótesis que contenga $-$ o $\div$ en jaulas de 3 o más celdas es podada inmediatamente del espacio de búsqueda, previniendo alucinaciones por ruido.

### **C. Clasificador GlyphCNN y Metodología de Fine-Tuning**
Para el reconocimiento óptico de caracteres se implementó una red convolucional especializada denominada `GlyphCNN`:
- **Arquitectura:** Entrada de $32 \times 32$ píxeles en escala de grises, 3 bloques convolucionales con *BatchNorm2D*, activación *ReLU* y *MaxPool2D* (filtros 32, 64, 64), seguidos de *Dropout* ($p=0.25$) y 2 capas lineales densas ($512 \to 14$), con aproximadamente 403,000 parámetros entrenables y 14 clases de salida (dígitos `0`-`9` y operadores `+`, `-`, `*`, `/`).
- **Preservación del Modelo Base:** El modelo original (`models/ocr_cnn.pt`, hash SHA-256 `CE25BB9D...`) fue preservado intacto.
- **Estrategia de Fine-Tuning:** Se entrenó una versión especializada (`models/ocr_cnn_finetuned.pt`, hash SHA-256 `D4A64C20...`, ahora predeterminado en el pipeline) con:
  1. *Corpus Focalizado de 33,122 glifos:* 31,093 glifos sintéticos generados con muestreo sesgado al 60% en operadores conflictivos (`/`, `+`, `-`, `*`) y dígitos ambiguos sobre diversas tipografías reales (`DejaVuSans`, `FreeMono`, etc.) con perturbaciones de trazo $x^\gamma$ ($\gamma \in [0.75, 1.35]$), rotaciones $\pm 10^\circ$, ruido impulsivo y desenfoque; más 2,029 glifos recortados de tableros reales degradados.
  2. *MultiClass Focal Loss:* Formulación de pérdida focal con modulación de muestras ambiguas y *Label Smoothing* ($\epsilon = 0.02$):
     $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
     con $\gamma = 1.5$ y ponderaciones de penalización severa para operadores: $\alpha_{/} = 1.40$, $\alpha_{+} = 1.35$, $\alpha_{-} = 1.25$, $\alpha_{*} = 1.25$, $\alpha_{\text{dígitos}} = 1.00$.
  3. *Congelamiento Selectivo:* Épocas 1-2 con Bloque 1 congelado ($lr = 2 \times 10^{-4}$); épocas 3-8 con descongelamiento global y planificador *Cosine Annealing* descendiendo hasta $10^{-5}$. El mejor checkpoint se alcanzó en la época 5 con `val_acc = 99.31%` y `loss = 0.0304`.

### **D. Métricas Experimentales de Percepción y Reconocimiento**
El pipeline se evaluó cuantitativamente sobre tres corpus con niveles crecientes de adversidad:
1. **Synthetic Normal (300 imágenes, 12,266 glifos):** Condiciones nítidas y controladas.
2. **Synthetic Hard (150 imágenes, 4,859 glifos):** Ruido físico severo, viñeteado, inclinación y degradación de trazo.
3. **Challenge Stress (50 imágenes, 1,778 glifos):** Tableros donde la lectura directa Top-1 presenta al menos una inconsistencia o fallo sensorial.

#### **1) Comparativa Global por Nivel de Entidad:**
La exactitud de reconocimiento se define como:
$$\text{Accuracy} = \frac{\text{Aciertos}}{\text{Total Evaluado}} \times 100\%$$

| Métrica Crítica | Dataset Normal (300 imgs) | Dataset Hard (150 imgs) | Dataset Stress (50 imgs) |
|---|:---:|:---:|:---:|
| **Glifos Analizados** | 12,266 | 4,859 | 1,778 |
| **Exactitud por Glifo Individual** | **99.51%** (12,206/12,266) | **90.99%** (4,421/4,859) | **89.43%** (1,590/1,778) |
| **Segmentación de Jaula Correcta** | 96.50% | 75.80% | 80.50% |
| **Jaula Top-1 Exacta** | 97.50% | 74.30% | 83.80% |
| **Jaula Top-$k$ (con alternativas)** | **99.20%** | **84.40%** | **93.90%** |
| **Tablero Top-1 Directo (sin CP)** | 82.00% (246/300) | 32.00% (48/150) | 4.00% (2/50) |
| **Tablero Resuelto (Top-$k$ + CP-SAT)** | **90.33% (271/300)** | **44.67% (67/150)** | **52.00% (26/50)** |
| **Tiempo Medio por Imagen** | **0.261 s** (-51.9% vs base) | **0.365 s** | **0.387 s** |

*Tabla I. Métricas comparativas globales de percepción a través de los tres conjuntos de evaluación.*

#### **2) Desglose Exhaustivo por Dimensión ($3\times3$ vs $4\times4$ vs $5\times5$ vs $6\times6$ vs $7\times7$ vs $8\times8$ vs $9\times9$):**
El incremento en el tamaño de la cuadrícula impone un desafío doble: la celda dispone de menor resolución en píxeles y el número de jaulas aumenta exponencialmente.

| Dimensión | Cant. Normal | Jaula Top-1 (Norm) | Tab. Top-1 (Norm) | Tab. + CP (Norm) | Cant. Hard | Jaula Top-1 (Hard) | Tab. Top-1 (Hard) | Tab. + CP (Hard) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$3 \times 3$** | 39 | 99.4% | 97.4% | **97.4%** | 23 | 95.0% | 82.6% | **91.3%** |
| **$4 \times 4$** | 41 | 99.7% | 97.6% | **100.0%** | 28 | 77.7% | 39.3% | **50.0%** |
| **$5 \times 5$** | 45 | 99.4% | 95.6% | **95.6%** | 18 | 75.2% | 27.8% | **27.8%** |
| **$6 \times 6$** | 40 | 97.7% | 82.5% | **92.5%** | 15 | 60.2% | 20.0% | **33.3%** |
| **$7 \times 7$** | 44 | 98.7% | 81.8% | **90.9%** | 16 | 59.4% | 6.2% | **12.5%** |
| **$8 \times 8$** | 47 | 96.7% | 61.7% | **80.9%** | 16 | 77.3% | 18.8% | **43.8%** |
| **$9 \times 9$** | 44 | 96.3% | 61.4% | **77.3%** | 34 | 77.8% | 17.6% | **38.2%** |

*Tabla II. Desglose detallado por orden de tablero ($n=3$ a $9$) comparando Synthetic Normal y Synthetic Hard.*

En tableros de menor dimensión ($3\times3$ y $4\times4$), la asistencia del solucionador CP permite elevar el éxito de resolución al $100.0\%$ y $91.3\%$. En tableros grandes ($8\times8$ y $9\times9$), aunque el reconocimiento directo decae al $\approx 61\%$ por acumulación de probabilidades de error, la inferencia conjunta rescata un volumen sustancial de tableros, alcanzando un $77.3\%$ de resolución exitosa.

#### **3) Desempeño por Carácter Individual (F1-Score) y Pares de Confusión Críticos:**
La evaluación de las matrices de confusión confirma que el operador de división ($\div$) es el elemento más vulnerable del sistema sensorial:

| Glifo / Carácter | Rol Funcional | F1 Normal (300 imgs) | F1 Hard (150 imgs) | F1 Stress (50 imgs) | Nivel de Vulnerabilidad |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **`/`** | Operador | **95.21%** | **76.13%** | **70.87%** |  **Crítica (Peor clase)** |
| **`-`** | Operador | 98.93% | 85.60% | 85.29% |  Alta |
| **`+`** | Operador | 98.90% | 87.95% | 91.24% |  Moderada |
| **`*`** | Operador | 99.91% | 92.34% | 98.55% |  Baja |
| **`0`** | Dígito | 99.66% | 89.11% | 94.17% |  Moderada |
| **`1`** | Dígito | 99.85% | 91.96% | 95.58% |  Baja (absorbe falsos positivos) |
| **`2`** | Dígito | 99.71% | 94.21% | 97.14% |  Baja |
| **`3`** | Dígito | 100.00% | 93.56% | 93.72% |  Baja |
| **`4`** | Dígito | 99.76% | 94.39% | 98.35% |  Baja |
| **`5`** | Dígito | 100.00% | 90.09% | 97.65% |  Moderada |
| **`6`** | Dígito | 99.92% | 92.84% | 96.12% |  Baja |
| **`7`** | Dígito | 99.74% | 96.18% | 96.70% |  Baja |
| **`8`** | Dígito | 100.00% | 92.34% | 98.01% |  Baja |
| **`9`** | Dígito | 100.00% | 87.00% | 97.50% |  Moderada |

*Tabla III. Desempeño por clase: F1-Score (\%) en los tres benchmarks evaluados.*

Las cinco confusiones físicas más frecuentes observadas y sus causas son:
1. **`/` $\to$ `+` (52 casos):** La barra oblicua al intersectar residuos de la línea divisoria de la cuadrícula genera un falso brazo horizontal, activando los filtros ortogonales de `+`.
2. **`+` $\to$ `/` (40 casos):** La erosión morfológica o atenuación por sombras elimina uno de los brazos del `+`, haciendo que la inclinación proyectiva lo asemeje a un trazo oblicuo.
3. **`-` $\to$ `+` (26 casos):** Una línea vertical espuria que cruza el guion central transforma el signo en una cruz.
4. **`2` $\to$ `1` (24 casos):** En tableros densos ($8\times8$, $9\times9$), la baja resolución de la celda desvanece las curvas superior e inferior del `2`, reteniendo únicamente la espina vertical.
5. **`*` $\to$ `+` (18 casos):** El signo de multiplicación $\times$ difiere de $+$ únicamente por una rotación de $45^\circ$, solapando filtros ante ángulos residuales de perspectiva.

---

## **III. MODELO MATEMÁTICO FORMAL DE CONSTRAINT PROGRAMMING**

### **A. Definición del CSP y Variables de Decisión**
El acertijo KenKen de orden $n$ se formula como un Problema de Satisfacción de Restricciones formal $\mathcal{P} = \langle X, D, C \rangle$:
- **Variables de Decisión:**
  $$X = \{x_{i,j} \mid 1 \le i \le n, \; 1 \le j \le n\}$$
  donde $x_{i,j}$ representa el valor entero asignado a la celda en la fila $i$ y columna $j$, con un total de $n^2$ variables.
- **Dominios Finitos:**
  $$D(x_{i,j}) = \{1, 2, \dots, n\}, \quad \forall x_{i,j} \in X$$
- **Partición de Jaulas:** La colección de jaulas $\mathcal{C} = \{c_1, \dots, c_m\}$ conforma una partición disjunta del tablero ($\bigcup V_c = X$ y $V_a \cap V_b = \emptyset$). Cada jaula $c = \langle V_c, T_c, \text{op}_c \rangle$ está definida por sus celdas $V_c \subseteq X$, su valor objetivo $T_c \in \mathbb{N}$ y su operador $\text{op}_c$.

### **B. Restricciones Globales de Cuadrado Latino**
Para asegurar la propiedad fundamental de no repetición de valores en filas y columnas, se aplican restricciones globales de desigualdad mediante el algoritmo de filtrado de hiperarco de Régin:
$$\text{AllDifferent}(\{x_{i,1}, x_{i,2}, \dots, x_{i,n}\}), \quad \forall i \in \{1, \dots, n\}$$
$$\text{AllDifferent}(\{x_{1,j}, x_{2,j}, \dots, x_{n,j}\}), \quad \forall j \in \{1, \dots, n\}$$

Asimismo, se formula la restricción redundante (implicada) de la suma triangular de Gauss:
$$\sum_{j=1}^n x_{i,j} = \frac{n(n+1)}{2}, \quad \sum_{i=1}^n x_{i,j} = \frac{n(n+1)}{2}$$
la cual acelera la propagación de cotas lineales en órdenes superiores ($n=9$).

### **C. Variante A: Modelado Aritmético Intensional**
Descompone las restricciones locales según la naturaleza matemática de cada jaula $c$:
1. **Identidad ($=$):** Para $|V_c| = 1$:
   $$v_1 = T_c, \quad v_1 \in V_c$$
2. **Suma ($+$):** Restricción afín lineal global:
   $$\sum_{v \in V_c} v = T_c$$
3. **Multiplicación ($\times$):** Descomposición encadenada mediante variables auxiliares enteras $p_k \in [1, n^{k+1}]$ vía `AddMultiplicationEquality`:
   $$p_1 = v_1 \cdot v_2; \quad p_k = p_{k-1} \cdot v_{k+1} \; (k \ge 2); \quad T_c = p_{|V_c|-2} \cdot v_{|V_c|}$$
4. **Diferencia Absoluta ($-$):** Para $|V_c| = 2$:
   $$|v_1 - v_2| = T_c \iff \text{AddAbsEquality}(T_c, v_1 - v_2)$$
5. **División Reificada ($\div$):** Como el orden de operandos no está predefinido en la imagen, se formula mediante reificación lógica con una variable booleana de dirección $b_{\text{dir}} \in \{0, 1\}$ y `OnlyEnforceIf`:
   $$b_{\text{dir}} \implies v_1 = T_c \cdot v_2$$
   $$\neg b_{\text{dir}} \implies v_2 = T_c \cdot v_1$$

### **D. Variante B: Restricciones Globales Extensionales de Tabla (GAC)**
En la Variante B, cada jaula se modela extensionalmente a través de la enumeración previa de las asignaciones factibles $\mathcal{T}_c \subset \{1, \dots, n\}^{|V_c|}$, aplicando `AddAllowedAssignments`:
$$\text{AllowedAssignments}(V_c, \mathcal{T}_c)$$
El compilador de la tabla aplica filtrado topológico a priori: si dos celdas de la jaula pertenecen a la misma fila o columna de la cuadrícula, cualquier tupla que asigne valores idénticos entre ellas es descartada antes de entrar al solver. Esto garantiza **Consistencia de Arco Generalizada (GAC)** en el nodo raíz sin requerir variables auxiliares.

### **E. Variante C: Inferencia Conjunta Neuro-Simbólica (MAP) con Regularización $\ell_0$**
Para tolerar fallos sensoriales del OCR, se formula un problema de Máxima Verosimilitud Consistente (MAP). Para cada jaula $c$ con $K_c$ candidatos de lectura y probabilidades normalizadas $P_{c,k}$, se introducen variables booleanas indicadoras:
$$r_{c,k} \in \{0, 1\}, \quad \forall c \in \mathcal{C}, \; \forall k \in \{0, \dots, K_c - 1\}$$
sujetas a selección única:
$$\text{ExactlyOne}(\{r_{c,0}, r_{c,1}, \dots, r_{c,K_c - 1}\}), \quad \forall c \in \mathcal{C}$$
Las condiciones de cada hipótesis se activan mediante reificación condicional con `OnlyEnforceIf`:
$$r_{c,k} \implies \mathcal{A}(V_c, T_{c,k}, \text{op}_{c,k})$$

Para evitar que el solver altere jaulas arbitrariamente inventando acertijos espurios, se incorpora una **penalización $\ell_0$ constante** ($\lambda_{\text{change}} = 1.5$, escalada a 1500 unidades enteras):
$$\max \sum_{c \in \mathcal{C}} \sum_{k=0}^{K_c - 1} r_{c,k} \cdot \left( \lfloor 1000 \cdot \log P_{c,k} \rfloor - \mathbf{1}_{\{k > 0\}} \cdot 1500 \right)$$
El solver cuantifica el total de jaulas ajustadas:
$$M_{\text{corregidas}} = \sum_{c \in \mathcal{C}} \sum_{k=1}^{K_c - 1} r_{c,k}$$

---

## **IV. ANÁLISIS DE COMPLEJIDAD Y EVALUACIÓN EXPERIMENTAL DEL SOLVER**

### **A. Complejidad Combinatoria Teórica vs. Espacio Podado**
El espacio de búsqueda no restringido de una cuadrícula KenKen de orden $n$ es:
$$|\mathcal{S}_{\text{bruto}}| = n^{n^2} = \mathcal{O}(n^{n^2})$$
Para $n=3$, $|\mathcal{S}| = 3^9 = 19,683$. Para $n=9$, el espacio combinatorial alcanza $9^{81} \approx 1.96 \times 10^{77}$ estados brutos (magnitud comparable al número de átomos en el universo observable).

Al incorporar las restricciones de permutación por filas, el espacio se reduce a $(n!)^n$ ($1.09 \times 10^{50}$ en $n=9$). Las restricciones de columnas acotan el problema al subconjunto de Cuadrados Latinos $L(n)$, y las restricciones aritméticas de las jaulas reducen el espacio a una solución única.

El solver CP-SAT aprovecha la tecnología de *Lazy Clause Generation* (LCG) y propagación de cotas para podar dominios de forma tan estricta que la solución se deduce en el **nodo raíz**, requiriendo **0 ramas de backtracking**.

### **B. Resultados Experimentales del Benchmark CP-SAT ($n=3$ a $n=9$)**
Se ejecutaron 12 repeticiones independientes por orden y formulación sobre tableros aleatorios generados desde $n=3$ hasta $n=9$. Los resultados cuantitativos se consolidan en la Tabla IV:

| Orden $n$ | Espacio Bruto $n^{n^2}$ | Var. A (Intensional) | Var. A + Redundante | Var. B (Tabla GAC) | Var. B + Redundante | Ramas Exploradas |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **3** | $1.97 \times 10^4$ | **9.07 ms** | 15.26 ms | 15.00 ms | 14.47 ms | **0** |
| **4** | $4.29 \times 10^9$ | **10.38 ms** | 14.28 ms | 15.46 ms | 14.70 ms | **0** |
| **5** | $2.98 \times 10^{17}$ | **13.53 ms** | 14.56 ms | 14.51 ms | 14.46 ms | **0** |
| **6** | $1.03 \times 10^{28}$ | **9.54 ms** | 14.39 ms | 15.42 ms | 14.22 ms | **0** |
| **7** | $2.56 \times 10^{41}$ | **9.23 ms** | 14.56 ms | 13.68 ms | 14.26 ms | **0** |
| **8** | $6.28 \times 10^{57}$ | 12.68 ms | 14.31 ms | 13.25 ms | **12.18 ms** | **0** |
| **9** | $1.96 \times 10^{77}$ | 20.03 ms | 13.03 ms | **12.11 ms** | 17.22 ms | **0** |

*Tabla IV. Comparativa experimental de rendimiento del Solver CP-SAT ($n=3$ a $n=9$): Wall Time promedio (ms) y ramas de búsqueda.*

### **C. Discusión de Hallazgos Empíricos del Solver**
1. **Aceleración GAC en Gran Escala (Variante B):** Para $n=9$, la formulación extensional de tabla (Variante B) resolvió en **12.11 ms** frente a **20.03 ms** de la formulación aritmética directa (Variante A), logrando una **reducción del 39.5% en tiempo de ejecución**. El filtrado GAC elimina inconsistencias antes de la propagación booleana.
2. **Efecto de Restricciones Redundantes:** En tableros pequeños ($n \le 6$), la suma Gaussiana introduce una sobrecarga marginal ($\approx 4\text{ ms}$). Sin embargo, para $n=9$ acelera la formulación intensional en un $35\%$ ($20.03\text{ ms} \to 13.03\text{ ms}$), demostrando su valor en problemas de gran escala.
3. **Cero Backtracking:** En todas las dimensiones evaluadas se observaron **0 ramas y 0 conflictos**, confirmando la efectividad del filtrado de arco en CP-SAT.

---

## **V. POLÍTICA DE FIDEDIGNIDAD, MITIGACIÓN DE DIVERGENCIA Y SEMÁFORO DE FIABILIDAD**

### **A. El Dilema Ético-Técnico y el Fenómeno de Divergencia**
En sistemas neuro-simbólicos sin regularización ocurre un fenómeno crítico: cuando una imagen está fuertemente degradada y la etiqueta real no figura entre las hipótesis Top-$k$, CP-SAT altera múltiples jaulas hacia opciones inverosímiles para satisfacer la grilla. El sistema proclama `OPTIMAL` y `solved = True`, pero entregando la solución de un **acertijo inventado**. Este fenómeno de **Divergencia (Pseudo-solución)** afectó al $32.0\%$ del dataset de estrés.

### **B. Marco de Clasificación Selectiva y Opción de Rechazo**
Fundamentado en la teoría de *Reject Option* (Chow, 1970; Cortes et al., 2016), el coste de entregar un error engañoso ($c_{\text{error}}$) es infinitamente superior al de una abstención transparente ($c_{\text{abstain}}$). El sistema debe advertir con honestidad la degradación de la imagen.

### **C. Semáforo de Fiabilidad Informativo y Métrica de Fidelidad Visual**
Para equilibrar el rescate legítimo de ambigüedades con la transparencia al usuario, se formuló la **Métrica de Fidelidad Visual**:
$$\text{Fidelidad} = \frac{|\mathcal{C}| - M_{\text{corregidas}}}{|\mathcal{C}|} \times 100\%$$

El sistema opera bajo los siguientes principios:
1. **No Censura de Soluciones Factibles:** Si CP-SAT demuestra factibilidad (`OPTIMAL`), la solución calculada siempre se proyecta (`solved = True`).
2. **Nivel Cualitativo de Confianza:**
   -  **Alta (`HIGH`):** 0 cambios sobre Top-1 ($100\%$ fidelidad) o 1 a 3 correcciones de alta verosimilitud ($\ge 70\%$ fidelidad).
   -  **Moderada (`MODERATE`):** Fidelidad entre $50\%$ y $70\%$, o correcciones múltiples. El sistema emite un aviso preventivo para que el usuario verifique las etiquetas ajustadas.
   -  **Baja / Divergencia Sospechada (`LOW`):** Fidelidad $< 50\%$, se activa la bandera consultiva `divergent = True`.

---

## **VI. INTEGRACIÓN DE SOFTWARE, VISUALIZACIÓN Y PROTOTIPO INTERACTIVO**

### **A. Puente End-to-End Autónomo**
El sistema integra la visión y el razonamiento de forma totalmente automática mediante la función `solve_image(path, method="auto")`:
1. El pipeline de visión extrae la geometría y la lista de candidatos OCR.
2. Se ejecuta un intento rápido con la Variante A sobre las predicciones Top-1.
3. Si el modelo es `INFEASIBLE`, conmuta automáticamente a la Variante C (MAP regularizada con $\ell_0$), resolviendo el acertijo sin intervención humana.

### **B. Proyección Gráfica de Soluciones**
Se proporcionan cuatro modalidades visuales:
- **Modo `original`:** Proyecta los dígitos resueltos directamente sobre la foto física original deformando coordenadas tipográficas mediante la homografía proyectiva inversa $H^{-1}$.
- **Modo `composite`:** Genera una infografía comparativa lado a lado (imagen de entrada con contorno y grilla vectorial resuelta).
- **Modo `clean`:** Cuadrícula vectorial nítida en alta resolución lista para publicación.
- **Modo `rectified`:** Dígitos sobre la cuadrícula frontal rectificada.

### **C. Prototipo Web Interactivo Desplegado en Hugging Face Spaces**
El prototipo interactivo fue empaquetado y desplegado públicamente en la nube en **Hugging Face Spaces** utilizando Gradio:  
 **https://huggingface.co/spaces/joako2202/kenken-solver**  
Repositorio GitHub: **https://github.com/JoanixX/kenken-vision-cp-solver**

La interfaz web integra:
- Arrastrar y soltar de imágenes en tiempo real con diagnóstico de ejecución y latencia.
- Panel de transparencia neuro-simbólica con badges del semáforo de fiabilidad, porcentaje de fidelidad visual y lista de jaulas corregidas.
- Catálogo interactivo de 21 muestras operacionales (3 por cada uno de los 7 perfiles: directo, rescate MAP, perspectiva, gran escala $9\times9$, rescate de estrés, aviso de verificación e infactibilidad honesta).

---

## **VII. DISCUSIÓN Y LIMITACIONES**

El contraste entre la visión y el razonamiento simbólico evidencia que el cuello de botella del sistema reside en la percepción visual de operadores densos (particularmente $\div$, con un F1 de $70.87\%$ en estrés). La inferencia conjunta MAP demostró ser crucial al elevar la recuperación de tableros del $4.0\%$ al $58.0\%$ en estrés extremo; no obstante, sin regularización $\ell_0$ puede derivar en soluciones divergentes. La introducción de la penalización $\lambda_{\text{change}} = 1.5$ y la poda canónica a priori resuelven este riesgo de manera determinista.

Como limitación actual, el algoritmo de rectificación presupone contornos rectilíneos convexos sobre superficies planas; imágenes sobre papel arrugado requerirían técnicas de desdoblamiento de superficie (*page dewarping*).

---

## **VIII. CONCLUSIONES**

Se presentó, implementó y validó una arquitectura híbrida neuro-simbólica completa para la resolución autónoma de acertijos KenKen:
1. El modelo `GlyphCNN` con *Fine-Tuning* focalizado y *Focal Loss* alcanzó un $99.51\%$ de exactitud por glifo y un $90.33\%$ de tableros resueltos en condiciones estándar, manteniendo una latencia promedio de $0.26\text{ s}$ por imagen.
2. El modelo de CP-SAT con restricciones de tabla extensionales (Variante B) demostró consistencia de arco generalizada (GAC), logrando una aceleración del $39.5\%$ en $n=9$ y resolviendo todas las instancias en el nodo raíz con 0 ramas de *backtracking*.
3. La regularización $\ell_0$ y el semáforo de fiabilidad gradual garantizan una recuperación transparente ante fallos de visión, previniendo la entrega de soluciones espurias.
4. El pipeline opera de extremo a extremo sin intervención manual y cuenta con una interfaz web en Gradio validada mediante 21 casos de demostración y una suite de pruebas automatizadas con 100% de éxito.

---

## **REFERENCIAS**

1. Bessière, C., Hebrard, E., Hnich, B., Kiziltan, Z., & Walsh, T. (2006). Filtering algorithms for table constraints: An overview. *Constraints*, 11(4), 271–304.
2. Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal: Software Tools for the Professional Programmer*, 25(11), 120–123.
3. Chow, C. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41–46.
4. Cortes, C., DeSalvo, G., & Mohri, M. (2016). Learning with rejection. En *Algorithmic Learning Theory (ALT)* (pp. 67–82). Springer.
5. De Raedt, L., Guns, T., & Nijssen, S. (2011). Constraint programming for data mining and machine learning. En *Proceedings of the AAAI Conference on Human Computation and Crowdsourcing*.
6. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. En *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 2980–2988).
7. Ohrimenko, O., Stuckey, P. J., & Codish, M. (2009). Propagation = lazy clause generation. En *Principles and Practice of Constraint Programming (CP)* (pp. 444–458). Springer.
8. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
9. Perron, L., Didier, F., & Gay, S. (2023). The CP-SAT-LP solver (invited talk). En *29th International Conference on Principles and Practice of Constraint Programming (CP 2023)* (Vol. 280, pp. 3:1–3:2). Schloss Dagstuhl.
10. Perron, L., & Furnon, V. (2024). Google OR-Tools: Software suite for combinatorial optimization. https://developers.google.com/optimization
11. Régin, J.-C. (1994). A filtering algorithm for constraints of difference in CSPs. En *AAAI*, 94, 362–367.
12. Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *Journal of the ACM*, 22(2), 215–225.