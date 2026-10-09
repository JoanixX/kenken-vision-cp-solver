# Política de Fidedignidad, Poda Sintáctica y Abstención Honesta

## 1. El Dilema Ético-Técnico: Éxito Falso vs. Abstención Honesta

En sistemas neuro-simbólicos que integran reconocimiento visual (CNN) y razonamiento formal (CP-SAT), surge un dilema fundamental:

> **¿Qué debe hacer el sistema cuando el reconocimiento visual falla pero el solver es capaz de encontrar *alguna* solución válida inventando un acertijo alternativo?**

Anteriormente, la **Inferencia Conjunta MAP (Variante C)** intentaba encontrar la combinación de lecturas visuales alternativas (Top-$k$) que maximizara la verosimilitud conjunta log-probabilística satisfaciendo el cuadrado latino. No obstante, en condiciones de degradación visual extrema (sombras densas, contraste nulo, artefactos de compresión):
1. La etiqueta real de una o más jaulas a menudo no figuraba en el Top-$k$ de candidatos visuales generados por la CNN.
2. CP-SAT, en su afán de satisfacer las restricciones y encontrar una solución factible, permutaba 3, 4 o hasta 12 jaulas hacia alternativas visualmente inverosímiles (por ejemplo, cambiando un `12*` no detectado por `2/`, `1-` o `3+` en otras posiciones).
3. El sistema proclamaba **`status = OPTIMAL`** y **`solved = True`**, dibujando sobre la foto una solución matemática válida... **pero para un tablero completamente diferente al que el usuario tenía impreso en papel**.

Presentar como "éxito" un acertijo modificado silenciosamente es engañoso, degrada la confianza del usuario y viola los principios de confiabilidad en visión computacional.

---

## 2. Marco Teórico: Clasificación Selectiva y Opción de Rechazo (*Reject Option*)

Siguiendo la teoría de **clasificación selectiva y aprendizaje con opción de rechazo** (*Chow, 1970; Cortes et al., 2016*), un clasificador o pipeline inteligente debe evaluar el trade-off entre:
- **Costo de error no detectado ($c_{\text{error}}$):** Entregar una solución falsa pretendiendo que es la del problema original (costo alto: engaño, desconfianza, frustración).
- **Costo de abstención ($c_{\text{abstain}}$):** Informar honestamente al usuario que la imagen no fue leída con suficiente nitidez y solicitar una nueva captura (costo bajo: transparencia y orientación clara).

Dado que $c_{\text{error}} \gg c_{\text{abstain}}$, la política óptima es **abstenerse activamente** cuando la discrepancia entre la percepción visual directa y el razonamiento simbólico supera un umbral de seguridad.

---

## 3. Arquitectura de Mitigación Implementada

Para resolver este problema sin perder la capacidad de corregir ambigüedades menores legítimas (como confundir `+` con `/`), se implementó una arquitectura en tres capas:

```
[ Imagen de Entrada ]
         │
         ▼
[ 1. Segmentación & OCR ] ──► Poda Sintáctica Canónica (descarte estricto de '-' y '/' en jaulas ≥ 3)
         │
         ▼
[ Candidatos Top-k ]
         │
         ▼
[ 2. Modelo CP-SAT (MAP) ] ──► Regularización L0 (penalización fija λ_change = 1.5 por alterar Top-1)
         │
         ▼
[ Resultado del Solver ]
         │
         ▼
[ 3. Pipeline de Fidedignidad ] ──► Umbral Dinámico: M_cambios ≤ max(2, ⌊0.15 × |Cages|⌋)
         ├── Si M_cambios ≤ M_max ──► ✅ OPTIMAL (Éxito verificado o corrección legítima menor)
         └── Si M_cambios > M_max ──► ⚠️ UNRELIABLE_DETECTION (Abstención Honesta, solved=False)
```

---

## 4. Detalles de Implementación

### 4.1. Poda Sintáctica A Priori en OCR (`kenken/ocr.py`)
En las reglas canónicas de KenKen:
- Las operaciones de **resta (`-`)** y **división (`/`)** son binarias y no asociativas: **están estrictamente restringidas a jaulas de exactamente 2 celdas**.
- Jaulas con 3 o más celdas únicamente pueden tener **suma (`+`)** o **multiplicación (`*`)**.

Se implementó un filtro de poda antes de ingresar las hipótesis al solucionador:
```python
# Poda de operaciones imposibles según el tamaño de la jaula
if cage_size >= 3 and op in "-/":
    continue
```
Esto elimina inmediatamente del espacio de búsqueda cualquier hipótesis absurda generada por ruido en jaulas grandes.

### 4.2. Regularización $\ell_0$ en la Función Objetivo (`kenken/model.py`)
Sin regularización, si dos candidatos tenían probabilidades bajas similares (e.g. $\log p = -1.8$ vs $\log p = -1.9$), la penalización por cambiar la jaula era casi despreciable ($\Delta = 0.1$), lo que facilitaba cascadas de mutaciones descontroladas.

Se incorporó una **penalización $\ell_0$ constante** de $\lambda_{\text{change}} = 1.5$ ($1500$ unidades en escala entera $\times 1000$) por cada jaula que elija una hipótesis $k > 0$ (distinta a la lectura Top-1 visual):

$$\max \sum_{c \in \mathcal{C}} \sum_{k} b_{c,k} \cdot \left( 1000 \cdot \log p(y_{c,k}) - \mathbf{1}_{\{k > 0\}} \cdot 1500 \right)$$

Adicionalmente, se audita y devuelve de forma determinista la métrica:
$$M_{\text{corregidas}} = \sum_{c \in \mathcal{C}} \mathbf{1}_{\{\text{k\_elegido}_c > 0\}}$$

### 4.3. Política de Fidedignidad y Umbral Dinámico (`kenken/pipeline.py`)
El pipeline evalúa el número de mutaciones realizadas por el solver:
$$M_{\text{max}} = \max\left(2, \lfloor 0.15 \times |\mathcal{C}|\rfloor\right)$$

- Si $M_{\text{corregidas}} \le M_{\text{max}}$: Se acepta la solución como una corrección visual legítima de ambigüedad.
- Si $M_{\text{corregidas}} > M_{\text{max}}$:
  - `divergent = True`
  - `solved = False`
  - `status = "UNRELIABLE_DETECTION"`
  - Se emite una advertencia explícita en consola (CLI).

---

## 5. Experiencia de Usuario en el Prototipo Gradio (`prototipo_interactivo/app.py`)

En la interfaz interactiva, cuando se activa `UNRELIABLE_DETECTION`:
1. **No se proclama éxito ni se dibuja la solución ficticia como definitiva.**
2. Se muestra un banner ambar prominente con diagnóstico claro:
   - **Estado:** `UNRELIABLE_DETECTION (Abstención Confiable)`
   - **Inconsistencias detectadas:** Indica cuántas jaulas necesitaron cambiarse y cuál era el límite permitido.
   - **Orientación al usuario:** Se explica con transparencia que para no entregar un acertijo inventado, el sistema prefiere solicitar una fotografía con mejor iluminación, encuadre frontal y sin sombras sobre los caracteres.

---

## 6. Validación Automatizada

Se implementaron pruebas unitarias específicas que verifican:
1. `test_decode_respects_cage_size`: Garantiza que `/` y `-` sean podados en jaulas de $\ge 3$ celdas.
2. `test_solve_joint_tracks_num_changed`: Verifica que $M_{\text{corregidas}}$ cuente exactamente las alteraciones respecto al Top-1.
3. `test_solve_joint_zero_changes_when_clean`: Comprueba que tableros nítidos tengan $M_{\text{corregidas}} = 0$.
4. `test_solve_image_honest_abstention_on_divergence`: Confirma que superar $M_{\text{max}}$ active `divergent = True`, `solved = False` y `status = UNRELIABLE_DETECTION`.
5. `test_resolver_kenken_divergent_warning`: Garantiza que el callback de Gradio reaccione correctamente y no devuelva un falso positivo.

**Resultado de la suite de pruebas:**
```
88 passed in 10.43s (100% de éxito)
```
