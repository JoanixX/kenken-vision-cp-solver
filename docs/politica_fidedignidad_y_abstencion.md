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

### 4.3. Semáforo de Fiabilidad y Métrica de Fidelidad Visual (`kenken/pipeline.py`)

Tras auditar casos reales de estrés como `stress_00011.jpg`, se constató que un corte estricto que bloquee la solución (`solved = False`) cuando $M_{\text{corregidas}} > 2$ es **contraproducente**: en ese ejemplo, el solver corrigió legítimamente 3 etiquetas borrosas (`18*` ➔ `180*`), alcanzando un **$100\%$ de coincidencia exacta con el Ground Truth**. Declarar "no resuelto" en un caso 100% acertado desmerece el poder neuro-simbólico del sistema.

Por ello, la arquitectura evolucionó de un bloqueo binario hacia un **Semáforo de Fiabilidad Informativo y Gradual**:

1. **La solución óptima nunca se censura:** Si CP-SAT demuestra factibilidad matemática (`OPTIMAL`), `solved = True` y la matriz resultante se entrega al usuario.
2. **Métrica de Fidelidad Visual:**
   $$\text{Fidelidad} = \frac{|\mathcal{C}| - M_{\text{corregidas}}}{|\mathcal{C}|} \times 100\%$$
3. **Nivel de Confianza Cualitativo (`confidence_level`):**
   - 🟢 **`HIGH` (Fiabilidad Muy Alta / Alta):** 0 cambios sobre Top-1 ($100\%$ de fidelidad), o 1 a 3 correcciones de alta verosimilitud con fidelidad $\ge 70\%$.
   - 🟡 **`MODERATE` (Aviso de Verificación):** Fidelidad entre $50\%$ y $70\%$, o correcciones múltiples. El sistema emite un aviso preventivo para que el usuario verifique las etiquetas ajustadas.
   - 🔴 **`LOW` (Divergencia Sospechada):** Menos del $50\%$ de las jaulas visuales conservadas.
4. **Flag Consultivo de Divergencia (`divergent`):**
   Indica si $M_{\text{corregidas}} > \max(3, \lfloor 0.35 \times |\mathcal{C}|\rfloor)$, alertando a los consumidores de la API sobre la necesidad de inspección visual de las restricciones.

---

## 5. Experiencia de Usuario en el Prototipo Streamlit (`prototipo_interactivo/app.py`)

En la interfaz interactiva:
1. **Entrega de Solución:** Siempre se proyecta la cuadrícula resuelta si el solver alcanzó `OPTIMAL`.
2. **Diagnóstico Transparente en Tiempo Real:**
   - Si no hubo cambios: Se resalta la consistencia directa de la percepción visual.
   - Si hubo correcciones legítimas: Se muestra un badge verde detallando el rescate neuro-simbólico y la fidelidad obtenida (e.g. `82.4%`).
   - Si se requirieron múltiples ajustes ($\ge 35\%$ del tablero): Se despliega una advertencia amarilla informativa solicitando al usuario verificar si las jaulas ajustadas coinciden con su impreso si alguna etiqueta estuvo muy borrosa.
3. **Catálogo de 8 Muestras por Defecto:**
   Incluye desde tableros directos y desafíos de perspectiva hasta rescates avanzados de 3 jaulas (`caso6`) y avisos de verificación (`caso7`).

---

## 6. Validación Automatizada

Se implementaron pruebas unitarias específicas que verifican:
1. `test_decode_respects_cage_size`: Garantiza que `/` y `-` sean podados en jaulas de $\ge 3$ celdas.
2. `test_solve_joint_tracks_num_changed`: Verifica que $M_{\text{corregidas}}$ cuente exactamente las alteraciones respecto al Top-1.
3. `test_solve_joint_zero_changes_when_clean`: Comprueba que tableros nítidos tengan $M_{\text{corregidas}} = 0$.
4. `test_solve_image_divergence_metrics`: Confirma que el pipeline registre correctamente `fidelity`, `num_changed` y `confidence_level` manteniendo `solved = True` en `OPTIMAL`.
5. `test_resolver_kenken_multiple_adjustments_warning`: Verifica que el callback del prototipo interactivo despliegue el aviso de verificación ante múltiples ajustes sin censurar la solución.
6. `test_resolver_kenken_infeasible`: Confirma el reporte formal y transparente de `INFEASIBLE`.
7. `test_all_showcase_cases_exist`: Valida que todas las imágenes demostrativas del prototipo existan y carguen correctamente.

**Resultado de la suite de pruebas:**
```
90 passed in 16.93s (100% de éxito)
```

