# Documentación Técnica del Proyecto (`docs/`)

Índice central y guía de lectura para la documentación técnica, arquitectónica y experimental del proyecto **KenKen Vision-CP Solver**.

---

## 1. Fundamentos Teóricos y Modelado

- [**Documentación del Modelo Constraint Programming (`cp_documentation.md`)**](cp_documentation.md):  
  Formalización matemática completa del CSP $\langle X, D, C \rangle$, formulación canónica de operaciones aritméticas intensionales, consistencia de arco generalizada (GAC) mediante restricciones globales de tabla (`AddAllowedAssignments`), formulación MAP conjunta neuro-simbólica y restricciones implicadas de suma en Cuadrado Latino.

- [**Arquitectura GlyphCNN y Clasificación (`cnn_explicada.md`)**](cnn_explicada.md):  
  Detalles de diseño de la red neuronal convolucional ligera, preprocesamiento de glifos, aumento de datos sintético y optimización con Cross-Entropy y Focal Loss.

---

## 2. Robustez, Inferencia Conjunta y Confiabilidad

- [**Política de Fidedignidad y Criterios de Abstención (`politica_fidedignidad_y_abstencion.md`)**](politica_fidedignidad_y_abstencion.md):  
  Definición de umbrales de divergencia perceptual, política de advertencias cuando el solver ajusta múltiples lecturas de jaula y mecanismos de abstención ante tableros contradictorios o infactibles.

- [**Análisis Profundo de Fallos en MAP y OCR (`analisis_profundo_fallos_map_y_ocr.md`)**](analisis_profundo_fallos_map_y_ocr.md):  
  Estudio empírico de casos de borde: colisiones fonético-visuales (+ vs *), oclusiones y rescate lógico exitoso.

- [**Guía de Fine-Tuning y Resultados Experimentales (`fine_tuning_guia_y_resultados.md`)**](fine_tuning_guia_y_resultados.md):  
  Protocolo de entrenamiento focalizado con minería de pares conflictivos y validación cruzada.

---

## 3. Métricas Cuantitativas y Datasets

- [**Análisis de Métricas de OCR y Pipeline (`analisis_metricas_ocr_y_pipeline.md`)**](analisis_metricas_ocr_y_pipeline.md):  
  Tablas consolidadas de exactitud por componente (segmentación de tablero, detección de orden $n$, $F_1$ de jaulas, precisión Top-1 y Top-$k$ de OCR, y éxito global end-to-end).

- [**Dataset de Estrés y Desafío (`dataset_challenge_stress.md`)**](dataset_challenge_stress.md):  
  Metodología de diseño y caracterización del benchmark de estrés para pruebas de fallos inducidos.

---

## 4. Bitácora de Implementación y Archivo Histórico

- [**Plan de Implementación del Solver CP (`plan_implementacion_cp.md`)**](plan_implementacion_cp.md):  
  Bitácora de diseño técnico, hitos y criterios de aceptación originales de la etapa de programación por restricciones.

- [**Plan Histórico Fase 1 (`archive/PLAN_FASE1.md`)**](archive/PLAN_FASE1.md):  
  Documento inicial de especificación y arquitectura del proyecto.
