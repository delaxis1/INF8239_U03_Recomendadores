# System Card — recomendación Top-10

Uso: demostración educativa local; no asesoramiento ni prueba de impacto real. Configuración elegida: **Conteo**, exclusivamente con validación. Regla: NDCG a 0.005 del máximo; mayor cobertura y luego menor ajuste. Registro previo a prueba: `reports/decision_locked.json` (2026-10-19T17:33:29.342189+00:00).

Siete configuraciones: conteo, media bayesiana suavizada, perfil de géneros TF-IDF/coseno, MF10, MF20, híbridos α0.25/0.75. α pesa la señal colaborativa, con scores normalizados min-max sobre el mismo catálogo; (1−α) pesa contenido. MF sin sesgos de usuario/ítem del código base, 12 épocas, learning rate 0.01, regularización 0.05. Semillas 42,7,2026. El contenido pondera por max(rating−2.5,0.1); las valoraciones bajas conservan peso pequeño positivo, no son rechazo explícito.

Fallback: usuario nuevo recibe popularidad suavizada (conteo conserva su baseline). Candidatos: catálogo observado antes del corte, excluyendo vistos; empate por movieId ascendente. Al servir se carga `models/serving.joblib`, generado localmente y no versionado.

Prueba: 74 usuarios; 72 con al menos un positivo, 2 sin positivos, 47 nuevos y 652 positivos fuera del catálogo. NDCG@10=0.25760; cobertura=0.00850. Métricas por usuario y segmento en reports/. Diversidad: promedio de distancia Jaccard entre géneros de pares de títulos; novedad: −log2(frecuencia de calificación). Son aproximaciones basadas en metadatos, no satisfacción medida.

Costos: ajuste compartido incluye preparar contenido/popularidad para todos y el ajuste MF cuando aplica. Latencia es media de generación de listas en lote para usuarios de prueba, no p95 ni garantía de servicio. Tamaño es serialización comprimida del artefacto comparativo, no RAM máxima; los artefactos comparativos son componentes y no incluyen todo el historial de serving. Frontera Pareto descriptiva NDCG/tiempo de ajuste. El tamaño no sustituye una medición de RAM. Métodos deterministas repetidos como controles; desviación entre semillas no es intervalo de confianza. RMSE de híbridos corresponde al componente MF, y en contenido/popularidad a una media de referencia.

Entorno: Python 3.12.12, scikit-learn 1.9.1; CPU Intel i9-13980HX, RAM 16 GB, GPU NVIDIA RTX 4070 Laptop 8 GB, Windows 11. Tiempo total observado 97.3 s. Pico de memoria del kernel: 290.1 MiB (peak working set de Windows desde la creación del proceso; no RAM de todo el sistema). Los tiempos cambian según carga y hardware.

Riesgos: sesgo de popularidad, exceso de similitud por géneros, usuarios/ítems nuevos y cambios de gustos; resultados históricos sin experimento online. No usar para inferir rasgos sensibles. Asistencia Codex y correcciones documentadas en AI_USE_DECLARATION.md y CORRECCIONES_BASE.md. No cargar joblib de terceros no confiables.
