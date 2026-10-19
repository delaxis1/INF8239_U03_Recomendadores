# INF8239_U03_Recomendadores

Alexis De La Cruz · 100805370 · INF-8239 · Unidad 03, Ejercicio 05.

Motor Top-10 sobre MovieLens. Base oficial LAB08/LAB09 del docente Edwin Ramón José Nolasco, conservada en el historial; adaptaciones en [CORRECCIONES_BASE](docs/CORRECCIONES_BASE.md). Notebook ejecutado: [05_recomendador](notebooks/05_recomendador.ipynb). Informe: [PDF](reports/informe_ejercicio05.pdf).

## Resultado
Elección fijada en validación: **Conteo**. Prueba final, promedios de tres semillas:

| model                 |   ndcg |    hit |   precision |   recall |   coverage |   fit_s |
|:----------------------|-------:|-------:|------------:|---------:|-----------:|--------:|
| Contenido (géneros)   | 0.2123 | 0.5694 |      0.1778 |   0.0462 |     0.0151 |  0.0645 |
| Conteo                | 0.2576 | 0.6250 |      0.2264 |   0.0613 |     0.0085 |  0.0645 |
| Híbrido · α=0.25      | 0.2262 | 0.6250 |      0.1903 |   0.0488 |     0.0201 |  7.6235 |
| Híbrido · α=0.75      | 0.2466 | 0.7037 |      0.2069 |   0.0564 |     0.0164 |  7.6235 |
| MF · 10 factores      | 0.2328 | 0.6620 |      0.1949 |   0.0542 |     0.0131 |  8.4741 |
| MF · 20 factores      | 0.2352 | 0.6806 |      0.1986 |   0.0554 |     0.0149 |  7.6235 |
| Popularidad suavizada | 0.2434 | 0.6528 |      0.2042 |   0.0560 |     0.0052 |  0.0645 |

El mayor NDCG en prueba no implica nueva selección. Baselines deterministas se repiten como controles. Ver [System Card](docs/SystemCard.md) para denominadores, RMSE, tiempos y limitaciones. Se evalúa todo el catálogo observado antes del corte, excluyendo vistos; no se restringe primero a un Top-100 colaborativo.

## Reproducir (PowerShell)
Desde esta carpeta, con [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --frozen --python 3.12.12
uv run python -m pytest -q
uv run python scripts/execute_notebook.py
uv run streamlit run app/streamlit_app.py
```

El notebook descarga, audita, selecciona en validación, reajusta, evalúa prueba y genera tablas, figuras, tarjetas e informe. No requiere intervención ni CSV manual. Tiempo observado: 1.6 minutos (varía con equipo). Para regenerar solo documentos: `uv run python scripts/report_exercise05.py`.

Si Windows informa DLL/ruta demasiado larga, usar un entorno externo corto **en todas las sesiones**:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:USERPROFILE/.venvs/u03-rec/.venv"
uv sync --frozen --python 3.12.12 --link-mode copy
uv run python -m pytest -q
uv run python scripts/execute_notebook.py
uv run streamlit run app/streamlit_app.py
```

Seleccionar ese mismo Python como kernel en VS Code. `uv.lock` fija dependencias; `requirements.txt` es exportación alternativa. El notebook ejecutado conserva la ruta real del intérprete. No hace falta modificar Smart App Control.

## Datos, protocolo y fuentes
[Dataset Card](docs/DatasetCard.md), [System Card](docs/SystemCard.md), [declaración de IA](AI_USE_DECLARATION.md) y [incidencias](docs/INCIDENCIAS.md).

MovieLens SHA-256: `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`. Términos de GroupLens en el README del ZIP. Datos originales, particiones por fila y modelos permanecen locales por la política de entrega de la práctica. No se redistribuye MovieLens. `latest-small` puede cambiar: una huella distinta obliga a revisar la fuente; no se acepta silenciosamente.

Cortes globales 80/10/10: 80669/10083/10084 filas. Relevante ≥4, k=10, MF10/20 con 12 épocas y semillas42/7/2026. Selección en validación con tolerancia fija0.005 NDCG, mayor cobertura y menor costo. Detalles del cambio frente al leave-last del ejemplo en CORRECCIONES_BASE.

Pruebas cubren descarga segura/esquema, MF/contenido, temporalidad, métricas, positivos fuera del catálogo, cold start, vistos, selección y Pareto. Estructura: src/, tests/, notebooks/, scripts/, app/, docs/, reports/; data/raw y models se crean localmente.

Fuentes: [mandato E05](https://uasdvirtual.uasd.edu.do/diplomados/mod/assign/view.php?id=238698), [manual](https://uasdvirtual.uasd.edu.do/diplomados/mod/book/view.php?id=238956), [base oficial](https://uasdvirtual.uasd.edu.do/diplomados/mod/resource/view.php?id=238958), [aclaraciones](https://uasdvirtual.uasd.edu.do/diplomados/mod/book/view.php?id=239974), [GroupLens](https://grouplens.org/datasets/movielens/latest/), [Harper y Konstan,2015](https://doi.org/10.1145/2827872).

## Asistencia de IA
Codex (OpenAI) asistió en código, documentación y comprobaciones. Se distingue el código base y sus correcciones; prompts y controles en AI_USE_DECLARATION.md. La conclusión es un borrador que el autor debe revisar antes de entregar. No se atribuye revisión humana no realizada.
