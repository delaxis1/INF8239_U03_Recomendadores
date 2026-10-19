"""Documentación e informe a partir de artefactos reales, sin reentrenar."""

import json, shutil
from pathlib import Path
import pandas as pd
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib.font_manager import findfont, FontProperties

for name, weight, style in [
    ("ReportSans", "normal", "normal"),
    ("ReportSans-Bold", "bold", "normal"),
    ("ReportSans-Italic", "normal", "oblique"),
    ("ReportSans-BoldItalic", "bold", "oblique"),
]:
    pdfmetrics.registerFont(
        TTFont(
            name,
            findfont(FontProperties(family="DejaVu Sans", weight=weight, style=style)),
        )
    )
pdfmetrics.registerFontFamily(
    "ReportSans",
    normal="ReportSans",
    bold="ReportSans-Bold",
    italic="ReportSans-Italic",
    boldItalic="ReportSans-BoldItalic",
)
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/delaxis1/INF8239_U03_Recomendadores"
NAMES = {
    "count": "Conteo",
    "popularity": "Popularidad suavizada",
    "content": "Contenido (géneros)",
    "mf10": "MF · 10 factores",
    "mf20": "MF · 20 factores",
    "hybrid25": "Híbrido · α=0.25",
    "hybrid75": "Híbrido · α=0.75",
}


def main():
    out = ROOT / "reports"
    meta = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    decision = json.loads((out / "decision_locked.json").read_text(encoding="utf-8"))
    d = meta["dataset"]
    split = meta["split"]
    table = pd.read_csv(out / "main_results.csv", index_col="model")
    val = (
        pd.read_csv(out / "validation_runs.csv")
        .groupby("model")[["ndcg", "coverage"]]
        .mean()
    )
    chosen = meta["chosen"]
    row = table.loc[chosen]
    baseline = table.loc["popularity"]
    best = table.ndcg.idxmax()
    segments = pd.read_csv(out / "segments.csv").set_index(["model", "segment"])
    warm_count = segments.loc[("count", "con_historial"), "ndcg"]
    warm_hybrid = segments.loc[("hybrid75", "con_historial"), "ndcg"]
    cold = int(row.cold_users)
    nusers = int(row.users)
    eligible = int(row.users_positive)
    fallback = "conteo de popularidad" if chosen == "count" else "popularidad suavizada"
    memory = (
        json.loads((out / "memory.json").read_text(encoding="utf-8"))
        if (out / "memory.json").exists()
        else {}
    )
    memory_text = (
        f"Pico de memoria del kernel: {memory['kernel_peak_working_set_mib']:.1f} MiB (peak working set de Windows desde la creación del proceso; no RAM de todo el sistema)."
        if memory
        else "Memoria no medida en esta ejecución."
    )
    conclusion = f"""# Conclusión personal — borrador para revisión del autor

En esta práctica quise comprobar si un recomendador más complejo realmente ayudaba a ordenar mejor las películas. Partí de MovieLens y comparé siete configuraciones, desde un conteo de popularidad hasta factorización e híbridos. Lo primero que tuve que aclarar fue qué significaba acertar: consideré relevante una película con cuatro estrellas o más y evalué listas de diez títulos. Predecir una calificación cercana a la real y recomendar una película que aparezca entre los intereses futuros son tareas relacionadas, pero no equivalentes.

Para evitar usar información del futuro, separé las interacciones por fecha en entrenamiento, validación y prueba. Esta decisión hizo más exigente la evaluación: de los {nusers} usuarios de prueba, {cold} todavía no tenían historial al momento del ajuste final. A esos usuarios les ofrecí {fallback}. Por eso no puedo interpretar sus listas como recomendaciones personalizadas. También conservé los positivos que estaban fuera del catálogo como fallos, porque eliminarlos habría dado una impresión demasiado favorable del sistema.

Con la regla fijada antes de revisar prueba, seleccioné {NAMES[chosen]}. En validación alcanzó NDCG@10 de {val.loc[chosen, "ndcg"]:.4f}; en prueba obtuvo {row.ndcg:.4f} y cobertura de {row.coverage:.1%}. La popularidad suavizada obtuvo {baseline.ndcg:.4f}. El mayor NDCG observado en prueba correspondió a {NAMES[best]}, con {table.loc[best, "ndcg"]:.4f}, pero ese resultado posterior no se utilizó para cambiar la elección. Para mí, separar ambas decisiones es una parte importante del ejercicio: de lo contrario estaría ajustando el sistema a la misma prueba con la que pretendo justificarlo.

Entre los usuarios con historial, el híbrido α0.75 obtuvo NDCG {warm_hybrid:.4f} frente a {warm_count:.4f} del conteo. Ese contraste me hizo bajar las expectativas de encontrar un ganador universal. La revisión de los perfiles me ayudó a entender los límites de cada enfoque. Con pocos datos del usuario, el respaldo de popularidad es fácil de explicar, aunque repite títulos. El contenido aprovecha géneros y puede ampliar el catálogo recomendado, pero conocer los géneros no significa conocer toda la película. La factorización incorpora patrones compartidos y añade tiempo de ajuste; mezclarla con contenido tampoco garantiza una mejora. La cobertura y la diversidad sirven para describir esas diferencias, sin reemplazar la relevancia.

Me quedo con un procedimiento que puedo volver a ejecutar y revisar: descarga con huella, cortes guardados, candidatos comunes, tres semillas y pruebas de los casos delicados. Los resultados corresponden a este conjunto histórico y a este protocolo; no demuestran cómo responderían personas reales en una aplicación. Antes de llevarlo a producción comprobaría el arranque en frío, contrastaría otras ventanas temporales y mediría utilidad con usuarios, manteniendo separada una evaluación final.
"""
    (out / "conclusion.md").write_text(conclusion, encoding="utf-8")
    (ROOT / "docs/DatasetCard.md").write_text(
        f"""# Dataset Card — MovieLens Latest Small

Fuente: GroupLens, {d["ratings"]:,} calificaciones, {d["users"]} usuarios y {d["movies"]:,} películas; {d["rated_movies"]:,} con calificaciones. Densidad: {d["density"]:.4%}. Rango observado: {d["date_min"]} a {d["date_max"]}.

Descarga: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
SHA-256: `{d["sha256"]}`. Registro: `data_source.json`. Latest-small puede cambiar; una huella diferente bloquea la ejecución hasta revisión explícita. Conservar el ZIP original local permite repetir esta versión.

Términos: README original de GroupLens dentro del ZIP, incluidos reconocimiento, uso de investigación y redistribución bajo las mismas condiciones y autorización para uso comercial. No es una licencia CC por defecto. No se publican CSV originales ni modelos que incorporen historiales en este repositorio.

Esquema: ratings.csv contiene userId (identificador anónimo), movieId (clave), rating (0.5 a 5 en pasos de 0.5), timestamp (Unix UTC). movies.csv contiene movieId, title y genres separados por `|`. Los identificadores no son atributos predictivos por sí mismos. Los títulos/géneros no tienen fecha de disponibilidad verificable; se limita el catálogo a ítems ya valorados antes de cada corte, aunque persiste esa limitación del metadato estático.

Auditoría: ausencia de nulos, duplicados de usuario/ítem, claves duplicadas de películas y claves huérfanas; rating en rango y timestamps positivos. Duplicados exactos: {d["duplicates"]}. Ver `reports/audit.json`.

Partición global por cuantiles de tiempo: {split["train"]:,} entrenamiento, {split["validation"]:,} validación, {split["test"]:,} prueba. Cortes Unix {split["cuts"]}. SHA-256 de asignaciones: `{split["assignment_sha256"]}`. No se separan timestamps iguales. Datos/índices locales en data/raw. Ajuste final usa entrenamiento + validación después de fijar configuración.

Sesgos: selección de usuarios voluntarios, exposición desconocida y ausencia de negativos reales. No observado no significa no gustado. No hay variables para una evaluación demográfica; no inferir equidad. Los relevantes fuera del catálogo cuentan como fallos. Los usuarios sin positivos no aportan a precisión/recall/NDCG/HitRate; sí se contabilizan. Referencia: Harper y Konstan (2015), DOI 10.1145/2827872.
""",
        encoding="utf-8",
    )
    (ROOT / "docs/SystemCard.md").write_text(
        f"""# System Card — recomendación Top-10

Uso: demostración educativa local; no asesoramiento ni prueba de impacto real. Configuración elegida: **{NAMES[chosen]}**, exclusivamente con validación. Regla: NDCG a 0.005 del máximo; mayor cobertura y luego menor ajuste. Registro previo a prueba: `reports/decision_locked.json` ({decision["locked_utc"]}).

Siete configuraciones: conteo, media bayesiana suavizada, perfil de géneros TF-IDF/coseno, MF10, MF20, híbridos α0.25/0.75. α pesa la señal colaborativa, con scores normalizados min-max sobre el mismo catálogo; (1−α) pesa contenido. MF sin sesgos de usuario/ítem del código base, 12 épocas, learning rate 0.01, regularización 0.05. Semillas 42,7,2026. El contenido pondera por max(rating−2.5,0.1); las valoraciones bajas conservan peso pequeño positivo, no son rechazo explícito.

Fallback: usuario nuevo recibe popularidad suavizada (conteo conserva su baseline). Candidatos: catálogo observado antes del corte, excluyendo vistos; empate por movieId ascendente. Al servir se carga `models/serving.joblib`, generado localmente y no versionado.

Prueba: {nusers} usuarios; {eligible} con al menos un positivo, {int(row.users_no_positive)} sin positivos, {cold} nuevos y {int(row.unreachable_positives)} positivos fuera del catálogo. NDCG@10={row.ndcg:.5f}; cobertura={row.coverage:.5f}. Métricas por usuario y segmento en reports/. Diversidad: promedio de distancia Jaccard entre géneros de pares de títulos; novedad: −log2(frecuencia de calificación). Son aproximaciones basadas en metadatos, no satisfacción medida.

Costos: ajuste compartido incluye preparar contenido/popularidad para todos y el ajuste MF cuando aplica. Latencia es media de generación de listas en lote para usuarios de prueba, no p95 ni garantía de servicio. Tamaño es serialización comprimida del artefacto comparativo, no RAM máxima; los artefactos comparativos son componentes y no incluyen todo el historial de serving. Frontera Pareto descriptiva NDCG/tiempo de ajuste. El tamaño no sustituye una medición de RAM. Métodos deterministas repetidos como controles; desviación entre semillas no es intervalo de confianza. RMSE de híbridos corresponde al componente MF, y en contenido/popularidad a una media de referencia.

Entorno: Python {meta["python"]}, scikit-learn {meta["sklearn"]}; CPU Intel i9-13980HX, RAM 16 GB, GPU NVIDIA RTX 4070 Laptop 8 GB, Windows 11. Tiempo total observado {meta["seconds"]:.1f} s. {memory_text} Los tiempos cambian según carga y hardware.

Riesgos: sesgo de popularidad, exceso de similitud por géneros, usuarios/ítems nuevos y cambios de gustos; resultados históricos sin experimento online. No usar para inferir rasgos sensibles. Asistencia Codex y correcciones documentadas en AI_USE_DECLARATION.md y CORRECCIONES_BASE.md. No cargar joblib de terceros no confiables.
""",
        encoding="utf-8",
    )
    results_md = (
        table[["ndcg", "hit", "precision", "recall", "coverage", "fit_s"]]
        .rename(index=NAMES)
        .to_markdown(floatfmt=".4f")
    )
    readme = f"""# INF8239_U03_Recomendadores

Alexis De La Cruz · 100805370 · INF-8239 · Unidad 03, Ejercicio 05.

Motor Top-10 sobre MovieLens. Base oficial LAB08/LAB09 del docente Edwin Ramón José Nolasco, conservada en el historial; adaptaciones en [CORRECCIONES_BASE](docs/CORRECCIONES_BASE.md). Notebook ejecutado: [05_recomendador](notebooks/05_recomendador.ipynb). Informe: [PDF](reports/informe_ejercicio05.pdf).

## Resultado
Elección fijada en validación: **{NAMES[chosen]}**. Prueba final, promedios de tres semillas:

{results_md}

El mayor NDCG en prueba no implica nueva selección. Baselines deterministas se repiten como controles. Ver [System Card](docs/SystemCard.md) para denominadores, RMSE, tiempos y limitaciones. Se evalúa todo el catálogo observado antes del corte, excluyendo vistos; no se restringe primero a un Top-100 colaborativo.

## Reproducir (PowerShell)
Desde esta carpeta, con [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --frozen --python 3.12.12
uv run python -m pytest -q
uv run python scripts/execute_notebook.py
uv run streamlit run app/streamlit_app.py
```

El notebook descarga, audita, selecciona en validación, reajusta, evalúa prueba y genera tablas, figuras, tarjetas e informe. No requiere intervención ni CSV manual. Tiempo observado: {meta["seconds"] / 60:.1f} minutos (varía con equipo). Para regenerar solo documentos: `uv run python scripts/report_exercise05.py`.

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

MovieLens SHA-256: `{d["sha256"]}`. Términos de GroupLens en el README del ZIP. Datos originales, particiones por fila y modelos permanecen locales por la política de entrega de la práctica. No se redistribuye MovieLens. `latest-small` puede cambiar: una huella distinta obliga a revisar la fuente; no se acepta silenciosamente.

Cortes globales 80/10/10: {split["train"]}/{split["validation"]}/{split["test"]} filas. Relevante ≥4, k=10, MF10/20 con 12 épocas y semillas42/7/2026. Selección en validación con tolerancia fija0.005 NDCG, mayor cobertura y menor costo. Detalles del cambio frente al leave-last del ejemplo en CORRECCIONES_BASE.

Pruebas cubren descarga segura/esquema, MF/contenido, temporalidad, métricas, positivos fuera del catálogo, cold start, vistos, selección y Pareto. Estructura: src/, tests/, notebooks/, scripts/, app/, docs/, reports/; data/raw y models se crean localmente.

Fuentes: [mandato E05](https://uasdvirtual.uasd.edu.do/diplomados/mod/assign/view.php?id=238698), [manual](https://uasdvirtual.uasd.edu.do/diplomados/mod/book/view.php?id=238956), [base oficial](https://uasdvirtual.uasd.edu.do/diplomados/mod/resource/view.php?id=238958), [aclaraciones](https://uasdvirtual.uasd.edu.do/diplomados/mod/book/view.php?id=239974), [GroupLens](https://grouplens.org/datasets/movielens/latest/), [Harper y Konstan,2015](https://doi.org/10.1145/2827872).

## Asistencia de IA
Codex (OpenAI) asistió en código, documentación y comprobaciones. Se distingue el código base y sus correcciones; prompts y controles en AI_USE_DECLARATION.md. La conclusión es un borrador que el autor debe revisar antes de entregar. No se atribuye revisión humana no realizada.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    pdf = out / "informe_ejercicio05.pdf"
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyOwn",
            fontName="ReportSans",
            fontSize=10,
            leading=14,
            spaceAfter=8,
            textColor=colors.HexColor("#263445"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallOwn",
            parent=styles["BodyOwn"],
            fontSize=8,
            leading=11,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadingOwn",
            fontName="ReportSans-Bold",
            fontSize=14,
            leading=17,
            spaceBefore=10,
            spaceAfter=9,
            textColor=colors.HexColor("#185594"),
        )
    )
    story = []

    def p(s, small=False):
        story.append(Paragraph(s, styles["SmallOwn" if small else "BodyOwn"]))

    def heading(s):
        story.append(Paragraph(s, styles["HeadingOwn"]))

    def image(name, width=480):
        path = out / "figures" / name
        w, h = ImageReader(str(path)).getSize()
        story.append(Image(str(path), width=width, height=width * h / w))
        story.append(Spacer(1, 6))

    def tab(rows, widths):
        data = [
            [Paragraph(escape(str(c)), styles["SmallOwn"]) for c in r] for r in rows
        ]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#2a78d6")),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d7dee5")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 8))

    heading("Unidad 03 · Ejercicio 05")
    p(
        "<b>Recomendación Top-10 con MovieLens</b><br/>Alexis De La Cruz · Matrícula 100805370<br/>INF-8239 · Ciencia de Datos II · UASD<br/>Docente: Edwin Ramón José Nolasco"
    )
    p(f'Repositorio privado: <link href="{REPO}" color="#185594">{REPO}</link>', True)
    heading("Datos y pregunta")
    p(
        f"¿Qué combinación de popularidad, contenido y factorización logra un ranking útil sin un costo de ajuste innecesario? MovieLens contiene {d['ratings']:,} calificaciones de {d['users']} usuarios y {d['movies']:,} películas. Se usan calificaciones, fecha y géneros; no se introducen atributos demográficos."
    )
    p(
        "Fuente: GroupLens, MovieLens Latest Small. Se respetan los términos del README original y se entrega descarga reproducible, sin redistribuir los datos. Auditoría sin nulos, claves huérfanas ni interacciones usuario/película duplicadas.",
        True,
    )
    p("SHA-256 del ZIP: " + d["sha256"], True)
    heading("Protocolo y selección")
    p(
        f"Corte temporal global: {split['train']:,} filas para ajuste, {split['validation']:,} para validación y {split['test']:,} para prueba. Los timestamps iguales quedan juntos. Se difiere del leave-last por usuario de la base para impedir que entren calificaciones de otros usuarios posteriores al corte evaluado. El catálogo contiene solo películas ya calificadas al corte; se excluyen las vistas."
    )
    p(
        "Relevante: ≥4 estrellas. Listas de diez títulos. MF con 10/20 factores, 12 épocas y semillas 42, 7 y 2026. Híbridos con peso colaborativo α=0.25/0.75 y scores normalizados sobre el mismo catálogo. Regla previa: a ≤0.005 del mejor NDCG de validación, preferir cobertura y luego menor ajuste."
    )
    p(
        f"Elección registrada antes de prueba: <b>{NAMES[chosen]}</b> (NDCG validación {val.loc[chosen, 'ndcg']:.4f}). Después se reajusta con entrenamiento + validación. No se cambia la elección por la tabla final."
    )
    heading("Entorno y reproducción")
    p(
        f"Python {meta['python']}; scikit-learn {meta['sklearn']}; uv.lock. Intel Core i9-13980HX; RAM 16 GB; NVIDIA RTX 4070 Laptop, 8 GB; Windows 11. Duración observada: {meta['seconds'] / 60:.1f} min. {memory_text} Notebook y 19 pruebas automatizadas acompañan el repositorio.",
        True,
    )
    story.append(PageBreak())
    heading("Resultados de prueba · promedio de tres semillas")
    rows = [
        ["Configuración", "NDCG@10", "Hit@10", "P@10", "R@10", "Cobertura", "Ajuste(s)"]
    ]
    for key, r in table.iterrows():
        rows.append(
            [
                NAMES[key],
                f"{r.ndcg:.4f}",
                f"{r.hit:.3f}",
                f"{r.precision:.3f}",
                f"{r.recall:.3f}",
                f"{r.coverage:.1%}",
                f"{r.fit_s:.2f}",
            ]
        )
    tab(rows, [133, 62, 48, 45, 45, 70, 67])
    p(
        f"Métricas de ranking sobre {eligible} usuarios con positivos; {int(row.users_no_positive)} sin positivos se cuentan aparte. Usuarios de prueba: {nusers}; nuevos sin historial: {cold}; relevantes fuera del catálogo: {int(row.unreachable_positives)} (cuentan como fallos). Cobertura se calcula sobre las listas de todos los usuarios y el catálogo disponible.",
        True,
    )
    image("ranking_coverage.png")
    p(
        "Figura 1. Ranking y amplitud del catálogo. Candidatos comunes para todos los métodos. La variación entre semillas no constituye intervalo de confianza; los controles deterministas se repiten sin aleatoriedad.",
        True,
    )
    p(
        f"RMSE del componente MF20: {table.loc['mf20', 'rmse']:.4f}; MF10: {table.loc['mf10', 'rmse']:.4f}. En pares desconocidos se usa media por ítem/global. RMSE no evalúa directamente el orden Top-10 y, en híbridos, corresponde a su componente MF.",
        True,
    )
    p(
        "Los tiempos son medianas: preparación compartida más ajuste MF cuando aplica; latencia es media por usuario en lote. Tamaño del artefacto comprimido se informa en las tablas del repositorio y no equivale a memoria RAM máxima.",
        True,
    )
    story.append(PageBreak())
    heading("Costo, cobertura y límites")
    image("pareto.png", 450)
    p(
        "Figura 2. Frontera de utilidad/costo de ajuste en prueba (descriptiva). Azul: no dominados bajo ambos criterios; gris: dominados. La elección permanece fijada con validación.",
        True,
    )
    profiles = pd.read_csv(out / "profiles.csv")
    rows = [["Perfil", "Historial", "Primeras recomendaciones"]]
    for key, g in profiles.groupby("profile", sort=False):
        rows.append([key, int(g.history.iloc[0]), "; ".join(g.title.head(3))])
    tab(rows, [65, 55, 350])
    p(
        f"En usuarios con historial, NDCG del híbrido α0.75={warm_hybrid:.4f}, frente a {warm_count:.4f} del conteo. El ranking global está influido por el alto número de usuarios nuevos; no permite afirmar que el conteo sea superior en todos los segmentos.",
        True,
    )
    p(
        "El perfil nuevo es sintético y usa el respaldo de popularidad del método elegido; los otros son extremos de historial observado. La aplicación permite explorar Top-10 por usuario y similitud por título. Contenido usa géneros y puede sobreespecializar las listas. Sin exposición registrada ni datos demográficos, no se demuestra satisfacción ni equidad entre grupos."
    )
    p(
        f"Diversidad media del sistema elegido: {row.diversity:.3f} (distancia Jaccard entre géneros); novedad: {row.novelty:.3f} bits, aproximada por frecuencia inversa de calificación. No son valoraciones humanas.",
        True,
    )
    story.append(PageBreak())
    heading("Conclusión")
    for paragraph in conclusion.split("\n\n")[1:]:
        p(escape(paragraph))
    heading("Referencias y nota del autor")
    p(
        'Harper, F. M., &amp; Konstan, J. A. (2015). The MovieLens datasets: History and context. <i>ACM Transactions on Interactive Intelligent Systems, 5</i>(4), artículo 19. <link href="https://doi.org/10.1145/2827872">https://doi.org/10.1145/2827872</link>',
        True,
    )
    p(
        'GroupLens. (s. f.). <i>MovieLens Latest Datasets</i>. <link href="https://grouplens.org/datasets/movielens/latest/">grouplens.org/datasets/movielens/latest/</link>. Nolasco, E. R. J. (s. f.). <i>Manual y proyecto base LAB08–LAB09, INF-8239</i> [Material de curso]. UASD Virtual.',
        True,
    )
    p(
        "Se utilizó Codex (OpenAI) como apoyo de programación, documentación y verificación. El código base del docente y las correcciones se identifican en Git; prompts y controles están en AI_USE_DECLARATION.md. La conclusión es un borrador sujeto a revisión personal antes de la entrega.",
        True,
    )

    def page(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#185594"))
        canvas.setLineWidth(2)
        canvas.line(48, 800, 547, 800)
        canvas.setFont("ReportSans", 8)
        canvas.setFillColor(colors.HexColor("#536477"))
        canvas.drawString(48, 28, "Alexis De La Cruz · INF-8239 · U03.E05")
        canvas.drawRightString(547, 28, str(doc.page))

    SimpleDocTemplate(
        str(pdf),
        pagesize=(595, 842),
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=45,
        title="U03.E05 · Recomendadores · Alexis De La Cruz",
        author="Alexis De La Cruz",
    ).build(story, onFirstPage=page, onLaterPages=page)
    delivery = ROOT.parent / "Unidad03-Ejercicio05-Recomendadores"
    delivery.mkdir(exist_ok=True)
    shutil.copy2(pdf, delivery / "100805370_DeLaCruz_U03E05_Recomendadores.pdf")
    print("Documentos e informe generados:", pdf)


if __name__ == "__main__":
    main()
