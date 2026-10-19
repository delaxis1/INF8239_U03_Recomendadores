# Dataset Card — MovieLens Latest Small

Fuente: GroupLens, 100,836 calificaciones, 610 usuarios y 9,742 películas; 9,724 con calificaciones. Densidad: 1.6968%. Rango observado: 1996-03-29 18:36:55+00:00 a 2018-09-24 14:27:30+00:00.

Descarga: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
SHA-256: `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`. Registro: `data_source.json`. Latest-small puede cambiar; una huella diferente bloquea la ejecución hasta revisión explícita. Conservar el ZIP original local permite repetir esta versión.

Términos: README original de GroupLens dentro del ZIP, incluidos reconocimiento, uso de investigación y redistribución bajo las mismas condiciones y autorización para uso comercial. No es una licencia CC por defecto. No se publican CSV originales ni modelos que incorporen historiales en este repositorio.

Esquema: ratings.csv contiene userId (identificador anónimo), movieId (clave), rating (0.5 a 5 en pasos de 0.5), timestamp (Unix UTC). movies.csv contiene movieId, title y genres separados por `|`. Los identificadores no son atributos predictivos por sí mismos. Los títulos/géneros no tienen fecha de disponibilidad verificable; se limita el catálogo a ítems ya valorados antes de cada corte, aunque persiste esa limitación del metadato estático.

Auditoría: ausencia de nulos, duplicados de usuario/ítem, claves duplicadas de películas y claves huérfanas; rating en rango y timestamps positivos. Duplicados exactos: 0. Ver `reports/audit.json`.

Partición global por cuantiles de tiempo: 80,669 entrenamiento, 10,083 validación, 10,084 prueba. Cortes Unix {'train_end': 1458635171, 'validation_end': 1498029447}. SHA-256 de asignaciones: `688196fecf11dcbdbfaa22df2dfcd2f2550b98f1e31c921a1029c72ec0c82372`. No se separan timestamps iguales. Datos/índices locales en data/raw. Ajuste final usa entrenamiento + validación después de fijar configuración.

Sesgos: selección de usuarios voluntarios, exposición desconocida y ausencia de negativos reales. No observado no significa no gustado. No hay variables para una evaluación demográfica; no inferir equidad. Los relevantes fuera del catálogo cuentan como fallos. Los usuarios sin positivos no aportan a precisión/recall/NDCG/HitRate; sí se contabilizan. Referencia: Harper y Konstan (2015), DOI 10.1145/2827872.
