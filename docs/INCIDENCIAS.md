# Bitácora del entorno

19-10-2026: Python 3.12.13 de uv falló al importar enum. Se instaló Python 3.12.12 para
este proyecto, sin modificar los proyectos anteriores.
La caché previa de uv tenía archivos faltantes (fonttools y pygments). Se usó una caché nueva
y reinstalación con modo copy. El lock del proyecto registra las versiones utilizadas.
No se modificaron protecciones de seguridad de Windows.

## Rutas largas de Windows
La importación de pyarrow falló con DLL load failed / filename too long dentro de la ruta del curso. Se creó el entorno en `%USERPROFILE%/.venvs/u03-rec/.venv`, usando UV_PROJECT_ENVIRONMENT. Las 16 pruebas pasaron. No se cambió la seguridad del equipo.
