# Procedencia y correcciones de la base

Base académica: Edwin Ramón José Nolasco, ZIP oficial del LAB08/LAB09.
Se conserva el código recibido en el commit c0f6565 (el primero contiene la preparación del proyecto).

1. El híbrido original reordena solo los Top-100 colaborativos: el experimento nuevo evalúa
   todos los métodos contra el mismo catálogo observado antes del corte, sin ítems vistos.
2. El perfil de géneros se convierte a ndarray: sklearn moderno rechaza numpy.matrix.
3. La última interacción por usuario evita futuro individual, pero puede incorporar el futuro
   de otros usuarios. El experimento principal añade cortes globales 80/10/10 por timestamp.
   Se conserva la función original temporal_leave_one_out y se informa su limitación.
4. La configuración completa, incluidos los alpha, se elige en validación con regla previa: dentro de .005 del mejor NDCG, mayor cobertura,
   luego menor costo. El conjunto de prueba no interviene.
5. Relevancia rating >=4; usuarios sin relevantes se informan aparte; positivos fuera del catálogo
   se cuentan como fallos. Precision@10 usa denominador fijo 10 aunque haya menos candidatos.
6. Empates se resuelven por movieId ascendente para reproducibilidad.

7. Popularidad suavizada se evalúa sobre todo el catálogo (sin descartar por mínimo de votos) para conservar candidatos homogéneos; el Top-10 ilustrativo de la base conserva su filtro original. El baseline conteo usa conteo también en usuarios nuevos; los métodos personalizados recurren a popularidad suavizada.
