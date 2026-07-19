# Concesión de crédito con IA explicable (B4-T2 · XAI)

Construcción, optimización y **auditoría con XAI** de un modelo de *scoring* de crédito que decide, para cada
cliente, si se le **concede** o se le **deniega** el crédito, bajo dos escenarios de coste distintos.

La variable objetivo es `SeriousDlqin2yrs` (1 = impago grave en los dos años siguientes). Convención:
**predicción 1 = se estima impago → se deniega**; **predicción 0 = se concede**.

| Escenario | Coste FP | Coste FN | Fichero de entrega |
|-----------|:--------:|:--------:|--------------------|
| 1 | 1 | 1 | `data/cs_produccion1.csv` |
| 2 | 1 | 10 | `data/cs_produccion2.csv` |

> **Idea central:** el modelo solo calcula una *probabilidad* de impago; la decisión se toma después
> comparándola con un **umbral** que depende de los costes. Un mismo modelo sirve para los dos escenarios
> cambiando solo el punto de corte.

## Estructura del repositorio

| Archivo | Contenido |
|---------|-----------|
| [`practica_XAI_credito.ipynb`](practica_XAI_credito.ipynb) | **Notebook principal.** EDA, limpieza, modelo, umbral por coste, auditoría XAI, bandit y extensiones (cap. 11). |
| [`ejercicio2_proyecto_XAI.ipynb`](ejercicio2_proyecto_XAI.ipynb) | Ejercicio 2 del proyecto. |
| `guia_practica_XAI.pdf` | Guía explicativa de principio a fin, pensada para defender el trabajo sin base previa en XAI. |
| `taller_XAI.pdf` | Enunciado de la práctica. |
| `data/` | `cs_construccion.csv` (histórico etiquetado), `cs_produccion.csv` (clientes a decidir), diccionario y los dos ficheros de salida. |

## Resultados principales

| Métrica | Valor | Lectura |
|---------|-------|---------|
| AUC del modelo (Gradient Boosting) | ≈ 0.870 | Buen nivel para *scoring*; supera a la logística (≈ 0.854) |
| PSI construcción vs producción | ≈ 0.0005 | Misma población → el coste de validación es extrapolable |
| Umbral escenario 1 / 2 | ≈ 0.50 / ≈ 0.09 | Casi idénticos a los teóricos → modelo bien calibrado |
| Coste medio por cliente (esc. 1 / 2) | ≈ 0.06 / ≈ 0.33 | Escalas distintas: solo comparables dentro del mismo escenario |

## Recorrido del notebook principal

1. **EDA, limpieza y *dataset shift*** — limpieza idéntica en construcción y producción (función `limpiar`), *flags* de ausencia informativa, capado al percentil 99, control de estabilidad con **PSI**.
2. **Modelos** — baseline logístico vs `HistGradientBoostingClassifier` (gestión nativa de ausentes), con curvas ROC/PR, **calibración** y lift.
3. **Umbral por coste** — umbral teórico `c_FP/(c_FP+c_FN)` contrastado con barrido empírico, para cada escenario.
4. **Auditoría XAI** — modelo surrogado (reglas), **SHAP** global y local, **contrafactuales** accionables (DiCE), importancia por permutación, **PDP** y **ALE**, **LIME**, e informe automático de denegación.
5. **Enfoque alternativo** — *Multiarmed Bandit* contextual (Thompson Sampling lineal).
6. **Extensiones (cap. 11)** — ver abajo.

## Extensiones del capítulo 11 (mejoras sobre la línea base)

Divididas según a qué mitad de la evaluación contribuyen (resultados 50 % / análisis 50 %):

**Bloque A — bajar el coste**
- **§11.1 Ingeniería de variables** — `TotalRetrasos`, `IngresoPorDependiente`, `DeudaMensualAbs`, `RatioLineasInmob`.
- **§11.2 Comparación de algoritmos** — HistGB, XGBoost, LightGBM, CatBoost por AUC en CV.
- **§11.2.1 Restricciones de monotonía** — se fuerza el signo del efecto donde el dominio lo exige (más morosidad → más riesgo; más edad/ingreso → menos): regulariza y es defendible ante un regulador.
- **§11.2.2 Test de DeLong** — *p*-valor e IC95 de las diferencias de AUC (¿son reales o ruido?). Confirma que el boosting supera a la logística de forma significativa y mide si el mejor algoritmo alternativo aporta una ventaja real.
- **§11.3 Ponderación del desbalanceo** (`sample_weight`) y **§11.3.1 Calibración explícita** — se demuestra que `sample_weight` descalibra (ECE alto, coste disparado) y que una **recalibración isotónica** lo recupera; para el modelo base, ya calibrado, no hace falta.
- **§11.4 Umbral robusto por CV repetida** y **§11.5 Selección final** — se elige la mejor configuración (incluido el mejor algoritmo) y se **reentrena sobre las 105 000 filas completas**, de donde salen los ficheros de producción definitivos.

**Bloque B — reforzar el análisis**
- **§11.6 Equidad por edad con métricas formales** — *Disparate Impact* (regla del 80 %), *Demographic Parity*, *Equal Opportunity* y *FPR gap*, comparando el modelo con y sin la variable edad.
- **§11.7 Intervalo de confianza del coste** (*bootstrap*), **§11.8 ALE**, **§11.9 LIME**, **§11.10 sesgo de selección y *reject inference***.

## Cómo ejecutar

Requiere Python 3.10+ y las siguientes librerías:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm \
            shap lime dice-ml xgboost lightgbm catboost
```

Después, abrir `practica_XAI_credito.ipynb` y ejecutar **Restart & Run All**. El notebook se ejecuta de
principio a fin sin errores y regenera `data/cs_produccion1.csv` y `data/cs_produccion2.csv` (45 000
decisiones cada uno) en la sección §11.5.
