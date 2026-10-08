# Informe técnico: Telco Customer Churn

**Asignatura:** MLY1101 Machine Learning\
**Caso:** Predicción de abandono de clientes\
**Metodología:** CRISP-DM

## 1. Resumen ejecutivo

Este proyecto desarrolla una solución de Machine Learning para estudiar
el abandono de clientes de telecomunicaciones (`Churn`) y descubrir
segmentos con características similares. El flujo contempla separación
Train/Test, análisis exploratorio (EDA), limpieza y transformación de
datos, comparación de modelos supervisados, optimización con Optuna y
segmentación no supervisada con K-Means.

La separación de datos se realizó antes del EDA: 5.634 registros para
Train y 1.409 para Test, con `test_size=0.20`, `random_state=42` y
estratificación por `Churn`. La tasa de abandono fue de 26,54% en ambos
subconjuntos. Test se reservó para evaluación final.

En Test, Random Forest alcanzó Accuracy 76,51%, Precision 53,94%, Recall
78,61%, F1-score 63,98% y ROC-AUC 0,8412. Regresión Logística alcanzó
Accuracy 73,95%, Precision 50,60%, Recall 78,34%, F1-score 61,49% y
ROC-AUC 0,8432. Random Forest fue mejor en las métricas calculadas con
el umbral predeterminado, mientras Regresión Logística obtuvo un ROC-AUC
marginalmente superior.

Para el análisis no supervisado, K-Means con dos clusters fue la
alternativa seleccionada inicialmente, con Silhouette 0,2492. Las tasas
de abandono descriptivas fueron aproximadamente 7% en Cluster 0 y 32% en
Cluster 1, frente al 26,54% general de Train. Los clusters describen
segmentos y no son predicciones individuales ni evidencia causal.

## 2. Problema de negocio y objetivos

La pérdida de clientes puede reducir ingresos y aumentar los costos de
adquisición. La empresa necesita comprender patrones asociados al
abandono para orientar mejor las acciones de retención.

**Objetivo general:** desarrollar y evaluar una solución reproducible de
Machine Learning para clasificar el abandono de clientes y explorar
segmentos útiles para el análisis comercial.

**Objetivos específicos:** 1. Examinar estructura, distribución y
calidad de los datos. 2. Separar Train y Test antes del EDA y del
modelamiento. 3. Limpiar y transformar variables para los algoritmos. 4.
Entrenar y comparar dos modelos supervisados. 5. Optimizar
hiperparámetros con validación cruzada sobre Train. 6. Evaluar los
modelos en Test con métricas apropiadas. 7. Aplicar clustering sin usar
`Churn` para formar los grupos. 8. Documentar limitaciones,
consideraciones éticas y recomendaciones.

### Métricas y KPIs técnicos

  -----------------------------------------------------------------------
  Métrica                             Interpretación
  ----------------------------------- -----------------------------------
  Accuracy                            Proporción total de predicciones
                                      correctas.

  Precision                           De los clientes predichos como
                                      abandono, proporción que realmente
                                      abandona.

  Recall                              De los clientes que abandonan,
                                      proporción detectada.

  F1-score                            Equilibrio entre Precision y
                                      Recall.

  ROC-AUC                             Capacidad de discriminar clases a
                                      través de distintos umbrales.

  Silhouette                          Calidad relativa de cohesión y
                                      separación de los clusters.
  -----------------------------------------------------------------------

No se definió una meta numérica de negocio previa. Las métricas permiten
comparar modelos; una implementación real debe ponderar el costo de
campañas innecesarias y el costo de no detectar clientes que
abandonarán.

## 3. Datos y separación Train/Test

La variable objetivo es `Churn` (`No`/`Yes`) y `customerID` es un
identificador, no un predictor. La partición se realizó en
`notebooks/00_separacion_train_test.ipynb` con una división 80/20,
`random_state=42` y `stratify=y`.

  Conjunto     Filas   Columnas originales
  ---------- ------- ---------------------
  Train        5.634                    21
  Test         1.409                    21
  Total        7.043                    21

La estratificación ayuda a conservar una distribución de clases similar.
La tasa de abandono fue 26,54% en Train y Test. También se comprobó que
no hubiera `customerID` compartidos entre los subconjuntos.

La separación previa al EDA reduce el riesgo de fuga de información. El
EDA y la optimización se desarrollaron sobre Train; Test se reservó para
la evaluación final.

## 4. Análisis exploratorio (EDA)

El EDA se realizó en `notebooks/01_EDA.ipynb` usando solo Train, para
comprender las variables, la distribución del objetivo y los problemas
de calidad.

### Hallazgos de calidad confirmados

-   `customerID` es un identificador y se excluyó de las variables
    predictoras.
-   `TotalCharges` estaba almacenada como texto y necesitaba conversión
    numérica.
-   Se detectaron 8 valores de `TotalCharges` en blanco en Train y 3 en
    Test; los registros correspondían a `tenure = 0` y `Churn = No`.
-   En la revisión de Train se reportaron 0 filas duplicadas.
-   La clase `Churn = No` representa 73,46% de Train y `Churn = Yes`,
    26,54%, lo que supone un desbalance moderado.

Debido al desbalance, Accuracy no es suficiente por sí sola. Precision,
Recall y F1-score permiten observar con mayor detalle el desempeño para
la clase positiva (`Yes`).

Las variables disponibles incluyen antigüedad (`tenure`), cargos
(`MonthlyCharges`, `TotalCharges`), `SeniorCitizen`, tipo de contrato,
método de pago y servicios contratados. El perfilado posterior mostró
diferencias descriptivas entre clusters en cargos, internet, contratos y
métodos de pago.

**Trazabilidad:** los hallazgos incluidos aquí son los confirmados en
las ejecuciones documentadas. Si el notebook EDA contiene otras cifras o
conclusiones concretas, deben añadirse con sus valores exactos y
gráficos correspondientes.

## 5. Preprocesamiento

El flujo se implementó en `src/data_processing.py` y mediante pipelines
de scikit-learn.

### Limpieza

-   Eliminar espacios al inicio/final de cadenas.
-   Convertir `TotalCharges` a tipo numérico.
-   Tratar los valores vacíos de `TotalCharges` como cero según la regla
    aplicada a registros con `tenure = 0`.
-   Separar `customerID` para trazabilidad y excluirlo de `X`.
-   Separar `Churn` como objetivo y mapear `No = 0`, `Yes = 1`.

### Transformaciones supervisadas

Se utilizaron 19 predictores originales: - Variables continuas `tenure`,
`MonthlyCharges` y `TotalCharges`: `RobustScaler`. - `SeniorCitizen`: se
mantuvo numérica. - Variables categóricas: `OneHotEncoder`. -
`customerID` y `Churn`: excluidas de los predictores.

El procesamiento produjo 45 variables transformadas. Se verificó que las
matrices procesadas no tuvieran valores faltantes y que Train y Test
tuvieran las mismas columnas transformadas.

En la optimización, el preprocesador se incluyó dentro de un `Pipeline`.
Así, cada partición de validación cruzada ajusta los transformadores
solo con su porción de entrenamiento, reduciendo el riesgo de fuga de
información.

## 6. Modelos supervisados y Optuna

### 6.1 Justificación de algoritmos

**Regresión Logística:** modelo de clasificación que estima
probabilidades y ofrece una referencia relativamente interpretable.
Permite evaluar si una frontera de decisión lineal es competitiva.

**Random Forest:** conjunto de árboles de decisión capaz de modelar
relaciones no lineales e interacciones entre variables. Se compara con
la Regresión Logística para evaluar si esa flexibilidad mejora el
desempeño.

### 6.2 Optimización

`src/model_optimization.py` entrenó ambos modelos utilizando Train. Se
ejecutaron 30 ensayos de Optuna por modelo y validación cruzada
estratificada de 5 particiones (`StratifiedKFold`, `random_state=42`).
La métrica de optimización fue F1-score, porque combina Precision y
Recall ante una clase minoritaria. Ambas métricas se revisaron por
separado en la evaluación final.

  Modelo                  F1 promedio CV   Desviación estándar
  --------------------- ---------------- ---------------------
  Random Forest                   0,6342                0,0187
  Regresión Logística             0,6298                0,0233

La diferencia de F1 promedio fue pequeña (\~0,0045), por lo que la
selección no se basó únicamente en validación cruzada.

**Mejores hiperparámetros de Regresión Logística** - `C = 6.3402` -
`class_weight = balanced`

**Mejores hiperparámetros de Random Forest** - `n_estimators = 100` -
`max_depth = None` - `min_samples_split = 18` -
`min_samples_leaf = 10` - `max_features = log2` -
`class_weight = balanced_subsample`

La ponderación de clases busca que la clase minoritaria tenga mayor
consideración durante el ajuste. Los pipelines completos se guardaron
como `models/logistic_regression_optuna.joblib` y
`models/random_forest_optuna.joblib`.

### 6.3 Evaluación final en Test

  Métrica       Regresión Logística   Random Forest
  ----------- --------------------- ---------------
  Accuracy                   0,7395          0,7651
  Precision                  0,5060          0,5394
  Recall                     0,7834          0,7861
  F1-score                   0,6149          0,6398
  ROC-AUC                    0,8432          0,8412

Se usó el umbral predeterminado de `predict`; las probabilidades se
utilizaron para ROC-AUC.

Random Forest tuvo mejor Accuracy, Precision, Recall y F1-score. Su
Recall de 78,61% indica que identificó cerca de cuatro quintos de los
clientes que realmente abandonaron en Test. Su Precision de 53,94%
significa que aproximadamente la mitad de los clientes marcados como
posibles desertores efectivamente abandonó. La Regresión Logística tuvo
un ROC-AUC apenas mayor (0,8432 frente a 0,8412), por lo que su
capacidad global de discriminación a través de umbrales fue ligeramente
superior, aunque la diferencia es pequeña.

### 6.4 Matrices de confusión

  Resultado                     Regresión Logística   Random Forest
  --------------------------- --------------------- ---------------
  Verdaderos negativos (TN)                     749             784
  Falsos positivos (FP)                         286             251
  Falsos negativos (FN)                          81              80
  Verdaderos positivos (TP)                     293             294

Random Forest detectó un cliente adicional que abandonó y redujo los
falsos positivos en 35 casos. Los falsos negativos son clientes que
abandonan pero no serían detectados por una campaña basada en el modelo;
los falsos positivos pueden ocasionar contactos innecesarios.

### 6.5 Curva ROC

Ambos modelos presentan ROC-AUC cercano a 0,84, por encima de 0,5, la
referencia de un clasificador sin capacidad de discriminación. La curva
ROC evalúa la relación entre la tasa de verdaderos positivos y la tasa
de falsos positivos para distintos umbrales. No determina por sí sola el
umbral óptimo: para producción se debe estimar el costo de errores y
campañas de retención.

## 7. Modelo no supervisado: K-Means

### 7.1 Preparación

K-Means se aplicó para descubrir segmentos sin utilizar `Churn` en la
formación de grupos. Se excluyeron `customerID` y `Churn`. Las variables
numéricas se escalaron con `RobustScaler` y las categóricas se
transformaron con `OneHotEncoder`.

### 7.2 Selección de k

Se evaluaron de 2 a 8 grupos mediante inercia y Silhouette.

    k     Inercia   Silhouette
  --- ----------- ------------
    2   41.922,85       0,2492
    3   35.733,02       0,2194
    4   33.349,64       0,1878
    5   31.883,48       0,1743
    6   30.842,91       0,1665
    7   29.951,98       0,1188
    8   28.982,63       0,1096

La inercia disminuye al aumentar k, como es esperable. El método del
codo no presentó un punto inequívoco, aunque la reducción se suavizó
tras los primeros grupos. El mejor Silhouette se obtuvo con `k=2`
(0,2492), por lo que se eligieron inicialmente dos clusters. El valor
indica que la separación es limitada y que existe cierto solapamiento.

### 7.3 Perfiles observados

  Variable                     Cluster 0   Cluster 1
  -------------------------- ----------- -----------
  `SeniorCitizen` (media)           0,03        0,20
  `tenure` (media)                 30,22       33,11
  `MonthlyCharges` (media)         21,11       76,96
  `TotalCharges` (media)          657,12    2.750,39

**Cluster 0:** el 100% no tiene servicio de internet. El 41,8% tiene
contrato de dos años y 23,5% contrato de un año. El método de pago más
común es cheque enviado por correo (49,0%). Los cargos mensuales
promedio son menores.

**Cluster 1:** todos tienen internet; 56,2% usa fibra óptica y 43,8%
DSL. El 60,6% tiene contrato mensual, 67,4% utiliza facturación
electrónica y 40,7% paga mediante cheque electrónico. Los cargos
mensuales y acumulados promedio son mayores.

Las distribuciones de género y `Partner` son parecidas entre los grupos.
Las diferencias más marcadas corresponden a internet, cargos, duración
del contrato y método de pago.

### 7.4 Abandono observado por cluster

Después de formar los grupos sin la variable objetivo, se comparó
descriptivamente `Churn`: - Cluster 0: tasa de abandono aproximada de
7%. - Cluster 1: tasa de abandono aproximada de 32%. - Train general:
26,54%.

El Cluster 1 podría ser prioritario para investigar acciones de
retención, ya que su tasa supera el promedio general. Esto no demuestra
causalidad ni convierte el cluster en un predictor individual.

Se guardaron cuatro CSV en `reports/clustering/`:
`train_customer_segments.csv`, `clustering_k_evaluation.csv`,
`clustering_numeric_profiles.csv` y `clustering_churn_distribution.csv`.
Los artefactos de modelo son
`models/kmeans_customer_segmentation.joblib` y
`models/clustering_preprocessor.joblib`.

## 8. Conclusiones técnicas

1.  La partición estratificada 80/20 permite evaluar los modelos con
    datos reservados, manteniendo Test fuera del desarrollo.
2.  La limpieza, el escalamiento y la codificación permiten utilizar las
    variables en los algoritmos.
3.  Optuna encontró configuraciones con desempeño CV parecido; Random
    Forest tuvo una ventaja pequeña en F1 promedio.
4.  En Test, Random Forest obtuvo mejores Accuracy, Precision, Recall y
    F1-score con el umbral predeterminado; Regresión Logística logró un
    ROC-AUC marginalmente superior.
5.  K-Means con dos clusters encontró perfiles descriptivamente
    diferentes, pero Silhouette de 0,2492 indica separación limitada.
6.  Los modelos supervisados estiman la clase de abandono; K-Means
    agrupa clientes por similitud. Cumplen objetivos distintos y sus
    resultados son complementarios.

## 9. Recomendaciones de negocio

-   Priorizar el análisis del Cluster 1 y estudiar motivos de abandono
    mediante datos adicionales, reclamos y satisfacción.
-   Evaluar umbrales de clasificación alternativos según costos de
    campaña y capacidad operativa.
-   Probar estrategias de retención mediante experimentos controlados y
    medir retención incremental, costo por cliente retenido e impacto
    económico.
-   Monitorear el desempeño y actualizar el modelo cuando cambien los
    patrones de clientes o servicios.
-   Utilizar las predicciones como apoyo a decisiones, no como
    decisiones automáticas incuestionables.

## 10. Ética, riesgos y limitaciones

-   **Privacidad:** manejar los datos conforme a políticas de privacidad
    y normas aplicables; el identificador se excluyó de los predictores.
-   **Sesgo y trato justo:** las diferencias entre grupos no deben
    producir automáticamente trato desfavorable. Revisar resultados por
    subgrupos y mantener supervisión humana.
-   **Explicabilidad:** las métricas globales no explican por qué se
    clasifica a un cliente específico; conviene incorporar herramientas
    explicativas antes del uso operativo.
-   **Calidad de datos:** reemplazar `TotalCharges` vacío por cero
    depende de que los registros correspondan a clientes con `tenure=0`;
    validar esta regla con el sistema de origen.
-   **Desbalance:** Accuracy puede ocultar errores en la clase
    minoritaria, por lo que se incluyeron Precision, Recall y F1.
-   **Umbral:** el umbral predeterminado no necesariamente maximiza el
    valor de negocio.
-   **Clustering:** Silhouette moderado-bajo implica separación
    limitada; los grupos son descriptivos, no causales.
-   **Generalización:** el rendimiento corresponde a esta partición y
    puede variar en otros periodos o poblaciones.

## 11. Reproducibilidad y estructura

``` text
telco-churn-ml/
├── data/
│   ├── raw/
│   ├── split/
│   └── processed/
├── notebooks/
│   ├── 00_separacion_train_test.ipynb
│   ├── 01_EDA.ipynb
│   ├── 02_Modelos_Supervisados.ipynb
│   └── 04_Modelo_No_Supervisado.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   └── model_optimization.py
├── models/
├── reports/
│   ├── optuna/
│   ├── clustering/
│   └── supervised_test_results.csv
├── images/
├── README.md
└── requirements.txt
```

### Artefactos principales

-   `data/split/telco_train.csv` y `data/split/telco_test.csv`:
    particiones de datos.
-   `data/processed/`: matrices y etiquetas procesadas.
-   `models/logistic_regression_optuna.joblib` y
    `models/random_forest_optuna.joblib`: pipelines supervisados
    completos.
-   `models/preprocessing_pipeline.joblib`: preprocesador independiente.
-   `models/kmeans_customer_segmentation.joblib` y
    `models/clustering_preprocessor.joblib`: modelo y transformaciones
    de clustering.
-   `reports/optuna/`: ensayos y mejores hiperparámetros.
-   `reports/supervised_test_results.csv`: métricas de Test.
-   `reports/clustering/`: perfiles y resultados de clustering.

### Orden de ejecución recomendado

1.  `00_separacion_train_test.ipynb`
2.  `01_EDA.ipynb`
3.  `src/data_processing.py`
4.  `src/model_optimization.py`
5.  `02_Modelos_Supervisados.ipynb`
6.  Notebook de clustering

Mantener las semillas aleatorias, los archivos de datos y las
dependencias de `requirements.txt` permite reproducir los resultados.
Ejecutar los notebooks desde la estructura del proyecto para que las
rutas relativas funcionen.

## 12. Referencias técnicas

-   Scikit-learn. Documentación de pipelines, preprocesamiento,
    métricas, K-Means y Silhouette. https://scikit-learn.org/stable/
-   Optuna. Documentación de optimización de hiperparámetros.
    https://optuna.org/
-   Harris, C. R., et al. (2020). *Array programming with NumPy*.
    Nature, 585, 357--362.
-   McKinney, W. (2010). *Data Structures for Statistical Computing in
    Python*. Proceedings of the 9th Python in Science Conference.

## Anexo: evidencias que conviene adjuntar

Para que el informe sea completamente trazable, insertar o referenciar
los gráficos reales de: 1. Distribución de `Churn` y gráficos
principales del EDA. 2. Calidad de datos y tratamiento de
`TotalCharges`. 3. Hiperparámetros de Optuna y F1 promedio CV. 4. Tabla
de métricas de Test. 5. Matrices de confusión y curva ROC. 6. Método del
codo y Silhouette. 7. Perfiles de los clusters y tasa de abandono por
cluster.

Las evidencias deben provenir de las ejecuciones reales de los
notebooks.
