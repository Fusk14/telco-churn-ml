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
segmentación no supervisada con K-Means, selección de componentes principales mediante PCA y ejecución integral mediante `main.py`.

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
alternativa seleccionada inicialmente, con Silhouette 0,2492 en el espacio original preprocesado. PCA se incorporó para estudiar la varianza acumulada, proyectar visualmente los grupos y comparar una segmentación en un espacio reducido. La comparación confirmó que, con `k=2`, Silhouette aumentó de **0,2492** en el espacio original a **0,2715** con PCA (mejora relativa aproximada del **8,96 %**), aunque la separación continúa siendo limitada. Las tasas
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

### Asociaciones relevantes con Churn

En el EDA se exploraron las 21 columnas originales, distinguiendo el identificador, la variable objetivo, tres variables continuas (`tenure`, `MonthlyCharges`, `TotalCharges`), una variable binaria (`SeniorCitizen`) y quince variables categóricas de servicios, contrato y perfil. Los gráficos de tasas de abandono por categoría permiten comparar proporciones dentro de cada categoría, sin confundirlas con el tamaño absoluto del segmento.

Entre las asociaciones categóricas más relevantes reportadas por el análisis de **V de Cramér corregida** destacan `Contract` (aprox. 0,411), `OnlineSecurity` (0,351), `TechSupport` (0,344), `InternetService` (0,326) y `PaymentMethod` (0,310). Estas asociaciones describen intensidad, no dirección ni causalidad. Los tipos de contrato y los servicios de seguridad y soporte son, por tanto, variables prioritarias para explorar comercialmente.

Para las variables numéricas se eligió **Spearman** por la asimetría observada. Se incluyó una codificación auxiliar de `Churn` (`No=0`, `Yes=1`) exclusivamente para el análisis de asociación, sin modificar los datos originales. Los coeficientes reportados con `Churn` fueron aproximadamente `tenure=-0,360`, `MonthlyCharges=0,190` y `TotalCharges=-0,224`. La correlación `tenure`–`TotalCharges` fue aproximadamente `0,891`. Son relaciones descriptivas; una correlación no demuestra que una variable cause el abandono.

### Comparación numérica por Churn: boxplots y evidencia exploratoria

Como complemento a histogramas, tasas de abandono y correlaciones, se incorporó en `notebooks/01_EDA.ipynb` una figura con **tres boxplots** (`tenure`, `MonthlyCharges` y `TotalCharges`) separados por `Churn = No/Yes`. Los boxplots permiten contrastar medianas, dispersión y valores extremos; su elaboración utiliza únicamente una representación numérica auxiliar de `TotalCharges` y **no modifica ni imputa el conjunto original**.

El patrón destacado es la menor antigüedad de los clientes que abandonan: en el análisis descrito, la mediana de `tenure` es aproximadamente **10 meses** para `Churn = Yes` y **38 meses** para `Churn = No`. Los cargos mensuales también muestran diferencias descriptivas entre grupos. Estas observaciones justifican mantener dichas variables como *candidatas predictivas*, pero un boxplot **no prueba significancia estadística, causalidad ni contribución independiente al clasificador**. La relación `tenure`–`TotalCharges` también obliga a considerar información redundante.

### Decisiones que el EDA fundamenta, sin ejecutarlas

El EDA no imputa, escala ni codifica el conjunto de entrenamiento. Detecta y documenta los problemas y patrones que motivan las operaciones posteriores: espacios en textos, `TotalCharges` almacenado como texto, vacíos vinculados a `tenure=0`, distribuciones asimétricas, variables binarias y categóricas, y desbalance del objetivo. La ejecución real de esas operaciones corresponde a los scripts de preprocesamiento y optimización.

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

### Correspondencia exacta entre los dos preprocesadores

**`src/data_processing.py` (exportación de matrices):** limpia espacios, reemplaza `TotalCharges=""` por `"0"`, convierte a numérico, valida identificadores y objetivo, codifica `Churn` como 0/1 y separa `customerID`. En las tres variables continuas aplica imputación por mediana y `RobustScaler`; `SeniorCitizen` pasa sin escalar; las quince categóricas usan imputación por moda y `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. El preprocesador se ajusta con Train y transforma Train y Test, generando **45 columnas** sin nulos y conservando idéntico esquema.

**`src/model_optimization.py` (validación cruzada y modelos finales):** realiza una limpieza determinista propia de Train; convierte `TotalCharges` a numérico, comprueba que los vacíos correspondan a `tenure=0` y los fija a cero; verifica el objetivo y lo codifica. El preprocesador forma parte de cada pipeline de clasificación: imputación por mediana y `RobustScaler` para continuas, imputación por moda para `SeniorCitizen` (sin escalado) e imputación por moda seguida de `OneHotEncoder(handle_unknown="ignore")` para categóricas. Al integrarlo dentro de la validación cruzada, cada fold aprende las estadísticas exclusivamente de su parte de entrenamiento.

**Distinción importante:** ambos preprocesadores persiguen la misma representación de variables, pero no son literalmente idénticos: el tratamiento de `SeniorCitizen` es `passthrough` en el primero e imputación por moda en el segundo; además, difieren en las validaciones de `TotalCharges` y en la configuración de salida del codificador. Estas diferencias se documentan sin alterar modelos ya entrenados. En ambos casos, `customerID` y `Churn` quedan fuera de las variables predictoras.

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
  ROC-AUC                    0,8412          0,8432

Se usó el umbral predeterminado de `predict`; las probabilidades se
utilizaron para ROC-AUC.

Random Forest tuvo mejor Accuracy, Precision, Recall y F1-score. Su
Recall de 78,61% indica que identificó cerca de cuatro quintos de los
clientes que realmente abandonaron en Test. Su Precision de 53,94%
significa que aproximadamente la mitad de los clientes marcados como
posibles desertores efectivamente abandonó.

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

### 6.6 Interpretabilidad: importancia por permutación de las 19 variables originales

Se añadió al notebook `02_Modelos_Supervisados.ipynb` una comparación de **importancia por permutación** para los pipelines ya entrenados de Regresión Logística y Random Forest. En lugar de estimar relevancia por separado para cada columna resultante del One-Hot Encoding, se altera una **variable original completa** y se mide cuánto disminuye el **F1-score**. Esto permite interpretar los predictores en términos comerciales sin modificar el entrenamiento de Optuna ni los modelos serializados.

El gráfico generado sobre **Train**, con cinco repeticiones de permutación y semilla 42, mostró el siguiente orden de variables destacadas; los valores son aproximaciones visuales del gráfico, no mediciones independientes:

| Modelo | Variables de mayor importancia observada | Interpretación |
|---|---|---|
| Regresión Logística | `tenure` (~0,164), `InternetService` (~0,130), `MonthlyCharges` (~0,082), `Contract` (~0,036) | La permanencia, el servicio de internet y el cargo mensual condicionan notablemente el F1 del modelo sobre Train. |
| Random Forest | `Contract` (~0,080), `InternetService` (~0,053), `tenure` (~0,038), `TotalCharges` (~0,034) | El contrato es la característica con mayor descenso de F1 al permutarse, seguido del servicio de internet y variables de permanencia/facturación. |

**Conexión con el EDA.** La importancia de `Contract`, `InternetService` y `tenure` resulta coherente con sus asociaciones y distribuciones exploratorias. El orden distinto en cada algoritmo demuestra que **asociación bivariada e importancia dentro del modelo no son lo mismo**. Por ejemplo, `Contract` presenta una asociación fuerte en el EDA y destaca especialmente en Random Forest, pero no encabeza el resultado de Regresión Logística.

**Alcance metodológico.** Los valores se calcularon sobre los mismos registros Train usados para ajustar los pipelines. Por tanto, son una **interpretación descriptiva potencialmente optimista**, no validación independiente de generalización; la permutación de variables correlacionadas puede subestimar su contribución individual. La importancia por permutación **no entrega valores p, intervalos de confianza inferenciales ni prueba de efecto causal**. También es sensible a la métrica elegida (`F1`), a la configuración del modelo y a la distribución de los datos.

**Decisión adoptada:** se conservan las **19 variables originales** y los modelos optimizados. No se ha realizado un procedimiento de selección de características ni puede afirmarse que las 19 sean estadísticamente significativas. Una posible mejora futura sería comparar, con **selección realizada dentro de cada fold de validación cruzada en Train**, un modelo reducido frente al modelo completo. No corresponde utilizar Test para decidir qué variables conservar.

### 6.7 Tres conceptos que no deben confundirse

- **Asociación exploratoria:** describe patrones entre una característica y `Churn` (boxplots, Spearman, V de Cramér); puede reflejar correlaciones entre predictores.
- **Importancia predictiva:** mide la dependencia de un modelo concreto respecto de una característica para una métrica definida (aquí, caída de F1 al permutarla en Train).
- **Significancia estadística:** se refiere a una inferencia formal bajo hipótesis y supuestos establecidos; ni los boxplots ni este gráfico de permutación prueban por sí solos significancia.

Del mismo modo, ninguna de las tres demuestra que modificar una característica provoque una reducción del abandono. Para afirmar causalidad harían falta diseños y datos adicionales, como experimentos controlados sobre intervenciones de retención.

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

### 7.5 PCA: selección justificada de componentes principales

El PCA se incorporó como análisis complementario al clustering. Partiendo de las **45 características preprocesadas**, se calculó la varianza explicada acumulada y se utilizó un umbral de referencia del **90 %**, buscando reducir dimensionalidad sin perder una proporción elevada de la variabilidad de los datos. Según los resultados registrados del notebook actualizado, se requieren **16 componentes** para conservar aproximadamente **92,00 %** de la varianza; **11** alcanzan el 80 % y **18** el 95 %.

Se guardan dos transformaciones PCA con propósitos distintos:

- **PCA de 16 componentes:** representación reducida para comparar experimentalmente K-Means con el clustering original de 45 características.
- **PCA de 2 componentes:** proyección bidimensional para dibujar los grupos. Sus dos ejes representan aproximadamente **40,32 %** de la varianza, por lo que el solapamiento visual no determina por sí solo la calidad de los clusters.

La comparación se realizó sobre los valores `k=2` a `k=8` en ambos espacios. Los resultados de `reports/clustering/clustering_pca_comparison.csv` son:

| k | Silhouette original (45 características) | Silhouette PCA (90 % de varianza) |
|---:|---:|---:|
| 2 | 0,2492 | **0,2715** |
| 3 | 0,2194 | 0,2388 |
| 4 | 0,1878 | 0,2046 |
| 5 | 0,1743 | 0,1862 |
| 6 | 0,1665 | 0,1800 |
| 7 | 0,1188 | 0,1182 |
| 8 | 0,1096 | 0,1245 |

**Interpretación:** en ambas representaciones, el mejor Silhouette entre los valores probados corresponde a **dos clusters**. El PCA de 16 componentes elevó Silhouette de **0,24918596** a **0,27151810**, es decir, **+0,02233 puntos** (aprox. **+8,96 %** relativo). Esto es una mejora descriptiva de la cohesión y separación en el espacio PCA, no una prueba de significación estadística ni de mayor utilidad comercial. Los dos valores siguen siendo modestos: la estructura de clusters presenta solapamiento. La inercia original para `k=2` fue **41.922,85**, frente a **37.582,31** en PCA; **no deben compararse directamente como evidencia de mejora**, porque están calculadas en espacios de distinta dimensionalidad y varianza total.

El K-Means original sigue siendo la referencia de perfiles de negocio descrita en las secciones 7.3 y 7.4. No se atribuyen automáticamente sus perfiles ni sus tasas de Churn a los clusters entrenados con PCA: para hacerlo sería necesario examinar las etiquetas y perfiles específicos de esa alternativa.

El PCA no utiliza `Churn` para construir componentes ni para formar clusters. `Churn` se utiliza después, solo para interpretar las tasas observadas por segmento.

**Archivos adicionales de la etapa PCA:**

- `models/pca_selected_components.joblib`: PCA con la cantidad de componentes seleccionada.
- `models/pca_visualization_2d.joblib`: proyección de visualización.
- `reports/clustering/pca_explained_variance.csv`: varianza explicada y acumulada.
- `reports/clustering/pca_customer_projections.csv`: coordenadas PCA.
- `reports/clustering/clustering_pca_comparison.csv`: comparación experimental del clustering.

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
5.  K-Means con dos clusters encontró perfiles descriptivamente diferentes. Con 16 componentes PCA (aprox. 92 % de varianza), el Silhouette para `k=2` aumentó de 0,2492 a 0,2715 (+8,96 % relativo), aunque ambos valores reflejan separación limitada. La visualización con dos componentes no equivale al entrenamiento con 16.
6.  La importancia por permutación confirma que el orden de contribución de variables depende del algoritmo: destacan `tenure` e `InternetService` en Regresión Logística y `Contract` e `InternetService` en Random Forest. Los hallazgos son descriptivos sobre Train y no justifican eliminar variables sin validación cruzada.
7.  Los modelos supervisados estiman la clase de abandono; K-Means
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
-   **Explicabilidad:** se incorporó importancia global por permutación en Train, pero no es explicación causal ni individual y puede ser optimista. Para uso operativo todavía serían necesarias explicaciones locales, verificación de estabilidad y revisión humana.
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
│   └── 03_Modelo_No_Supervisado.ipynb
├── src/
│   ├── data_processing.py
│   └── model_optimization.py
├── models/
├── reports/
│   ├── optuna/
│   ├── clustering/
│   └── supervised_test_results.csv
├── images/
├── main.py
├── Informe_Tecnico_Telco_Customer_Churn.md
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

### Orquestación verificada y orden de ejecución

1.  `00_separacion_train_test.ipynb`
2.  `01_EDA.ipynb`
3.  `src/data_processing.py`
4.  `src/model_optimization.py`
5.  `02_Modelos_Supervisados.ipynb`
6.  `03_Modelo_No_Supervisado.ipynb` (PCA y K-Means)

Desde la raíz del repositorio en Windows, ejecutar:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe main.py
```

`main.py` ejecuta las seis etapas en secuencia y se detiene ante errores. La **última ejecución compartida terminó con «EJECUCIÓN COMPLETADA»**, produjo los artefactos esperados y confirmó 5.634 registros Train, 1.409 Test, 45 columnas procesadas, ausencia de nulos en matrices y consistencia de columnas. El aviso anterior de `ExecutePreprocessor.cwd` desapareció. Permanecieron avisos no fatales de ZeroMQ y comunicación TCP del kernel en Windows. La ejecución local exitosa no sustituye una prueba en un entorno recién creado.

El código actualmente invoca el kernel `python3` para ejecutar notebooks; ese kernel debe estar instalado y apuntar a un entorno con las dependencias. Para una reproducción en otro equipo, revisar también la configuración de kernel, las rutas relativas de los notebooks y la presencia del dataset original. Mantener las semillas aleatorias, los datos y las versiones de `requirements.txt` facilita repetir el experimento; la igualdad bit a bit no está garantizada en todas las plataformas.

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
cluster. 8. Curva de varianza acumulada PCA, proyección bidimensional y comparación K-Means original frente a K-Means con PCA. 9. Boxplots comparativos por Churn e importancia por permutación de ambos modelos.

Las evidencias deben provenir de las ejecuciones reales de los
notebooks.


## 13. Alcance de la verificación, preparación de la exposición y próximos pasos

Los resultados supervisados en Test y los perfiles del K-Means original provienen del informe anterior; la ejecución más reciente del orquestador confirmó la finalización de sus etapas. Además, el CSV de comparación PCA fue aportado y permite verificar los Silhouette para `k=2..8` documentados en la sección 7.5. La consola no imprime las métricas finales de Test; para contrastarlas con la ejecución actual, consultar `reports/supervised_test_results.csv`. Para validar los perfiles de los clusters originales y sus tasas de Churn, consultar `reports/clustering/train_customer_segments.csv` y `reports/clustering/clustering_churn_distribution.csv`.

Antes de uso productivo sería necesario validar el desempeño fuera de esta partición, la estabilidad temporal de los clusters, la rentabilidad de campañas de retención, la equidad entre grupos y el manejo seguro de los datos. El proyecto es un análisis académico reproducible, no un sistema de decisión desplegado.


### Puntos clave para la defensa académica

1. **Variables de entrada:** 21 columnas originales; 19 predictores, un identificador (`customerID`) y un objetivo (`Churn`). El codificador genera 45 columnas procesadas en el preprocesador de exportación; no se realizó selección automática de variables.
2. **EDA → modelado:** las distribuciones y asociaciones justifican estudiar variables como antigüedad, contrato y servicio de internet. Se conservan todas inicialmente para evaluar modelos con la información disponible.
3. **Evidencia predictiva:** la importancia por permutación diferencia las dependencias de Regresión Logística y Random Forest, sin atribuir causalidad ni significancia formal.
4. **Comparación de modelos:** Random Forest obtuvo F1 de **0,6398 en Test** y Regresión Logística ROC-AUC de **0,8432**. Los F1 de la optimización (**0,6342** y **0,6298**) son promedios de **validación cruzada en Train** y no deben confundirse con Test.
5. **PCA y segmentación:** se seleccionaron **16 componentes** por aproximadamente **92 % de varianza acumulada**; dos componentes sirven solo para visualización 2D. Silhouette de K-Means aumentó de **0,2492** a **0,2715** con PCA a `k=2`, aunque la separación continúa limitada.
6. **Reproducibilidad y prudencia:** `main.py` finalizó todas las etapas en la última ejecución compartida. Los resultados no constituyen una evaluación de impacto causal ni garantizan generalización fuera de la población analizada.

**Control documental previo a presentar:** guardar las celdas nuevas del EDA y del notebook supervisado con sus gráficos y texto; revisar que el README remita a este informe; confirmar que los valores y títulos de las gráficas coinciden con las tablas y que la última ejecución no dejó celdas con error. No es necesario volver a ejecutar Optuna ni modificar los modelos para exponer los nuevos análisis descriptivos.
