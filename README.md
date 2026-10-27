# Telco Customer Churn — Machine Learning

Proyecto académico de Machine Learning para analizar y predecir el abandono de clientes (`Churn`) y explorar segmentos mediante K-Means y análisis de componentes principales (PCA).

> **Documentación técnica:** consulta [`Informe_Tecnico_Telco_Customer_Churn.md`](Informe_Tecnico_Telco_Customer_Churn.md) para la explicación completa de la metodología, las decisiones técnicas, los resultados y las limitaciones.

## Objetivos

- Analizar la calidad y las características de los datos.
- Preparar variables numéricas y categóricas para Machine Learning.
- Comparar Regresión Logística y Random Forest.
- Optimizar hiperparámetros con Optuna y validación cruzada.
- Evaluar el desempeño con métricas de clasificación.
- Explorar segmentos de clientes mediante K-Means y PCA.

## Tecnologías

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Optuna, joblib y Jupyter.

## Estructura del proyecto

```text
telco-churn-ml/
├── data/
│   ├── raw/                  # Dataset original
│   ├── split/                # Train y Test
│   └── processed/            # Matrices y etiquetas procesadas
├── notebooks/
│   ├── 00_separacion_train_test.ipynb
│   ├── 01_EDA.ipynb
│   ├── 02_Modelos_Supervisados.ipynb
│   └── 03_Modelo_No_Supervisado.ipynb
├── src/
│   ├── data_processing.py
│   └── model_optimization.py
├── models/                   # Pipelines y modelos guardados
├── reports/
│   ├── optuna/
│   ├── clustering/
│   └── supervised_test_results.csv
├── images/                   # Gráficos exportados, si corresponde
├── main.py                   # Orquestador del flujo reproducible
├── Informe_Tecnico_Telco_Customer_Churn.md
├── README.md
└── requirements.txt
```

Los nombres de archivos indicados deben coincidir con los de tu carpeta local. La estructura refleja los nombres de notebooks utilizados en la última ejecución correcta del orquestador.

## Requisitos

- Python compatible con las versiones declaradas en `requirements.txt`.
- Dataset original ubicado en `data/raw/Telco_Customer_Churn_Dataset.csv`.
- Los notebooks y scripts del proyecto en sus carpetas correspondientes.

## Instalación en Windows

Abre una terminal en la raíz del proyecto y crea un entorno virtual:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name=python3 --display-name "Python (telco-churn-ml)"
```

## Ejecución reproducible

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe main.py
```

`main.py` ejecuta en orden los notebooks y scripts configurados, deteniéndose si una etapa falla. La primera ejecución puede tardar varios minutos debido a los ensayos de Optuna.

Orden metodológico:

1. Separación de Train/Test.
2. EDA sobre Train.
3. Preprocesamiento y exportación de matrices.
4. Optimización de modelos supervisados con Optuna.
5. Evaluación de los modelos sobre Test.
6. Análisis no supervisado con selección de componentes PCA, K-Means original y comparación experimental con PCA.

**Importante:** Test debe mantenerse fuera del EDA, del ajuste del preprocesamiento y de la optimización. Solo debe utilizarse para la evaluación final. El preprocesamiento usado durante la validación cruzada debe estar dentro de los pipelines para reducir la fuga de información.

### Ejecutar una etapa por separado

Los notebooks pueden ejecutarse desde Jupyter, en el orden anterior. Los scripts se ejecutan desde la raíz:

```powershell
.\.venv\Scripts\python.exe src\data_processing.py
.\.venv\Scripts\python.exe src\model_optimization.py
```

Ejecuta los scripts solo cuando las particiones requeridas ya existan.

## Resultados principales registrados

La evaluación sobre Test reportó los siguientes resultados:

| Métrica | Regresión Logística | Random Forest |
|---|---:|---:|
| Accuracy | 0,7395 | 0,7651 |
| Precision | 0,5060 | 0,5394 |
| Recall | 0,7834 | 0,7861 |
| F1-score | 0,6149 | 0,6398 |
| ROC-AUC | 0,8412 | 0,8432 |

Random Forest obtuvo mejores Accuracy, Precision, Recall y F1-score con el umbral predeterminado; Regresión Logística obtuvo un ROC-AUC ligeramente inferior que random forest. La selección final debe considerar el costo de falsos positivos y falsos negativos, no una sola métrica.

En clustering, la solución original de K-Means seleccionó `k=2`, con Silhouette de 0,2492. El PCA seleccionó 16 componentes para conservar aproximadamente 92 % de la varianza de las 45 características procesadas. Las dos primeras componentes conservan aproximadamente 40,32 % y se utilizan para visualización 2D. La comparación experimental confirmó que **K-Means con PCA y `k=2` alcanzó Silhouette 0,2715**, frente a **0,2492** en el espacio original (mejora relativa aproximada de **8,96 %**). Ambos valores reflejan separación limitada. Los resultados para `k=2..8`, la interpretación y las limitaciones se documentan en el [informe técnico](Informe_Tecnico_Telco_Customer_Churn.md) y en `reports/clustering/clustering_pca_comparison.csv`.

Los resultados pueden variar si se cambian los datos, las dependencias, los parámetros o la semilla aleatoria.

## Artefactos generados

- `data/split/`: particiones Train/Test.
- `data/processed/`: matrices transformadas y etiquetas.
- `models/`: pipelines y modelos serializados con joblib.
- `reports/optuna/`: ensayos e hiperparámetros de Optuna.
- `reports/supervised_test_results.csv`: métricas de evaluación en Test.
- `reports/clustering/`: resultados, perfiles de segmentos, varianza PCA y comparación experimental.
- `models/pca_selected_components.joblib` y `models/pca_visualization_2d.joblib`: PCA para reducción y visualización.

## Buenas prácticas

- Ejecutar `main.py` desde la raíz del proyecto.
- Mantener las semillas aleatorias configuradas en los scripts.
- No editar manualmente los CSV generados sin documentarlo.
- No utilizar `Churn` como predictor en K-Means.
- No interpretar asociación como causalidad.
- No usar predicciones para decisiones automáticas sin validación de negocio, revisión de sesgos y supervisión humana.

## Estado de ejecución

La última ejecución compartida de `main.py` finalizó correctamente: separación, EDA, preprocesamiento, optimización Optuna, evaluación supervisada y clustering. Se eliminaron las advertencias anteriores sobre `ExecutePreprocessor.cwd`. Los avisos de ZeroMQ/TCP del kernel en Windows fueron no fatales. Esta comprobación es local; para asegurar portabilidad, ejecutar también desde un entorno nuevo con las dependencias instaladas.

## Documentación ampliada

**Para cualquier consulta metodológica o de resultados, consulta primero el [Informe técnico completo](Informe_Tecnico_Telco_Customer_Churn.md).** Allí se explican el EDA, las decisiones reales de preprocesamiento, la optimización, la evaluación Test, las matrices de confusión, PCA, la selección de `k`, los perfiles de clientes, las limitaciones y las recomendaciones de negocio.
