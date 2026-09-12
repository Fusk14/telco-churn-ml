# Telco Customer Churn - Análisis Exploratorio y Preparación de Datos

## 1. Descripción del proyecto

Este proyecto corresponde a la **Evaluación Parcial N°1 de la asignatura MLY1101 - Machine Learning**.

El objetivo es analizar un conjunto de datos de clientes de una empresa de telecomunicaciones para identificar **patrones relacionados con el abandono de clientes (Churn)**.

Durante esta primera etapa se realiza un proceso completo de comprensión del problema, exploración, evaluación de calidad, limpieza y preparación de los datos, siguiendo la metodología **CRISP-DM**.

El proyecto se centra principalmente en:

* Comprender el problema de negocio.
* Analizar la estructura y calidad de los datos.
* Detectar valores faltantes, vacíos, duplicados e inconsistencias.
* Realizar limpieza y transformación de variables.
* Explorar la distribución de la variable objetivo `Churn`.
* Analizar patrones relacionados con contrato, antigüedad, cargos y servicios.
* Analizar relaciones entre variables numéricas mediante **correlación de Spearman**.
* Analizar asociaciones entre variables categóricas mediante **V de Cramér**.
* Identificar posibles sesgos y aspectos éticos.
* Considerar aspectos relacionados con privacidad y uso responsable de los datos.
* Dejar los datos preparados para una futura etapa de modelamiento predictivo.

---

## 2. Problema de negocio

Las empresas de telecomunicaciones enfrentan el problema de la **pérdida de clientes**, también conocido como *Customer Churn*.

Cuando un cliente abandona la compañía, la empresa no solamente pierde los ingresos asociados a ese cliente, sino que además debe invertir recursos para captar nuevos usuarios.

Por esta razón, resulta importante identificar patrones que permitan reconocer qué características están asociadas con una mayor probabilidad de abandono.

### Problema planteado

> ¿Qué características de los clientes están asociadas con el abandono del servicio y cómo pueden utilizarse estos patrones para apoyar futuras estrategias de retención?

El análisis desarrollado en este proyecto busca responder esta pregunta mediante técnicas de exploración y análisis estadístico, preparando posteriormente los datos para construir un modelo de Machine Learning capaz de predecir el abandono.

---

## 3. Objetivos

### 3.1 Objetivo general

Analizar y preparar un conjunto de datos de clientes de telecomunicaciones para identificar patrones asociados al abandono de clientes y establecer una base para el desarrollo posterior de un modelo predictivo de Churn.

### 3.2 Objetivos específicos

* Comprender el problema de negocio relacionado con la pérdida de clientes.
* Analizar la estructura del dataset y sus principales variables.
* Evaluar la calidad de los datos.
* Identificar valores faltantes, vacíos, duplicados e inconsistencias.
* Corregir problemas de tipos de datos.
* Explorar la distribución de la variable objetivo `Churn`.
* Analizar el comportamiento del abandono según diferentes características de los clientes.
* Analizar la relación entre variables numéricas mediante correlación de Spearman.
* Analizar la asociación entre variables categóricas mediante V de Cramér.
* Identificar variables potencialmente relevantes para un futuro modelo predictivo.
* Identificar posibles problemas de sesgo, ética y privacidad.
* Dejar los datos preparados para la etapa de modelamiento.

---

## 4. Preguntas de análisis

Durante el análisis exploratorio se plantearon las siguientes preguntas:

### Pregunta 1

**¿Cuál es la proporción de clientes que abandonan el servicio?**

Permite conocer la distribución de la variable objetivo `Churn` y detectar un posible desbalance entre las clases.

### Pregunta 2

**¿Existe una relación entre el tipo de contrato y el abandono?**

Permite analizar si los diferentes tipos de contrato presentan distintas tasas de Churn.

### Pregunta 3

**¿La antigüedad del cliente está relacionada con el abandono?**

Se analiza el comportamiento de Churn según diferentes rangos de `tenure`.

### Pregunta 4

**¿Los cargos mensuales presentan diferencias entre clientes que permanecen y clientes que abandonan?**

Se analiza el comportamiento de `MonthlyCharges` y su posible asociación con Churn.

### Pregunta 5

**¿Qué variables numéricas presentan mayor asociación con Churn?**

Para responder esta pregunta se utiliza la correlación de **Spearman**.

### Pregunta 6

**¿Qué variables categóricas presentan mayor asociación con Churn?**

Para responder esta pregunta se utiliza el **V de Cramér**.

---

# 5. Metodología CRISP-DM

El proyecto se estructura utilizando la metodología **CRISP-DM (Cross Industry Standard Process for Data Mining)**.

Las etapas consideradas son:

```text
Comprensión del negocio
        ↓
Comprensión de los datos
        ↓
Preparación de los datos
        ↓
Modelamiento
        ↓
Evaluación
        ↓
Despliegue
```

En esta evaluación se profundiza principalmente en las primeras etapas del proceso.

### 5.1 Comprensión del negocio

Se define como problema principal el abandono de clientes de una empresa de telecomunicaciones.

El objetivo es encontrar patrones que permitan posteriormente anticipar qué clientes presentan mayor riesgo de abandono.

### 5.2 Comprensión de los datos

Se analiza el dataset para conocer:

* Cantidad de registros.
* Cantidad de variables.
* Tipos de datos.
* Variables numéricas y categóricas.
* Distribución de la variable objetivo.
* Valores únicos.
* Posibles inconsistencias.

### 5.3 Preparación de los datos

Se realizan tareas como:

* Limpieza de espacios.
* Conversión de tipos de datos.
* Tratamiento de valores no válidos.
* Eliminación de registros que no pueden utilizarse correctamente.
* Revisión de duplicados.
* Preparación de variables para el análisis.

### 5.4 Modelamiento

Esta etapa queda preparada para una siguiente fase del proyecto.

Se contempla posteriormente:

* Codificación de variables categóricas.
* División entre entrenamiento y prueba.
* Selección de algoritmos.
* Entrenamiento de modelos.
* Ajuste de hiperparámetros.

### 5.5 Evaluación

En una futura etapa se evaluará el rendimiento del modelo utilizando métricas como:

* Accuracy.
* Precision.
* Recall.
* F1-Score.
* Matriz de confusión.
* ROC-AUC.

Debido al posible desbalance de la variable `Churn`, no se utilizará únicamente Accuracy para determinar el rendimiento del modelo.

### 5.6 Despliegue

Como etapa futura, el modelo podría utilizarse para identificar clientes con una mayor probabilidad de abandonar el servicio.

Esto permitiría apoyar estrategias de retención y priorizar acciones comerciales.

---

# 6. Dataset

El dataset utilizado corresponde a **Telco Customer Churn**, basado en datos de clientes de una empresa de telecomunicaciones.

### Características principales

| Característica      |                 Valor |
| ------------------- | --------------------: |
| Registros iniciales |                 7.043 |
| Variables           |                    21 |
| Variable objetivo   |               `Churn` |
| Tipo de problema    | Clasificación binaria |
| Clases objetivo     |          `Yes` / `No` |

Cada registro representa a un cliente.

La variable objetivo es:

```text
Churn
```

Sus valores representan:

* `Yes`: el cliente abandonó el servicio.
* `No`: el cliente permaneció en la compañía.

---

# 7. Variables del dataset

| Variable           | Descripción                               |
| ------------------ | ----------------------------------------- |
| `customerID`       | Identificador único del cliente           |
| `gender`           | Género del cliente                        |
| `SeniorCitizen`    | Indicador de cliente adulto mayor         |
| `Partner`          | Si el cliente tiene pareja                |
| `Dependents`       | Si el cliente tiene personas dependientes |
| `tenure`           | Antigüedad del cliente en meses           |
| `PhoneService`     | Si posee servicio telefónico              |
| `MultipleLines`    | Si posee múltiples líneas                 |
| `InternetService`  | Tipo de servicio de Internet              |
| `OnlineSecurity`   | Servicio de seguridad en línea            |
| `OnlineBackup`     | Servicio de respaldo en línea             |
| `DeviceProtection` | Protección del dispositivo                |
| `TechSupport`      | Soporte técnico                           |
| `StreamingTV`      | Servicio de televisión por streaming      |
| `StreamingMovies`  | Servicio de películas por streaming       |
| `Contract`         | Tipo de contrato                          |
| `PaperlessBilling` | Facturación electrónica                   |
| `PaymentMethod`    | Método de pago                            |
| `MonthlyCharges`   | Cargo mensual                             |
| `TotalCharges`     | Cargo total acumulado                     |
| `Churn`            | Indicador de abandono                     |

---

# 8. Fuente de datos

El dataset utilizado fue obtenido desde Kaggle:

**Telco Customer Churn - IBM Sample Dataset**

Fuente:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

El dataset es utilizado con fines académicos para realizar análisis exploratorio y desarrollar conocimientos relacionados con Machine Learning.

---

# 9. Herramientas utilizadas

El proyecto fue desarrollado principalmente utilizando:

| Herramienta      | Uso                                    |
| ---------------- | -------------------------------------- |
| Python           | Lenguaje principal                     |
| Google Colab     | Ejecución del notebook                 |
| Pandas           | Manipulación y análisis de datos       |
| NumPy            | Operaciones numéricas                  |
| Plotly           | Visualización interactiva              |
| SciPy            | Análisis estadístico                   |
| Scikit-learn     | Preparación y futuro modelamiento      |
| GitHub           | Control y publicación del proyecto     |
| Jupyter Notebook | Documentación y ejecución del análisis |

---

# 10. Calidad y preparación de los datos

Antes de realizar el análisis exploratorio se efectuó una revisión de la calidad de los datos.

Se revisaron los siguientes aspectos:

* Dimensiones del dataset.
* Tipos de datos.
* Valores faltantes.
* Valores vacíos o espacios.
* Duplicados.
* Identificadores duplicados.
* Valores únicos.
* Consistencia de las variables categóricas.
* Variables numéricas almacenadas incorrectamente.

## 10.1 Estructura del dataset

El dataset contiene inicialmente:

```text
7.043 registros
21 variables
```

La variable objetivo corresponde a `Churn`.

---

## 10.2 Valores faltantes

Se revisaron los valores faltantes mediante `isnull()`.

Además, se revisaron valores vacíos o compuestos únicamente por espacios, ya que estos pueden no ser detectados como valores nulos tradicionales.

---

## 10.3 Duplicados

Se verificó la existencia de registros completamente duplicados mediante:

```python
df.duplicated().sum()
```

También se revisó la unicidad de `customerID`, debido a que corresponde al identificador de cada cliente.

---

## 10.4 Problema detectado en `TotalCharges`

Una de las principales inconsistencias encontradas corresponde a la variable:

```text
TotalCharges
```

Esta variable representa un valor monetario, por lo que debería ser numérica. Sin embargo, originalmente se encontraba almacenada como tipo `object`.

Por esta razón se realizó:

1. Eliminación de espacios innecesarios.
2. Conversión a tipo numérico.
3. Identificación de valores que no podían convertirse.
4. Tratamiento de los registros afectados.

Ejemplo:

```python
df_clean["TotalCharges"] = (
    df_clean["TotalCharges"]
    .astype(str)
    .str.strip()
)

df_clean["TotalCharges"] = pd.to_numeric(
    df_clean["TotalCharges"],
    errors="coerce"
)
```

Posteriormente se revisaron los registros que quedaron como valores nulos producto de la conversión.

---

## 10.5 Duplicados y registros finales

Después de la limpieza se volvió a verificar:

* Valores faltantes.
* Duplicados.
* Dimensiones finales.
* Tipos de datos.

De esta forma se aseguró que los datos utilizados en el análisis fueran consistentes y adecuados para las siguientes etapas.

---

# 11. Análisis Exploratorio de Datos (EDA)

El análisis exploratorio busca comprender el comportamiento de los datos y encontrar patrones que puedan ser relevantes para explicar el abandono de clientes.

Las visualizaciones fueron realizadas utilizando **Plotly**, permitiendo generar gráficos interactivos.

---

## 11.1 Distribución de Churn

Se analizó la distribución de:

```text
Churn = Yes
Churn = No
```

El análisis permite observar que existe una diferencia importante entre ambas categorías.

La mayoría de los clientes permanece en la compañía, mientras que una proporción menor abandona.

Esto representa un posible **desbalance de clases**, aspecto que deberá considerarse durante el futuro modelamiento.

Por este motivo, para evaluar un modelo no será suficiente utilizar únicamente Accuracy.

Se deberán considerar métricas como:

* Precision.
* Recall.
* F1-Score.
* Matriz de confusión.
* ROC-AUC.

---

# 12. Análisis de contrato

Se analizó la tasa de abandono según:

```text
Month-to-month
One year
Two year
```

El análisis muestra diferencias importantes entre los tipos de contrato.

Los clientes con contratos de menor compromiso presentan una mayor tasa de abandono, mientras que los contratos de mayor duración presentan una tasa de Churn menor.

### Interpretación

El tipo de contrato aparece como una variable potencialmente relevante para el análisis del abandono.

Sin embargo, esta relación debe interpretarse como una **asociación** y no como una relación causal.

---

# 13. Análisis de antigüedad

La variable:

```text
tenure
```

representa la cantidad de meses que el cliente lleva utilizando el servicio.

Para facilitar el análisis, se crearon grupos de antigüedad:

```text
0-12 meses
13-24 meses
25-48 meses
49-72 meses
```

El análisis permite observar una tendencia en la que los clientes con menor antigüedad presentan una mayor tasa de abandono.

### Interpretación

Los primeros meses de relación con el cliente pueden representar un período relevante para estrategias de retención.

Esto convierte a `tenure` en una variable potencialmente importante para una futura predicción de Churn.

---

# 14. Análisis de cargos mensuales

Se analizó la variable:

```text
MonthlyCharges
```

comparando sus valores entre clientes que abandonaron y clientes que permanecieron.

Se utilizaron medidas estadísticas como:

* Cantidad de registros.
* Media.
* Mediana.
* Mínimo.
* Máximo.

También se dividieron los cargos mensuales en grupos para analizar si existen diferencias en la tasa de abandono.

Este análisis permite estudiar si los niveles de cargos mensuales presentan algún patrón asociado con `Churn`.

---

# 15. Análisis de servicios contratados

También se analizaron diferentes servicios de los clientes, entre ellos:

* Internet.
* Seguridad en línea.
* Respaldo en línea.
* Protección de dispositivos.
* Soporte técnico.
* Streaming TV.
* Streaming Movies.
* Servicio telefónico.
* Múltiples líneas.

El objetivo es identificar si determinados servicios presentan diferentes tasas de abandono.

Estos resultados permiten generar hipótesis que pueden ser evaluadas posteriormente mediante modelos predictivos.

---

# 16. Correlación de variables numéricas

Para analizar la relación entre las variables numéricas se utilizó la **correlación de Spearman**.

Las variables consideradas fueron:

```text
tenure
MonthlyCharges
TotalCharges
Churn
```

Para poder incluir `Churn` en el análisis se realizó una codificación:

```text
No  → 0
Yes → 1
```

### ¿Por qué Spearman?

Se utilizó Spearman debido a que permite identificar **relaciones monotónicas** entre variables sin asumir que la relación sea estrictamente lineal.

Además, al trabajar con rangos presenta menor sensibilidad a valores extremos que Pearson.

Esto resulta útil para explorar relaciones entre variables numéricas antes del modelamiento.

### Interpretación

El coeficiente de correlación toma valores entre:

```text
-1 y +1
```

Donde:

* `+1`: asociación positiva fuerte.
* `0`: ausencia de asociación monotónica.
* `-1`: asociación negativa fuerte.

Es importante destacar que:

> **Correlación no implica causalidad.**

Una asociación entre una variable y `Churn` no significa necesariamente que esa variable sea la causa del abandono.

---

# 17. Asociación entre variables categóricas

Debido a que una gran cantidad de variables del dataset son categóricas, no resulta apropiado aplicar directamente la correlación de Spearman a todas ellas.

Por esta razón se utilizó el **V de Cramér**.

## 17.1 ¿Qué es V de Cramér?

El V de Cramér es una medida estadística utilizada para evaluar la fuerza de asociación entre dos variables categóricas.

Su valor se encuentra entre:

```text
0 y 1
```

De forma general:

* `0`: poca o ninguna asociación.
* Valores mayores: mayor asociación.
* `1`: asociación muy fuerte.

La medida se basa en tablas de contingencia y utiliza el estadístico Chi-cuadrado.

---

## 17.2 Matriz de asociación

Se construyó una matriz de V de Cramér utilizando las variables categóricas del dataset.

El objetivo principal es analizar la fila o columna correspondiente a:

```text
Churn
```

De esta manera se pueden identificar las variables categóricas que presentan una mayor asociación con el abandono.

Al igual que en el análisis de correlación:

> **Una asociación estadística no implica causalidad.**

Los resultados deben utilizarse como una herramienta exploratoria para identificar variables potencialmente relevantes.

---

# 18. Principales hallazgos del EDA

A partir del análisis exploratorio realizado, se identificaron los siguientes patrones:

### 1. Desbalance de la variable objetivo

La cantidad de clientes que permanece en la compañía es superior a la cantidad de clientes que abandona.

Esto debe ser considerado durante la evaluación de futuros modelos.

### 2. Tipo de contrato

Se observan diferencias en la tasa de abandono según el tipo de contrato.

Los contratos de menor duración presentan mayores tasas de Churn.

### 3. Antigüedad

Los clientes con menor antigüedad presentan una mayor tasa de abandono.

Esto indica que los primeros meses pueden ser importantes para estrategias de retención.

### 4. Variables numéricas

Las variables `tenure`, `MonthlyCharges` y `TotalCharges` presentan diferentes niveles de asociación con `Churn`, los cuales fueron explorados mediante Spearman.

### 5. Variables categóricas

El análisis mediante V de Cramér permite identificar qué variables categóricas presentan una mayor asociación con `Churn`.

### 6. Importancia para el modelamiento

Los patrones encontrados permiten seleccionar variables potencialmente relevantes para una futura etapa de Machine Learning.

---

# 19. Sesgos y consideraciones éticas

El desarrollo de un modelo de predicción de Churn puede presentar diferentes riesgos relacionados con sesgo y toma de decisiones.

## 19.1 Desbalance de clases

Debido a que la cantidad de clientes que no abandona es superior a la cantidad de clientes que abandona, un modelo podría favorecer la clase mayoritaria.

Por esta razón se deben utilizar métricas adecuadas y técnicas de evaluación que permitan analizar correctamente ambas clases.

---

## 19.2 Variables demográficas

El dataset contiene variables relacionadas con características demográficas, como:

* `gender`
* `SeniorCitizen`
* `Partner`
* `Dependents`

Estas variables podrían generar comportamientos diferenciados en el modelo.

Por lo tanto, su utilización debe ser evaluada cuidadosamente para evitar decisiones discriminatorias.

---

## 19.3 Uso responsable del modelo

Un modelo predictivo debería utilizarse como una herramienta de apoyo a la toma de decisiones.

No debería utilizarse para tomar decisiones automáticas que puedan perjudicar a un cliente sin supervisión humana.

Por ejemplo, si el modelo identifica un cliente con alta probabilidad de abandono, la empresa podría utilizar esta información para ofrecer una estrategia de retención, pero no debería utilizarla automáticamente para negar servicios o aplicar condiciones desfavorables.

---

# 20. Privacidad y protección de datos

El dataset contiene un identificador:

```text
customerID
```

Este identificador permite distinguir individualmente a los clientes.

Para el análisis exploratorio no resulta necesario utilizarlo como variable predictora.

Por esta razón, `customerID` debe considerarse principalmente como un identificador y no como una característica del cliente.

En un escenario real sería necesario:

* Proteger los datos personales.
* Evitar exponer información identificable.
* Limitar el acceso a los datos.
* Utilizar solamente las variables necesarias.
* Aplicar medidas de anonimización o seudonimización cuando corresponda.
* Cumplir con la legislación aplicable sobre protección de datos personales.

---

# 21. Preparación para el modelamiento

Después del análisis exploratorio, los datos quedan preparados para continuar con una futura etapa de Machine Learning.

Las siguientes tareas serán necesarias:

### 21.1 Codificación de variables categóricas

Las variables categóricas deberán transformarse a una representación numérica utilizando técnicas como:

* One-Hot Encoding.
* Label Encoding, cuando corresponda.

### 21.2 Separación de datos

Se deberá dividir el dataset en:

```text
Training set
Test set
```

Por ejemplo:

```text
80% entrenamiento
20% prueba
```

La división deberá realizarse considerando el desbalance de `Churn`.

### 21.3 Selección de modelos

Como problema de clasificación binaria, podrían evaluarse modelos como:

* Regresión Logística.
* Árbol de Decisión.
* Random Forest.
* K-Nearest Neighbors.
* Otros algoritmos de clasificación.

### 21.4 Evaluación

Se deberán comparar los modelos utilizando métricas adecuadas:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
```

Debido al desbalance de clases, **Recall y F1-Score** tendrán especial importancia para evaluar la capacidad del modelo de identificar clientes que realmente abandonan.

---

# 22. KPIs propuestos

Para evaluar el problema de negocio se consideran los siguientes indicadores:

| KPI                | Descripción                                                             |
| ------------------ | ----------------------------------------------------------------------- |
| Tasa de Churn      | Porcentaje de clientes que abandonan                                    |
| Tasa de retención  | Porcentaje de clientes que permanecen                                   |
| Clientes en riesgo | Cantidad de clientes identificados con alta probabilidad de abandono    |
| Recall de Churn    | Capacidad del modelo para detectar clientes que efectivamente abandonan |
| F1-Score           | Equilibrio entre Precision y Recall                                     |
| ROC-AUC            | Capacidad general del modelo para diferenciar clases                    |

Los KPIs de Machine Learning deberán complementarse posteriormente con indicadores de negocio.

---

# 23. Estructura del proyecto

```text
telco-churn-ml/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── 01_telco_churn_eda.ipynb
│
├── images/
│   └── [gráficos utilizados en la documentación]
│
├── models/
│   └── [modelos generados en futuras etapas]
│
├── README.md
│
└── requirements.txt
```

---

# 24. Ejecución del proyecto

## Opción 1: Google Colab

El notebook puede ejecutarse directamente utilizando Google Colab.

Se recomienda cargar el dataset desde la carpeta `data/` o desde Google Drive.

## Opción 2: Ejecución local

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Luego ejecutar:

```bash
jupyter notebook
```

y abrir:

```text
notebooks/01_telco_churn_eda.ipynb
```

---

# 25. Dependencias

El archivo `requirements.txt` considera las principales librerías utilizadas:

```text
pandas
numpy
plotly
scipy
scikit-learn
jupyter
```

---

# 26. Resultados esperados

Al finalizar esta etapa se espera contar con:

* Dataset revisado y limpiado.
* Variables correctamente tipadas.
* Problemas de calidad identificados y tratados.
* Análisis exploratorio documentado.
* Visualizaciones interactivas.
* Análisis de variables numéricas mediante Spearman.
* Análisis de variables categóricas mediante V de Cramér.
* Identificación de patrones asociados con Churn.
* Consideraciones sobre sesgo, ética y privacidad.
* Datos preparados para el futuro desarrollo de modelos predictivos.

---

# 27. Conclusiones

El análisis realizado permitió comprender el problema de abandono de clientes y explorar los principales factores relacionados con `Churn`.

En primer lugar, se realizó una revisión de calidad de los datos, donde se identificaron aspectos como tipos de datos incorrectos y valores que requerían tratamiento. Uno de los principales casos correspondió a `TotalCharges`, que inicialmente se encontraba almacenada como texto a pesar de representar información numérica.

Posteriormente se realizó un análisis exploratorio para identificar patrones asociados al abandono. Entre los principales resultados se observaron diferencias importantes según el tipo de contrato y la antigüedad del cliente.

También se analizaron las relaciones entre variables numéricas mediante correlación de Spearman y las asociaciones entre variables categóricas mediante V de Cramér.

Finalmente, se consideraron aspectos relacionados con el desbalance de clases, posibles sesgos, privacidad y uso responsable de un futuro modelo predictivo.

Con los datos preparados y los patrones explorados, el siguiente paso del proyecto consiste en desarrollar un modelo de Machine Learning capaz de estimar la probabilidad de abandono de los clientes y evaluar su desempeño mediante métricas apropiadas.

---

# 28. Trabajo futuro

Como continuación del proyecto se propone:

1. Preparar las variables para Machine Learning.
2. Codificar las variables categóricas.
3. Separar los datos en entrenamiento y prueba.
4. Aplicar técnicas para abordar el desbalance de clases.
5. Entrenar diferentes modelos de clasificación.
6. Comparar sus resultados.
7. Evaluar Precision, Recall, F1-Score y ROC-AUC.
8. Analizar la matriz de confusión.
9. Seleccionar el modelo con mejor desempeño.
10. Interpretar las variables más relevantes.
11. Evaluar posibles sesgos del modelo.
12. Proponer una estrategia de utilización del modelo para retención de clientes.

---

# 29. Autores

**Proyecto académico - MLY1101 Machine Learning**
Luis Hurtubia
Desarrollado como parte de la Evaluación Parcial N°1.

---

## 30. Licencia y uso

El dataset utilizado corresponde a un conjunto de datos público disponible en Kaggle y se utiliza exclusivamente para desarrollar el análisis presentado en este proyecto.
