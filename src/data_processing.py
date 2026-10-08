"""Pipeline completa de preprocesamiento para Telco Customer Churn."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# ============================================================
# 1. RUTAS Y CONFIGURACION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT_DIR / "data" / "split" / "telco_train.csv"
TEST_PATH = ROOT_DIR / "data" / "split" / "telco_test.csv"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Churn"
ID_COLUMN = "customerID"

TARGET_MAPPING = {
    "No": 0,
    "Yes": 1
}


# ============================================================
# 2. CARGA DE DATOS
# ============================================================

def load_data():
    """
    Carga los conjuntos Train y Test previamente separados.
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo Train: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo Test: {TEST_PATH}"
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    return train, test


# ============================================================
# 3. LIMPIEZA DETERMINISTA
# ============================================================

def clean_dataset(df):
    """
    Aplica transformaciones de limpieza que son iguales
    para Train y Test.

    Estas transformaciones NO aprenden parametros de los datos.
    """

    df = df.copy()

    # --------------------------------------------------------
    # 3.1 Eliminar espacios al inicio y final de textos
    # --------------------------------------------------------

    text_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    # --------------------------------------------------------
    # 3.2 TotalCharges
    # --------------------------------------------------------
    # En el EDA se detectaron valores vacios.
    # Los casos encontrados corresponden a clientes con
    # tenure = 0, por lo que se interpretan como 0.
    #
    # Primero reemplazamos los textos vacios por 0
    # y luego convertimos la columna a numerica.

    df["TotalCharges"] = df["TotalCharges"].replace(
        "", "0"
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # En caso de que existiera algun valor que no pudiera
    # convertirse a numerico, se mantiene como NaN para que
    # el imputador del pipeline pueda tratarlo posteriormente.

    return df


# ============================================================
# 4. VALIDACIONES
# ============================================================

def validate_dataset(df, dataset_name):
    """
    Valida estructura, identificadores y variable objetivo.
    """

    # --------------------------------------------------------
    # 4.1 Columnas obligatorias
    # --------------------------------------------------------

    required_columns = {
        ID_COLUMN,
        TARGET,
        "TotalCharges"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: faltan columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # 4.2 customerID duplicados
    # --------------------------------------------------------

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(
            f"{dataset_name}: existen customerID duplicados."
        )

    # --------------------------------------------------------
    # 4.3 Valores validos de Churn
    # --------------------------------------------------------

    invalid_target = (
        set(df[TARGET].dropna().unique())
        - set(TARGET_MAPPING.keys())
    )

    if invalid_target:
        raise ValueError(
            f"{dataset_name}: valores invalidos en Churn: "
            f"{sorted(invalid_target)}"
        )

    # --------------------------------------------------------
    # 4.4 Informacion general
    # --------------------------------------------------------

    print(f"\n--- {dataset_name} ---")
    print(f"Filas: {len(df)}")
    print(f"Columnas: {df.shape[1]}")
    print(f"Duplicados: {df.duplicated().sum()}")
    print(f"Nulos totales: {df.isna().sum().sum()}")
    print(f"Tipo TotalCharges: {df['TotalCharges'].dtype}")


# ============================================================
# 5. SEPARACION DE ID, X E Y
# ============================================================

def split_features_target(df):
    """
    Separa:

    - customerID: identificador
    - X: variables predictoras
    - y: variable objetivo
    """

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    customer_ids = df[ID_COLUMN].copy()

    # --------------------------------------------------------
    # Variable objetivo
    # --------------------------------------------------------

    y = df[TARGET].map(TARGET_MAPPING)

    if y.isna().any():
        raise ValueError(
            "Churn contiene valores que no pudieron "
            "mapearse correctamente a 0/1."
        )

    # --------------------------------------------------------
    # Variables predictoras
    # --------------------------------------------------------

    X = df.drop(
        columns=[
            ID_COLUMN,
            TARGET
        ]
    )

    return X, y, customer_ids


# ============================================================
# 6. CONSTRUCCION DEL PIPELINE
# ============================================================

def build_pipeline(X_train):
    """
    Construye el preprocesador utilizando la estructura
    de Train.

    Importante:
    En esta funcion NO se hace fit.
    """

    # --------------------------------------------------------
    # Variables continuas
    # --------------------------------------------------------
    # Estas variables presentan comportamiento numerico
    # continuo y serán escaladas con RobustScaler.

    continuous_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    # --------------------------------------------------------
    # Variable binaria
    # --------------------------------------------------------
    # SeniorCitizen contiene solamente 0 y 1.
    # No se escala.

    binary_columns = [
        "SeniorCitizen"
    ]

    # --------------------------------------------------------
    # Variables categoricas
    # --------------------------------------------------------

    categorical_columns = [
        column
        for column in X_train.columns
        if column not in continuous_columns
        and column not in binary_columns
    ]

    # --------------------------------------------------------
    # Pipeline numerico
    # --------------------------------------------------------
    # Median se utiliza como respaldo en caso de que exista
    # algun valor numerico faltante no contemplado previamente.
    #
    # RobustScaler utiliza mediana y rango intercuartilico,
    # por lo que es menos sensible a valores extremos que
    # StandardScaler.

    continuous_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                RobustScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Pipeline categorico
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # --------------------------------------------------------
    # ColumnTransformer
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous",
                continuous_pipeline,
                continuous_columns
            ),
            (
                "binary",
                "passthrough",
                binary_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    return (
        preprocessor,
        continuous_columns,
        binary_columns,
        categorical_columns
    )


# ============================================================
# 7. TRANSFORMACION A DATAFRAME
# ============================================================

def transform_to_dataframe(preprocessor, X):
    """
    Aplica el preprocesador y devuelve un DataFrame
    con los nombres de las variables resultantes.
    """

    values = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()

    return pd.DataFrame(
        values,
        columns=feature_names,
        index=X.index
    )


# ============================================================
# 8. GUARDADO DE RESULTADOS
# ============================================================

def save_outputs(
    X_train,
    X_test,
    y_train,
    y_test,
    ids_train,
    ids_test
):
    """
    Guarda los datasets procesados.
    """

    # --------------------------------------------------------
    # Variables predictoras
    # --------------------------------------------------------

    X_train.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False
    )

    # --------------------------------------------------------
    # Variable objetivo
    # --------------------------------------------------------

    y_train.to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False,
        header=[TARGET]
    )

    y_test.to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False,
        header=[TARGET]
    )

    # --------------------------------------------------------
    # Identificadores
    # --------------------------------------------------------
    # Se guardan para trazabilidad, pero no entran al modelo.

    ids_train.to_csv(
        PROCESSED_DIR / "customerID_train.csv",
        index=False,
        header=[ID_COLUMN]
    )

    ids_test.to_csv(
        PROCESSED_DIR / "customerID_test.csv",
        index=False,
        header=[ID_COLUMN]
    )


# ============================================================
# 9. EJECUCION PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("PIPELINE COMPLETA - TELCO CUSTOMER CHURN")
    print("=" * 70)

    # ========================================================
    # PASO 1 - CARGAR DATOS
    # ========================================================

    print("\n[1/7] Cargando Train y Test...")

    train, test = load_data()

    print(f"Train original: {train.shape}")
    print(f"Test original : {test.shape}")

    # ========================================================
    # PASO 2 - LIMPIEZA
    # ========================================================

    print("\n[2/7] Aplicando limpieza determinista...")

    train = clean_dataset(train)
    test = clean_dataset(test)

    # ========================================================
    # PASO 3 - VALIDACION
    # ========================================================

    print("\n[3/7] Validando datasets...")

    validate_dataset(
        train,
        "TRAIN"
    )

    validate_dataset(
        test,
        "TEST"
    )

    # ========================================================
    # PASO 4 - SEPARAR X, Y E ID
    # ========================================================

    print("\n[4/7] Separando X, y e identificadores...")

    X_train, y_train, ids_train = split_features_target(train)

    X_test, y_test, ids_test = split_features_target(test)

    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")

    print(
        f"Churn Train: "
        f"{y_train.mean() * 100:.2f}%"
    )

    print(
        f"Churn Test : "
        f"{y_test.mean() * 100:.2f}%"
    )

    # ========================================================
    # PASO 5 - CONSTRUIR PREPROCESADOR
    # ========================================================

    print(
        "\n[5/7] Construyendo pipeline "
        "de preprocesamiento..."
    )

    (
        preprocessor,
        continuous_columns,
        binary_columns,
        categorical_columns
    ) = build_pipeline(X_train)

    print(
        "\nVariables continuas "
        "(RobustScaler):"
    )

    print(
        continuous_columns
    )

    print(
        "\nVariables binarias "
        "(sin escalamiento):"
    )

    print(
        binary_columns
    )

    print(
        "\nVariables categoricas "
        "(One-Hot Encoding):"
    )

    print(
        categorical_columns
    )

    # ========================================================
    # PASO 6 - FIT TRAIN / TRANSFORM TRAIN Y TEST
    # ========================================================

    print(
        "\n[6/7] Ajustando transformaciones "
        "SOLO con Train..."
    )

    # --------------------------------------------------------
    # IMPORTANTE:
    #
    # fit() aprende:
    # - medianas de imputacion
    # - parametros de RobustScaler
    # - categorias del OneHotEncoder
    #
    # Todo esto se aprende exclusivamente desde Train.
    # --------------------------------------------------------

    preprocessor.fit(X_train)

    print(
        "Fit realizado exclusivamente sobre Train."
    )

    # --------------------------------------------------------
    # Train:
    # fit + transform
    # --------------------------------------------------------

    print(
        "Transformando Train..."
    )

    X_train_processed = transform_to_dataframe(
        preprocessor,
        X_train
    )

    # --------------------------------------------------------
    # Test:
    # SOLO transform
    # --------------------------------------------------------

    print(
        "Transformando Test sin realizar fit..."
    )

    X_test_processed = transform_to_dataframe(
        preprocessor,
        X_test
    )

    # ========================================================
    # PASO 7 - GUARDAR RESULTADOS
    # ========================================================

    print(
        "\n[7/7] Guardando resultados..."
    )

    save_outputs(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        ids_train,
        ids_test
    )

    # --------------------------------------------------------
    # Guardar pipeline entrenado
    # --------------------------------------------------------

    pipeline_path = (
        MODELS_DIR /
        "preprocessing_pipeline.joblib"
    )

    joblib.dump(
        preprocessor,
        pipeline_path
    )

    # ========================================================
    # VALIDACION FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDACION FINAL")
    print("=" * 70)

    print(
        f"X_train procesado: "
        f"{X_train_processed.shape}"
    )

    print(
        f"X_test procesado : "
        f"{X_test_processed.shape}"
    )

    print(
        "Nulos X_train:",
        int(
            X_train_processed
            .isna()
            .sum()
            .sum()
        )
    )

    print(
        "Nulos X_test :",
        int(
            X_test_processed
            .isna()
            .sum()
            .sum()
        )
    )

    print(
        "customerID dentro de X:",
        ID_COLUMN in X_train_processed.columns
    )

    print(
        "Churn dentro de X:",
        TARGET in X_train_processed.columns
    )

    # --------------------------------------------------------
    # Comprobar que Train y Test tienen las mismas columnas
    # --------------------------------------------------------

    same_columns = (
        list(X_train_processed.columns)
        ==
        list(X_test_processed.columns)
    )

    print(
        "Mismas variables en Train/Test:",
        same_columns
    )

    if not same_columns:
        raise ValueError(
            "Train y Test no tienen las mismas "
            "variables despues del procesamiento."
        )

    # --------------------------------------------------------
    # Archivos generados
    # --------------------------------------------------------

    print(
        "\nArchivos generados en "
        "data/processed:"
    )

    print("  - X_train.csv")
    print("  - X_test.csv")
    print("  - y_train.csv")
    print("  - y_test.csv")
    print("  - customerID_train.csv")
    print("  - customerID_test.csv")

    print(
        "\nPipeline guardada en:"
    )

    print(
        f"  - {pipeline_path}"
    )

    print(
        "\nProceso terminado correctamente."
    )


# ============================================================
# 10. EJECUTAR SCRIPT
# ============================================================

if __name__ == "__main__":
    main()