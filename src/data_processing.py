"""Pipeline completa de preprocesamiento para Telco Customer Churn."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
TARGET_MAPPING = {"No": 0, "Yes": 1}


# ============================================================
# 2. CARGA
# ============================================================

def load_data():
    """Carga los datasets Train y Test ya separados."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"No se encontro Train: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"No se encontro Test: {TEST_PATH}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


# ============================================================
# 3. LIMPIEZA DETERMINISTA
# ============================================================

def clean_dataset(df):
    """Limpieza que puede aplicarse de la misma forma a Train y Test."""
    df = df.copy()

    # Quitar espacios al inicio/final de las variables de texto.
    text_columns = df.select_dtypes(include="object").columns
    for column in text_columns:
        df[column] = df[column].str.strip()

    # TotalCharges: los espacios vacios detectados en EDA pasan a NA
    # y luego se convierten a numero.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    return df


# ============================================================
# 4. VALIDACIONES
# ============================================================

def validate_dataset(df, dataset_name):
    """Valida estructura y valores basicos."""
    required = {ID_COLUMN, TARGET, "TotalCharges"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset_name}: faltan columnas requeridas: {sorted(missing)}"
        )

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"{dataset_name}: existen customerID duplicados.")

    invalid_target = set(df[TARGET].dropna().unique()) - set(TARGET_MAPPING)
    if invalid_target:
        raise ValueError(
            f"{dataset_name}: valores invalidos en Churn: {sorted(invalid_target)}"
        )

    print(f"\n--- {dataset_name} ---")
    print(f"Filas: {len(df)}")
    print(f"Columnas: {df.shape[1]}")
    print(f"Duplicados: {df.duplicated().sum()}")
    print(f"Nulos totales: {df.isna().sum().sum()}")
    print(f"Tipo TotalCharges: {df['TotalCharges'].dtype}")


# ============================================================
# 5. X / y
# ============================================================

def split_features_target(df):
    """Separa ID, predictores X y objetivo y."""
    customer_ids = df[ID_COLUMN].copy()

    y = df[TARGET].map(TARGET_MAPPING)
    if y.isna().any():
        raise ValueError("Churn contiene valores que no pudieron mapearse a 0/1.")

    X = df.drop(columns=[ID_COLUMN, TARGET])
    return X, y, customer_ids


# ============================================================
# 6. PIPELINE DE PREPROCESAMIENTO
# ============================================================

def build_pipeline(X_train):
    """Construye el preprocesador y determina columnas usando SOLO Train."""
    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_columns, categorical_columns


def transform_to_dataframe(preprocessor, X):
    """Transforma X y devuelve DataFrame con nombres de variables."""
    values = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(values, columns=feature_names, index=X.index)


# ============================================================
# 7. GUARDADO
# ============================================================

def save_outputs(
    X_train, X_test, y_train, y_test, ids_train, ids_test
):
    """Guarda los datasets listos para modelamiento."""
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False, header=[TARGET])
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False, header=[TARGET])

    # IDs separados: sirven para trazabilidad, pero no entran al modelo.
    ids_train.to_csv(
        PROCESSED_DIR / "customerID_train.csv",
        index=False,
        header=[ID_COLUMN],
    )
    ids_test.to_csv(
        PROCESSED_DIR / "customerID_test.csv",
        index=False,
        header=[ID_COLUMN],
    )


# ============================================================
# 8. EJECUCION PRINCIPAL
# ============================================================

def main():
    print("=" * 65)
    print("PIPELINE COMPLETA - TELCO CUSTOMER CHURN")
    print("=" * 65)

    # 1. Cargar
    print("\n[1/7] Cargando Train y Test...")
    train, test = load_data()
    print(f"Train original: {train.shape}")
    print(f"Test original : {test.shape}")

    # 2. Limpiar
    print("\n[2/7] Aplicando limpieza...")
    train = clean_dataset(train)
    test = clean_dataset(test)

    # 3. Validar
    print("\n[3/7] Validando datasets...")
    validate_dataset(train, "TRAIN")
    validate_dataset(test, "TEST")

    # 4. Separar X/y
    print("\n[4/7] Separando X, y e identificadores...")
    X_train, y_train, ids_train = split_features_target(train)
    X_test, y_test, ids_test = split_features_target(test)

    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"Churn Train: {y_train.mean() * 100:.2f}%")
    print(f"Churn Test : {y_test.mean() * 100:.2f}%")

    # 5. Construir preprocesador usando Train
    print("\n[5/7] Construyendo pipeline desde Train...")
    preprocessor, numeric_columns, categorical_columns = build_pipeline(X_train)

    print("Variables numericas:", numeric_columns)
    print("Variables categoricas:", categorical_columns)

    # 6. FIT SOLO TRAIN y luego transform Train/Test
    print("\n[6/7] Ajustando SOLO con Train y transformando ambos conjuntos...")
    preprocessor.fit(X_train)

    X_train_processed = transform_to_dataframe(preprocessor, X_train)
    X_test_processed = transform_to_dataframe(preprocessor, X_test)

    # 7. Guardar
    print("\n[7/7] Guardando resultados...")
    save_outputs(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        ids_train,
        ids_test,
    )

    joblib.dump(
        preprocessor,
        MODELS_DIR / "preprocessing_pipeline.joblib",
    )

    # Validaciones finales
    print("\n" + "=" * 65)
    print("VALIDACION FINAL")
    print("=" * 65)
    print(f"X_train procesado: {X_train_processed.shape}")
    print(f"X_test procesado : {X_test_processed.shape}")
    print(
        "Nulos X_train:",
        int(X_train_processed.isna().sum().sum()),
    )
    print(
        "Nulos X_test :",
        int(X_test_processed.isna().sum().sum()),
    )
    print("customerID dentro de X:", ID_COLUMN in X_train_processed.columns)
    print("Churn dentro de X:", TARGET in X_train_processed.columns)
    print("\nArchivos generados en data/processed:")
    print("  - X_train.csv")
    print("  - X_test.csv")
    print("  - y_train.csv")
    print("  - y_test.csv")
    print("  - customerID_train.csv")
    print("  - customerID_test.csv")
    print("\nPipeline guardada en:")
    print("  - models/preprocessing_pipeline.joblib")
    print("\nProceso terminado correctamente.")


if __name__ == "__main__":
    main()
