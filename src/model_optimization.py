
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 1. CONFIGURACIÓN DEL PROYECTO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "split" / "telco_train.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "optuna"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 30

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

BINARY_COLUMNS = [
    "SeniorCitizen",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


# ============================================================
# 2. CARGA Y LIMPIEZA DE LOS DATOS DE TRAIN
# ============================================================

def load_and_clean_train():
    """
    Carga únicamente el conjunto Train y aplica limpieza
    determinista antes de la validación cruzada.
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo Train: {TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    required_columns = (
        [TARGET, ID_COLUMN]
        + NUMERIC_COLUMNS
        + BINARY_COLUMNS
        + CATEGORICAL_COLUMNS
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en el dataset: {missing_columns}"
        )

    # Limpiar espacios de columnas de texto.
    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    # Convertir TotalCharges a numérico.
    # Los espacios vacíos se convierten temporalmente en NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # En este dataset, los TotalCharges vacíos corresponden
    # a clientes con tenure = 0, sin cargos acumulados.
    empty_total = df["TotalCharges"].isna()

    invalid_empty_rows = empty_total & (df["tenure"] != 0)

    if invalid_empty_rows.any():
        raise ValueError(
            "Hay TotalCharges vacíos en filas con tenure distinto de 0. "
            "Revisa estos registros antes de continuar."
        )

    df.loc[empty_total, "TotalCharges"] = 0

    # Validar variable objetivo.
    df = df.dropna(subset=[TARGET])

    df[TARGET] = df[TARGET].map({
        "No": 0,
        "Yes": 1,
    })

    if df[TARGET].isna().any():
        raise ValueError(
            "La columna Churn contiene valores distintos de Yes/No."
        )

    # Separar identificador, variables predictoras y objetivo.
    customer_ids = df[ID_COLUMN].copy()

    X = df[
        NUMERIC_COLUMNS
        + BINARY_COLUMNS
        + CATEGORICAL_COLUMNS
    ].copy()

    y = df[TARGET].astype(int).copy()

    # Validaciones básicas.
    if X.isna().all().any():
        raise ValueError(
            "Hay columnas completamente vacías en las variables predictoras."
        )

    if y.nunique() != 2:
        raise ValueError(
            "Churn debe contener las dos clases: 0 y 1."
        )

    print("=" * 60)
    print("CARGA Y LIMPIEZA DE TRAIN")
    print("=" * 60)
    print(f"Registros Train: {len(X)}")
    print(f"Variables predictoras originales: {X.shape[1]}")
    print(f"Clientes únicos: {customer_ids.nunique()}")
    print(f"Churn Yes: {y.mean():.2%}")
    print(f"Churn No: {(1 - y.mean()):.2%}")
    print("customerID excluido de las variables predictoras.")

    return X, y


# ============================================================
# 3. CREAR EL PREPROCESAMIENTO
# ============================================================

def create_preprocessor():
    """
    Crea transformaciones que se ajustarán dentro de cada fold:
    - RobustScaler para variables continuas.
    - SeniorCitizen se conserva como variable binaria.
    - OneHotEncoder para variables categóricas.
    """

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("binary", binary_pipeline, BINARY_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# 4. CREAR PIPELINE COMPLETO
# ============================================================

def create_model_pipeline(model):
    """
    Une el preprocesamiento y el clasificador.
    El pipeline se ajusta desde cero en cada fold.
    """

    return Pipeline(steps=[
        ("preprocessor", create_preprocessor()),
        ("classifier", model),
    ])


# ============================================================
# 5. DEFINIR LOS MODELOS Y SUS ESPACIOS DE BÚSQUEDA
# ============================================================

def suggest_model(trial, model_name):

    if model_name == "logistic_regression":

        model = LogisticRegression(
            C=trial.suggest_float(
                "C", 0.001, 100.0, log=True
            ),
            class_weight=trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )

    elif model_name == "random_forest":

        model = RandomForestClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators", 100, 500, step=100
            ),
            max_depth=trial.suggest_categorical(
                "max_depth", [None, 5, 10, 15, 20, 30]
            ),
            min_samples_split=trial.suggest_int(
                "min_samples_split", 2, 20
            ),
            min_samples_leaf=trial.suggest_int(
                "min_samples_leaf", 1, 10
            ),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            class_weight=trial.suggest_categorical(
                "class_weight",
                [None, "balanced", "balanced_subsample"]
            ),
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    return model


# ============================================================
# 6. OPTIMIZAR UN MODELO CON OPTUNA Y VALIDACIÓN CRUZADA
# ============================================================

def optimize_model(model_name, X, y):

    print("\n" + "=" * 60)
    print(f"OPTIMIZANDO: {model_name.upper()}")
    print("=" * 60)

    # Se reutilizan las mismas particiones para comparar modelos
    # de forma más consistente.
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial):

        model = suggest_model(trial, model_name)

        pipeline = create_model_pipeline(model)

        # El preprocesamiento se ajusta únicamente con el
        # subconjunto de entrenamiento de cada fold.
        scores = cross_val_score(
            estimator=pipeline,
            X=X,
            y=y,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
        )

        mean_f1 = float(np.mean(scores))

        trial.set_user_attr(
            "std_f1_cv", float(np.std(scores))
        )

        return mean_f1

    study = optuna.create_study(
        direction="maximize",
        study_name=model_name,
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
    )

    print(f"Ensayos completados: {len(study.trials)}")
    print(f"Mejor F1 promedio CV: {study.best_value:.4f}")
    print("Mejores hiperparámetros:")

    for parameter, value in study.best_params.items():
        print(f"  {parameter}: {value}")

    # Guardar todos los ensayos para analizar los resultados.
    trials_path = REPORTS_DIR / f"{model_name}_trials.csv"

    study.trials_dataframe(
        attrs=("number", "value", "params", "state")
    ).to_csv(trials_path, index=False)

    # Guardar los mejores parámetros y resultados.
    results_path = REPORTS_DIR / f"{model_name}_best.json"

    result = {
        "model": model_name,
        "optimization_metric": "F1-score",
        "positive_class": "Churn = Yes (1)",
        "cv_folds": N_SPLITS,
        "n_trials": N_TRIALS,
        "best_cv_f1": study.best_value,
        "best_cv_f1_std": study.best_trial.user_attrs.get(
            "std_f1_cv"
        ),
        "best_params": study.best_params,
    }

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # Reconstruir el mejor clasificador.
    best_model = build_best_model(
        model_name,
        study.best_params,
    )

    # Entrenar el pipeline final con TODO el conjunto Train.
    # No se utiliza Test en esta etapa.
    final_pipeline = create_model_pipeline(best_model)
    final_pipeline.fit(X, y)

    model_path = MODELS_DIR / f"{model_name}_optuna.joblib"
    joblib.dump(final_pipeline, model_path)

    print(f"Resultados guardados en: {trials_path}")
    print(f"Mejores parámetros guardados en: {results_path}")
    print(f"Pipeline final guardado en: {model_path}")

    return {
        "model": model_name,
        "best_cv_f1": study.best_value,
        "best_cv_f1_std": study.best_trial.user_attrs.get(
            "std_f1_cv"
        ),
        "best_params": study.best_params,
        "model_path": str(model_path),
    }


# ============================================================
# 7. RECONSTRUIR EL MODELO CON LOS MEJORES HIPERPARÁMETROS
# ============================================================

def build_best_model(model_name, params):

    if model_name == "logistic_regression":

        return LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )

    if model_name == "random_forest":

        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Modelo desconocido: {model_name}")


# ============================================================
# 8. EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    # Optuna recibe exclusivamente Train.
    X_train, y_train = load_and_clean_train()

    results = []

    for model_name in [
        "logistic_regression",
        "random_forest",
    ]:
        result = optimize_model(
            model_name,
            X_train,
            y_train,
        )
        results.append(result)

    # Resumen comparativo basado en validación cruzada.
    summary = pd.DataFrame([
        {
            "modelo": result["model"],
            "mejor_f1_cv": result["best_cv_f1"],
            "desviacion_f1_cv": result["best_cv_f1_std"],
            "pipeline_guardado": result["model_path"],
        }
        for result in results
    ])

    summary = summary.sort_values(
        by="mejor_f1_cv",
        ascending=False,
    )

    summary_path = REPORTS_DIR / "optuna_comparison.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("RESUMEN DE OPTUNA")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nResumen guardado en: {summary_path}")
    print("\nOptimización terminada.")


if __name__ == "__main__":
    main()