"""Shared code for the disease prediction notebooks.

All three notebooks (heart disease, diabetes, cardiovascular disease) use the same steps,
so the steps live here once:

1. load and clean a dataset
2. split it once into a training set and a test set (stratified, fixed seed)
3. train every model as a scikit-learn pipeline (imputation and scaling are fitted on the
   training data only, so nothing leaks from the test set)
4. score each model with 5-fold cross-validation on the training set and once on the test set
5. train a small neural network (Keras) on the same split
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------- data loading

def load_heart():
    """UCI heart disease data (303 patients). Target: 1 = heart disease."""
    df = pd.read_csv(DATA_DIR / "heart.csv")
    df = df.drop_duplicates().reset_index(drop=True)
    return df.drop(columns="target"), df["target"]


def load_diabetes():
    """Pima Indians diabetes data (768 women). Target: 1 = diabetes.

    In this dataset a value of 0 for glucose, blood pressure, skin thickness, insulin or BMI
    means "not measured", so these zeros are turned into missing values and imputed later.
    """
    df = pd.read_csv(DATA_DIR / "diabetes.csv")
    not_measured = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[not_measured] = df[not_measured].replace(0, np.nan)
    return df.drop(columns="Outcome"), df["Outcome"]


def load_cardio():
    """Cardiovascular disease data (70,000 patients). Target: 1 = cardiovascular disease.

    Age is converted from days to years, BMI is computed from height and weight, and rows
    with physically impossible blood pressure, height or weight are removed.
    """
    df = pd.read_csv(DATA_DIR / "cardio_train.csv", sep=";").drop(columns="id")
    df["age"] = (df["age"] / 365.25).round(1)
    df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(1)
    plausible = (
        df["ap_hi"].between(80, 250)
        & df["ap_lo"].between(40, 160)
        & (df["ap_lo"] < df["ap_hi"])
        & df["height"].between(120, 220)
        & df["weight"].between(30, 200)
    )
    df = df[plausible].reset_index(drop=True)
    return df.drop(columns="cardio"), df["cardio"]


def split(X, y, test_size=0.2):
    """One fixed, stratified train/test split used by every model."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


# ---------------------------------------------------------------- models

def _pipe(model, scale=True):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def make_models(large=False, with_smote=False):
    """The classical models compared in the project, each wrapped in a pipeline.

    large=True swaps the kernel SVM for a linear SVM, which scales to tens of thousands
    of rows. with_smote=True adds a logistic regression trained on SMOTE-balanced data.
    """
    models = {
        "Baseline (majority class)": _pipe(DummyClassifier(strategy="most_frequent"), scale=False),
        "Logistic regression": _pipe(LogisticRegression(max_iter=1000)),
        "k-nearest neighbours": _pipe(KNeighborsClassifier(n_neighbors=15)),
        "Naive Bayes": _pipe(GaussianNB()),
        "SVM (linear)" if large else "SVM (RBF kernel)": _pipe(
            LinearSVC(dual="auto", random_state=SEED) if large else SVC(random_state=SEED)
        ),
        "Decision tree": _pipe(DecisionTreeClassifier(max_depth=5, random_state=SEED), scale=False),
        "Random forest": _pipe(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
            scale=False,
        ),
    }
    if with_smote:
        models["Logistic regression + SMOTE"] = ImbPipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=SEED)),
            ("model", LogisticRegression(max_iter=1000)),
        ])
    return models


def _scores(model, X):
    """Probability of class 1 if the model has one, otherwise its decision value."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _test_metrics(y_true, y_pred, y_score):
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred),
        "test_f1": f1_score(y_true, y_pred),
        "test_roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_score)) > 1 else 0.5,
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Cross-validate on the training set, then fit and score once on the test set."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, model in models.items():
        cvres = cross_validate(model, X_train, y_train, cv=cv, scoring=["accuracy", "roc_auc"])
        model.fit(X_train, y_train)
        fitted[name] = model
        rows.append({
            "model": name,
            "cv_accuracy": cvres["test_accuracy"].mean(),
            "cv_accuracy_std": cvres["test_accuracy"].std(),
            "cv_roc_auc": cvres["test_roc_auc"].mean(),
            **_test_metrics(y_test, model.predict(X_test), _scores(model, X_test)),
        })
    table = pd.DataFrame(rows).set_index("model").round(3)
    return table, fitted


# ---------------------------------------------------------------- neural network

def train_ann(X_train, X_test, y_train, y_test, epochs=200, verbose=0):
    """Small Keras network on the same split. Preprocessing is fitted on the training set only.

    Early stopping watches a validation part of the training set, so the test set is used
    exactly once, for the final score.
    """
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    Xtr = prep.fit_transform(X_train)
    Xte = prep.transform(X_test)

    model = keras.Sequential([
        keras.Input(shape=(Xtr.shape[1],)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    history = model.fit(Xtr, np.asarray(y_train), validation_split=0.2, epochs=epochs,
                        batch_size=32 if len(Xtr) < 5000 else 256, callbacks=[stop], verbose=verbose)

    score = model.predict(Xte, verbose=0).ravel()
    metrics = _test_metrics(y_test, (score > 0.5).astype(int), score)
    metrics["epochs_trained"] = len(history.history["loss"])
    return model, history, {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}


# ---------------------------------------------------------------- plots

def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=120, bbox_inches="tight")


def plot_class_balance(y, labels, title):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = y.value_counts().sort_index()
    ax.bar([labels[i] for i in counts.index], counts.values, color=["#4C9BE8", "#E8704C"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / counts.sum():.0%})", ha="center", va="bottom")
    ax.set_title(title)
    ax.set_ylabel("Patients")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_correlation_with_target(X, y, title):
    corr = X.assign(_target=y).corr(numeric_only=True)["_target"].drop("_target").sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(corr) + 1))
    ax.barh(corr.index, corr.values, color=np.where(corr.values > 0, "#E8704C", "#4C9BE8"))
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Correlation with the target")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_model_comparison(table, title):
    t = table.sort_values("test_roc_auc")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(t) + 1))
    ax.barh(t.index, t["test_roc_auc"], color="#4C9BE8", label="ROC-AUC (test)")
    ax.scatter(t["test_accuracy"], t.index, color="#E8704C", zorder=3, label="Accuracy (test)")
    ax.set_xlim(0.4, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_roc_curves(fitted, X_test, y_test, title, skip=("Baseline (majority class)",)):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        if name in skip:
            continue
        RocCurveDisplay.from_predictions(y_test, _scores(model, X_test), name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title)
    return fig


def plot_confusion(model, X_test, y_test, labels, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test), display_labels=labels,
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(title)
    return fig


def plot_feature_importance(forest_pipeline, feature_names, title, top=12):
    imp = pd.Series(forest_pipeline.named_steps["model"].feature_importances_, index=feature_names)
    imp = imp.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(imp) + 1))
    ax.barh(imp.index, imp.values, color="#4C9BE8")
    ax.set_title(title)
    ax.set_xlabel("Importance (random forest)")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_history(history, title):
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(h["loss"], label="training")
    axes[0].plot(h["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(h["accuracy"], label="training")
    axes[1].plot(h["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    return fig


sns.set_theme(style="white", context="notebook")
