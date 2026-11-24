# Disease prediction with machine learning

Final project of the SPICED Academy Data Science bootcamp (2021), rebuilt in November 2026 with current
libraries and a cleaner, fairer evaluation.

**Question:** how well can routine medical measurements predict three diseases, and which kind of model works best?

| Notebook | Dataset | Patients | Best test ROC-AUC | Test accuracy | Baseline accuracy |
|---|---|---|---|---|---|
| [01_heart_disease](notebooks/01_heart_disease.ipynb) | UCI Heart Disease (Cleveland) | 302 | 0.90 (k-nearest neighbours) | 82 % | 54 % |
| [02_diabetes](notebooks/02_diabetes.ipynb) | Pima Indians Diabetes | 768 | 0.81 (logistic regression) | 71 % | 65 % |
| [03_cardiovascular](notebooks/03_cardiovascular.ipynb) | Cardiovascular Disease (Kaggle) | 68,608 after cleaning | 0.80 (neural network) | 73 % | 50 % |

*Baseline: always predicting the larger class.*

## Method

The same steps for all three datasets, implemented once in [`src/disease_ml.py`](src/disease_ml.py):

1. **Load and clean.** Diabetes: zeros that mean "not measured" become missing values. Cardio: age in years,
   BMI from height and weight, rows with impossible blood pressure, height or weight removed.
2. **One stratified train/test split** (80/20, fixed seed) shared by every model.
3. **Pipelines.** Missing values are filled and features scaled inside each model pipeline, fitted on the training
   data only.
4. **Seven classical models** (baseline, logistic regression, k-nearest neighbours, naive Bayes, SVM, decision tree,
   random forest) compared with 5-fold cross-validation on the training set and scored once on the test set.
   For diabetes, also logistic regression trained on SMOTE-balanced data.
5. **A small neural network** in Keras (16 → 8 → 1, dropout, early stopping) on the same split.

## Findings

- **Heart disease** is the easiest to predict: the best models find about 94 % of sick patients in the test set.
  The test set has only 61 patients, so small differences between the top models are not meaningful.
- **Diabetes:** standard models miss about half of the women with diabetes. Balancing the training data with SMOTE
  raises recall from 50 % to 65 % at the same ROC-AUC.
- **Cardiovascular disease:** all models end up between 0.78 and 0.80 ROC-AUC. The limit is the information in the
  data, not the model. Blood pressure, age and BMI carry most of the signal.
- The neural network never clearly beats logistic regression or a random forest on data of this size.

## What changed in the 2026 rebuild

The original notebooks from 2021 had grown to over 100 cells each and no longer ran with current Keras.
While rebuilding them I also fixed problems in the evaluation:

- Every model used a different train/test split, so the scores could not be compared. Now there is one split.
- The data was scaled before splitting, so information from the test set leaked into training. Scaling is now
  part of each pipeline.
- `index_col=0` when reading the CSV files silently turned the first feature (age, number of pregnancies) into the
  row index, so the models never saw it.
- The neural network had its input size hard-coded for the heart data and was evaluated on its own training data.
- Zeros meaning "not measured" in the diabetes data and impossible blood pressure values in the cardio data were
  treated as real measurements.

The old notebooks are still available in the Git history of this repository.

## Run it

```bash
pip install -r ../requirements.txt
jupyter lab notebooks/
```

Figures are saved to [`results/figures/`](results/figures), result tables to [`results/`](results).

## Data sources

- Heart disease: UCI Machine Learning Repository, Heart Disease dataset (Cleveland subset), as published on Kaggle
- Diabetes: Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases), via Kaggle
- Cardiovascular disease: "Cardiovascular Disease dataset" by Svetlana Ulianova on Kaggle

These are public teaching datasets. The models here are for learning, not for medical use.
