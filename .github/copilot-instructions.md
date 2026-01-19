# Copilot Instructions for Machine Learning A-Z Codebase

## Project Overview

This is a **Machine Learning practice repository** organized as a course-based learning structure. It contains 30+ Jupyter notebooks covering the entire ML pipeline from data preprocessing to deep learning, with parallel implementations in Python and R.

**Core dependencies**: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tensorflow/keras` (for deep learning sections)

## Architecture & Organization

### Directory Structure
- **Part 1**: Data Preprocessing (missing value imputation, encoding, scaling)
- **Parts 2-3**: Regression & Classification algorithms (SLR, MLR, SVR, Logistic Regression, SVM, etc.)
- **Parts 4-5**: Unsupervised Learning (clustering, association rules)
- **Part 6**: Reinforcement Learning (UCB, Thompson Sampling)
- **Part 7**: NLP
- **Parts 8-9**: Deep Learning & Dimensionality Reduction
- **Part 10**: Model Selection & Boosting (XGBoost, Grid Search, K-Fold CV)

Each section follows: `Section [N] - [Algorithm Name]/Python/` with a single `*.ipynb` notebook.

### Standard Notebook Pattern

All notebooks follow this structure:
1. **Title header** (Markdown)
2. **Library imports** cell (numpy, pandas, matplotlib, scikit-learn utilities)
3. **Dataset loading** cell using `pd.read_csv()` with standard feature/target split: `X = dataset.iloc[:, :-1].values` and `y = dataset.iloc[:, -1].values`
4. **Algorithm-specific preprocessing** (encoding, scaling, train/test split)
5. **Model training** using scikit-learn
6. **Predictions & evaluation**
7. **Visualization** using matplotlib

Example: [simple_linear_regression.ipynb](Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/Python/simple_linear_regression.ipynb)

## Key Patterns & Conventions

### Data Handling
- **Missing values**: Use `sklearn.impute.SimpleImputer` with `strategy='mean'` for numerical columns
- **Categorical encoding**: Use `sklearn.preprocessing.OneHotEncoder` via `ColumnTransformer` for feature columns
- **Feature/target split**: Always extract `X = dataset.iloc[:, :-1]` (all but last column) and `y = dataset.iloc[:, -1]` (last column)

### Model Training
- **Train/test split**: Use `sklearn.model_selection.train_test_split()` with `test_size=1/3, random_state=0` for reproducibility
- **Model instantiation**: Create model object, call `.fit()` on training data, then `.predict()` on test data
- **Naming convention**: Model instances use descriptive names (`regressor`, `classifier`, `kmeans_clusterer`)

### Visualization
- Use `matplotlib.pyplot` with:
  - `plt.scatter()` for data points
  - `plt.plot()` for regression lines
  - `plt.title()`, `plt.xlabel()`, `plt.ylabel()` for labels
  - `plt.show()` to display

## Important Developer Workflows

### Running Notebooks
1. Cell execution order matters - always run from top to bottom
2. Early cells load data; later cells depend on previous model objects
3. For preprocessing notebooks: verify missing data with `dataset.isnull().sum()` before imputation

### Adding New Sections
When implementing a new algorithm notebook:
1. Follow the standard structure above
2. Ensure CSV file exists in same directory (typically `Data.csv` or algorithm-specific CSV like `Salary_Data.csv`)
3. Always include train/test visualization cells after model predictions
4. Use `random_state=0` for reproducible results across runs

### Debugging Data Issues
- **Missing values**: Use `dataset[dataset.isnull().any(axis=1)]` to identify problematic rows
- **Feature/target mismatch**: Check shapes with `X.shape`, `y.shape` after splitting
- **Encoding errors**: Verify categorical columns are properly indexed: `ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])])`

## Scikit-learn API Patterns

Standard pattern across all algorithm notebooks:
```python
from sklearn.Algorithm import AlgorithmClass
model = AlgorithmClass(hyperparameter=value)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Common imports by section:
- **Preprocessing**: `SimpleImputer`, `StandardScaler`, `ColumnTransformer`, `OneHotEncoder`
- **Regression**: `LinearRegression`, `SVR`, `DecisionTreeRegressor`, `RandomForestRegressor`
- **Classification**: `LogisticRegression`, `SVC`, `KNeighborsClassifier`, `DecisionTreeClassifier`
- **Clustering**: `KMeans`, `AgglomerativeClustering`
- **Model Selection**: `train_test_split`, `cross_val_score`, `GridSearchCV`

## Common Pitfalls to Avoid

1. **Forgetting `fit()` before `transform()`**: Imputers and scalers must be fitted on training data first
2. **Encoding mismatch**: OneHotEncoder needs explicit column indices in `ColumnTransformer`
3. **Variable naming**: `X`/`y` for raw data, `X_train`/`X_test`/`y_train`/`y_test` after splitting
4. **CSV path issues**: Data files referenced as relative paths must exist in the notebook's directory

## Integration Points

- All notebooks are **self-contained** - no cross-notebook dependencies
- Each algorithm section can be studied independently
- Datasets are typically course-provided CSVs (not version-controlled; assume they exist locally)
- Python and R implementations are parallel but separate (no interop between them)

---

**Last Updated**: January 2026  
**Primary Language**: Python (Jupyter notebooks)  
**Test/Validation**: Run notebooks end-to-end to validate algorithm implementations
