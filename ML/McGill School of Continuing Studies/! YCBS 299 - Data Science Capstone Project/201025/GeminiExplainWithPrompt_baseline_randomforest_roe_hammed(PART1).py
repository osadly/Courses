Act as a professional data scientist and machine learning expert. Provide clarification of the below code. Start with highl level description follow with a table that has columns: One for each part of the code that its lines are related , a second column for the high level description of what this part of the code does , and how it does it, a third column for more details about that explaining the techniques and the info communicated in the second column, a fourth column for the underlying concepts, algorithms, background techniques , theories, ... for that code part (to explore), a fifth column for best reliable references (with links) to understand more about the what and how of this code and its expected result and how its connected to the code section before it and the code section next to it and a final column on the Datacamp and LinkedIn course that I can follow for more info, knowledge, and details about concepts and theories and technical info covered for this code part (up to three DataCamp and LinkedIn courses)- Use simple clear wording for all section in the provided response

""" # Data Loading and Initial Inspection - Start """
# Data Loading and Initial Inspection Loads the dataset and prints basic summary statistics. Loads data from a Parquet file into a Pandas DataFrame (df). Prints the dataset dimensions (rows/columns), the date range, the number of unique spatial units, and the distribution of the target variable.
# Uses Pandas (pd.read_parquet) for efficient data loading. Uses f-strings (f"...") for clear, formatted output. df.shape gives dimensions. value_counts(normalize=True) shows the proportion of each target class, highlighting potential class imbalance.	
# Panel Data/Time-Series Data: Data observed across multiple entities (cells) and over time. Parquet Format: Columnar storage format, often faster for reading analytical queries. Class Imbalance: Unequal distribution of classes in a classification problem.
# Ref.: Pandas Documentation: read_parquet, Panel Data - Investopedia
# DataCamp: Introduction to Data Science in Python, Data Manipulation with pandas , LinkedIn: Learning Python for Data Analysis and Visualization

df = pd.read_parquet('fire_risk_panel_data_201025.parquet') #you could use the csv version if preferred
print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
print(f"Unique cells: {df['cell_id'].nunique()}")
print(f"Target distribution:")
print(df['Target_Variable'].value_counts(normalize=True))
df.head()
""" # Data Loading and Initial Inspection - End """

""" # Feature Selection and Definition - Start """
# Defines the sets of features (predictors) and the target variable. Features are grouped into meaningful categories: Building, Temporal Lag (past fire counts/target values), Seasonality (month/year/sin/cos transforms), and Spatial Lag (neighbor information).
# Features are defined as Python lists of column names. The Temporal Lag Features and Spatial Lag Features are crucial for time-series and spatial modeling, capturing autoregressive and spatial dependence. PREDICTOR_FEATURES aggregates all these into a single list.
# Feature Engineering: Creating relevant features from raw data. Time Series Lags/Autoregression (AR): Using past values of a variable to predict the future. Seasonality: Periodic fluctuations in time-series data. Spatial Autocorrelation: Dependence among observations located close together in space.
# Ref.: Feature Engineering for Machine Learning - Scikit-learn, Understanding Time Series (AR/MA)
# DataCamp: "DataCamp: Feature Engineering for Machine Learning in Python, Time Series Analysis in Python, LinkedIn: Advanced Feature Engineering for Machine Learning

# Feature sets
BUILDING_FEATURES = ['num_buildings', 'avg_floors', 'median_construction_year']
TEMPORAL_LAG_FEATURES = [
    'fires_lag_1m', 'fires_lag_2m', 'fires_lag_3m', 'fires_lag_6m', 'fires_lag_12m',
    'target_lag_1m', 'target_lag_2m', 'target_lag_3m'
]
SEASONALITY_FEATURES = ['month', 'year', 'month_sin', 'month_cos']
SPATIAL_LAG_FEATURES = [
    'neighbors_fires_lag1m', 'neighbors_target_lag1m',
    'neighbor_count', 'neighbors_fires_avg'
]
# All predictors
PREDICTOR_FEATURES = (
    BUILDING_FEATURES +
    TEMPORAL_LAG_FEATURES +
    SEASONALITY_FEATURES +
    SPATIAL_LAG_FEATURES
)
TARGET = 'Target_Variable'
print(f"Total predictors: {len(PREDICTOR_FEATURES)}")
print(f"All predictors:")
for i, feat in enumerate(PREDICTOR_FEATURES, 1):
    print(f"  {i:2d}. {feat}")   
""" # Feature Selection and Definition - End """

""" # Data Cleaning (Missing Values) - Start """
# Removes rows that have missing values (NaN) in any of the predictor features. This is specifically done to handle missing lagged values that occur at the start of each spatial cell's time series.
# Uses Pandas' df.dropna(subset=...) method. The subset argument ensures rows are dropped only if they are missing in the defined list of PREDICTOR_FEATURES. Prints the count of removed rows.
# Missing Data Handling (Imputation vs. Deletion): Deleting rows (dropna) is a simple strategy, suitable when the missingness is related to the time-series structure (e.g., initial lags) and the number of removed rows is small. Panel Data Structure: Each cell_id starts its time series, and lags for early time steps are often missing.
# Pandas Documentation: dropna, Handling Missing Data - Towards Data Science
# DataCamp: Preprocessing for Machine Learning in Python, Dealing with Missing Data in R

# Leaving this here on how to remove rows with missing lagged values
# In the group's codebase (using 2.5km by 2.5km grid), we may have missing values- although not applicable here.
# This ensures we only train on complete observations

df_clean = df.dropna(subset=PREDICTOR_FEATURES)
print(f"Original data: {len(df):,} rows")
print(f"After removing NaNs: {len(df_clean):,} rows")
print(f"Removed: {len(df) - len(df_clean):,} rows")
# Convert year_month to datetime for easier manipulation
df_clean['year_month_dt'] = pd.to_datetime(df_clean['year_month'])
df_clean = df_clean.sort_values(['cell_id', 'year_month_dt'])
print(f"Date range: {df_clean['year_month'].min()} to {df_clean['year_month'].max()}")
"""## Exploratory Data Analysis"""
X_all = df_clean[PREDICTOR_FEATURES]
X_all.head()
""" # Data Cleaning (Missing Values) - End """

""" # Exploratory Data Analysis (EDA) - Correlation - Start """
# Calculates and visualizes the correlation between all predictor features to identify potential multicollinearity.
# Calculates the Pearson Correlation Matrix using X_all.corr(). Visualizes it using a Seaborn Heatmap (sns.heatmap) for easy interpretation. Identifies pairs with absolute correlation greater than 0.8, which is a common threshold for high correlation.
# Multicollinearity: A phenomenon where two or more predictor variables in a model are highly correlated, which can destabilize model estimates (though less of an issue for tree-based models like Random Forest). Heatmap: A graphical representation of data where values in a matrix are represented by colors.
# Ref.: Scipy Documentation: Pearson Correlation, Seaborn Heatmap Documentation
# DataCamp: Introduction to Data Visualization with Seaborn, Exploratory Data Analysis in Python & LinkedIn: Data Science & Analytics: Data Visualization

# Correlation matrix
corr_matrix = X_all.corr()
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
# Identify highly correlated features (threshold |r| > 0.8)
high_corr = []
for i in range(len(corr_matrix.columns)):           # Sam: minor (discard) comment should be len(corr_matrix.columns)-1
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))
if high_corr:
    print("\nHighly correlated features (|r| > 0.8):")
    for feat1, feat2, corr in high_corr:
        print(f"  {feat1} <-> {feat2}: {corr:.3f}")
else:
    print("\nNo highly correlated features found (|r| > 0.8)")
""" # Exploratory Data Analysis (EDA) - Correlation - End """

"""## Rolling Origin Evaluation"""
""" Rolling Origin Evaluation (ROE) Configuration - Start """
# Sets up the time windows for the Rolling Origin Evaluation (ROE). Defines the initial training period end, the start and end of the evaluation period, and the final target month for future prediction.
# Defines key date string variables (INITIAL_TRAIN_END, EVALUATION_START, etc.). Generates the list of months (eval_months) that will serve as the one-month-ahead test set in the ROE loop. ROE involves training a model on an ever-expanding historical window (origin) and testing on the next single time step.
# Time Series Cross-Validation: Techniques designed to respect the chronological order of data, preventing data leakage. Rolling Origin/Forward Chaining: A robust method where the training set grows over time, mimicking real-world prediction where more historical data is available with each passing month.
# Rolling Origin Evaluation (Forecasting) - Towards Data Science, Cross-validation for time series - Scikit-learn documentation
# DataCamp: Forecasting in R/Python, Time Series Analysis in Python

# ROE config
#Here, we define the ROE params, we can decide to shift the dates as needed, but just make sure you avoid data leakage (especially since we will tune the hyperparameter- see below cell).
INITIAL_TRAIN_END = '2022-08'  # Initial training period ends # Sam: better to define train size as a number (home many month)
EVALUATION_START = '2022-09'   # Start making predictions from this month # Sam: will always be INITIAL_TRAIN_END + 1
EVALUATION_END = '2025-09'     # Last observed month # Sam: will always be last period in the dataset
FUTURE_PREDICT = '2025-10'     # Future month to predict # Sam: will always be EVALUATION_END + 1
# Get list of all months to predict in ROE
all_months = sorted(df_clean['year_month'].unique())
eval_months = [m for m in all_months if m >= EVALUATION_START and m <= EVALUATION_END]
print("Rolling Origin Evaluation Configuration:")
print(f"  Initial training period: {df_clean['year_month'].min()} to {INITIAL_TRAIN_END}")
print(f"  Evaluation period: {EVALUATION_START} to {EVALUATION_END}")
print(f"  Number of rolling predictions: {len(eval_months)}")
print(f"  Future prediction target: {FUTURE_PREDICT}")
print(f"  First 5 prediction months: {eval_months[:5]}")
print(f"  Last 5 prediction months: {eval_months[-5:]}")
""" Rolling Origin Evaluation (ROE) Configuration - End """

"""## Hyperparameter Tuning
Here, we tune hyperparameters on a holdout period (2021-2022) before running ROE evaluation.
"""

""" Hyperparameter Tuning Data Split - Start """
# Splits the cleaned data into a dedicated Training and Validation set for hyperparameter tuning. A specific historical period (TUNE_TRAIN_END and TUNE_VAL_START to TUNE_VAL_END) is reserved for this purpose, preventing data leakage into the main ROE set.
# Creates two subsets of the cleaned DataFrame (df_clean) based on the year_month column. Separates the predictor features (X) and the target variable (y) for both the tuning train and tuning validation sets. This is a time-based hold-out split.
# Hold-out Validation: Reserving a portion of the data (often the most recent) for evaluation to assess generalizability. Data Leakage Prevention: Ensuring information from the validation or test set does not influence the training or tuning process.
# Model Validation and Evaluation - Google Developers, Time-Series Holdout Sets - O'Reilly
# DataCamp: Model Validation in Python, Machine Learning for Time Series Data in Python

# Validation set for hyperparameter tuning
TUNE_TRAIN_END = '2021-12'
TUNE_VAL_START = '2022-01'
TUNE_VAL_END = '2022-12'
tune_train_data = df_clean[df_clean['year_month'] <= TUNE_TRAIN_END]
tune_val_data = df_clean[(df_clean['year_month'] >= TUNE_VAL_START) &
                          (df_clean['year_month'] <= TUNE_VAL_END)]
X_tune_train = tune_train_data[PREDICTOR_FEATURES]
y_tune_train = tune_train_data[TARGET]
X_tune_val = tune_val_data[PREDICTOR_FEATURES]
y_tune_val = tune_val_data[TARGET]
print("Hyperparameter Tuning Data Split:")
print(f"  Tuning train set: {len(tune_train_data):,} observations ({tune_train_data['year_month'].min()} to {TUNE_TRAIN_END})")
print(f"  Tuning validation set: {len(tune_val_data):,} observations ({TUNE_VAL_START} to {TUNE_VAL_END})")
print(f"  Train fire rate: {y_tune_train.mean():.2%}")
print(f"  Validation fire rate: {y_tune_val.mean():.2%}")
""" Hyperparameter Tuning Data Split - End """

"""### Randomized Search with Time Series Cross-Validation"""
# Parameter distributions for randomized search
param_distributions = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [10, 15, 20, 25, 30, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', 0.5, 0.7]
}
print("Parameter distributions for randomized search:")
for param, values in param_distributions.items():
    print(f"  {param}: {values}")
# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
print(f"Performing randomized search with {tscv.n_splits} CV folds...")
print("Testing 30 random combinations (may take some time)...")
rf_random = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight='balanced',   # Handles class imbalance
    n_jobs=-1
)
random_search = RandomizedSearchCV(
    estimator=rf_random,
    param_distributions=param_distributions,
    n_iter=30,  # Number of random combinations to try
    cv=tscv,
    scoring='f1',
    n_jobs=-1,
    verbose=2,
    random_state=RANDOM_STATE,
    return_train_score=True
)
random_search.fit(X_tune_train, y_tune_train)
# Extract best parameters
BEST_PARAMS = random_search.best_params_
BEST_PARAMS['class_weight'] = 'balanced'
BEST_PARAMS['random_state'] = RANDOM_STATE
BEST_PARAMS['n_jobs'] = -1
print("Randomized search complete!")
print(f"Best parameters found:")
for param, value in BEST_PARAMS.items():
    print(f"  {param}: {value}")
print(f"Best CV F1-score: {random_search.best_score_:.4f}")
"""### Validate Best Parameters on Hold-out Set"""
# Train model with best parameters on tuning train set
print("Validating best parameters on hold-out validation set (2022)...")
rf_best_tuned = RandomForestClassifier(**BEST_PARAMS, verbose=0)
rf_best_tuned.fit(X_tune_train, y_tune_train)
# Predict on validation set
y_val_pred = rf_best_tuned.predict(X_tune_val)
y_val_pred_proba = rf_best_tuned.predict_proba(X_tune_val)[:, 1]
# Evaluate
val_accuracy = accuracy_score(y_tune_val, y_val_pred)
val_balanced_accuracy = balanced_accuracy_score(y_tune_val, y_val_pred)
val_precision = precision_score(y_tune_val, y_val_pred)
val_recall = recall_score(y_tune_val, y_val_pred)
val_f1 = f1_score(y_tune_val, y_val_pred)
val_roc_auc = roc_auc_score(y_tune_val, y_val_pred_proba)
print("Validation Set Performance (2020):")
print("="*80)
print(f"  Accuracy: {val_accuracy:.4f}")
print(f"  Balanced Accuracy: {val_balanced_accuracy:.4f}")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall: {val_recall:.4f}")
print(f"  F1-Score: {val_f1:.4f}")
print(f"  ROC-AUC: {val_roc_auc:.4f}")
print("="*80)
print("HYPERPARAMETER TUNING COMPLETE")
print("="*80)
print(f"So, now we can use these hyperparameters in the ROE evaluation.")
"""## Rolling Origin Evaluation Loop
Now we use the tuned hyperparameters for the rolling origin evaluation.
"""
# Init storage for results
roe_results = []
roe_predictions = []
print("Starting Rolling Origin Evaluation...")
print(f"This will train {len(eval_months)} models (one for each prediction month)")
print("Warning- this may take several minutes...\n")
# Progress bar
for target_month in tqdm(eval_months, desc="ROE Progress"):
    # Training data: all data before target_month
    train_data = df_clean[df_clean['year_month'] < target_month]
    # Test data: only the target_month
    test_data = df_clean[df_clean['year_month'] == target_month]
    # Skip if no test data
    if len(test_data) == 0:
        continue
    # Prepare features
    X_train = train_data[PREDICTOR_FEATURES]
    y_train = train_data[TARGET]
    X_test = test_data[PREDICTOR_FEATURES]
    y_test = test_data[TARGET]
    # Train model with tuned hyperparameters
    model = RandomForestClassifier(**BEST_PARAMS, verbose=0)
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # Calculate metrics
    metrics = {
        'target_month': target_month,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'train_start': train_data['year_month'].min(),
        'train_end': train_data['year_month'].max(),
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(y_test.unique()) > 1 else np.nan
    }
    roe_results.append(metrics)
    # Store predictions
    pred_df = test_data[['cell_id', 'year_month']].copy()
    pred_df['actual'] = y_test.values
    pred_df['predicted'] = y_pred
    pred_df['predicted_proba'] = y_pred_proba
    roe_predictions.append(pred_df)
print("\nROE Complete!")
print(f"Total predictions made: {len(roe_results)}")
"""## ROE Results Analysis"""
# Convert results to DataFrame
roe_metrics_df = pd.DataFrame(roe_results)
roe_metrics_df['target_month_dt'] = pd.to_datetime(roe_metrics_df['target_month'])
print("Rolling Origin Evaluation Results Summary:")
print("="*80)
print(f"\nOverall Performance Across All {len(roe_metrics_df)} Predictions:")
print(f"  Mean Accuracy: {roe_metrics_df['accuracy'].mean():.4f} ± {roe_metrics_df['accuracy'].std():.4f}")
print(f"  Mean Balanced Accuracy: {roe_metrics_df['balanced_accuracy'].mean():.4f} ± {roe_metrics_df['balanced_accuracy'].std():.4f}")
print(f"  Mean Precision: {roe_metrics_df['precision'].mean():.4f} ± {roe_metrics_df['precision'].std():.4f}")
print(f"  Mean Recall: {roe_metrics_df['recall'].mean():.4f} ± {roe_metrics_df['recall'].std():.4f}")
print(f"  Mean F1-Score: {roe_metrics_df['f1_score'].mean():.4f} ± {roe_metrics_df['f1_score'].std():.4f}")
print(f"  Mean ROC-AUC: {roe_metrics_df['roc_auc'].mean():.4f} ± {roe_metrics_df['roc_auc'].std():.4f}")
print(f"\nBest Performing Month:")
best_month = roe_metrics_df.loc[roe_metrics_df['f1_score'].idxmax()]
print(f"  Month: {best_month['target_month']}")
print(f"  F1-Score: {best_month['f1_score']:.4f}")
print(f"\nWorst Performing Month:")
worst_month = roe_metrics_df.loc[roe_metrics_df['f1_score'].idxmin()]
print(f"  Month: {worst_month['target_month']}")
print(f"  F1-Score: {worst_month['f1_score']:.4f}")
print("\nFirst 10 predictions:")
print(roe_metrics_df[['target_month', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].head(10))