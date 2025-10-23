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
# Ref.: Pandas Documentation: dropna, Handling Missing Data - Towards Data Science
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
# Ref.: Rolling Origin Evaluation (Forecasting) - Towards Data Science, Cross-validation for time series - Scikit-learn documentation
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
# Ref.: Model Validation and Evaluation - Google Developers, Time-Series Holdout Sets - O'Reilly
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
""" Randomized Search with Time Series CV - Start """
# Performs hyperparameter optimization using Randomized Search combined with Time Series Cross-Validation (TSCV). This efficiently searches the hyperparameter space for the Random Forest Classifier.
# Defines a param_distributions dictionary for the parameters to search. Uses Scikit-learn's TimeSeriesSplit with 5 splits for cross-validation within the tuning train set. RandomizedSearchCV fits multiple models with random parameter combinations, evaluating performance (scoring='f1') to find the best set of parameters. class_weight='balanced' addresses class imbalance.
# Hyperparameter Tuning: The process of choosing a set of optimal hyperparameters for a learning algorithm. Randomized Search: Efficiently explores a large hyperparameter space by sampling combinations. Random Forest Classifier: An ensemble learning method based on decision trees, popular for its robustness and handling of complex data. F1-Score: The harmonic mean of precision and recall, a robust metric for imbalanced classification.
# Ref.: Scikit-learn: RandomizedSearchCV, Scikit-learn: TimeSeriesSplit, Random Forest Algorithm - Explained.
# DataCamp: Hyperparameter Tuning in Python, Supervised Learning with Scikit-learn & LinkedIn: Machine Learning Essential Training: Model Validation and Evaluation
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
""" Randomized Search with Time Series CV - End """

"""### Validate Best Parameters on Hold-out Set"""
""" Validation of Best Parameters - Start """
# Evaluates the model with the best-found hyperparameters on the dedicated hold-out validation set (2022 data).
# Initializes a RandomForestClassifier using the BEST_PARAMS. Trains this model on the tuning training data and predicts on the tuning validation data. Calculates and prints various classification metrics (Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, ROC-AUC) to confirm the model's performance on unseen data before the final ROE.
# Initializes a RandomForestClassifier using the BEST_PARAMS. Trains this model on the tuning training data and predicts on the tuning validation data. Calculates and prints various classification metrics (Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, ROC-AUC) to confirm the model's performance on unseen data before the final ROE.
# Evaluation Metrics: Metrics like Balanced Accuracy (important for imbalance), ROC-AUC (Area Under the Receiver Operating Characteristic Curve), and F1-Score provide a comprehensive view of the classifier's performance, particularly its ability to balance Precision (correct positive predictions) and Recall (finding all positive cases).
# Ref.: Scikit-learn: Classification Metrics, Understanding ROC AUC - Towards Data Science
# DataCamp: Machine Learning for Everyone, Supervised Learning with Scikit-learn
# Train model with best parameters on tuning train set
print("Validating best parameters on hold-out validation set (2022)...")
rf_best_tuned = RandomForestClassifier(**BEST_PARAMS, verbose=0)  # Sam : more clarity required for the ** in the **BEST_PARAMS (first parameter in the RandomForestClassifier function)
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
""" Validation of Best Parameters - End """

"""## Rolling Origin Evaluation Loop
Now we use the tuned hyperparameters for the rolling origin evaluation.
"""
""" Rolling Origin Evaluation Loop - Start """
# Executes the main ROE process: The model is retrained monthly on all data preceding the target month, and predictions are made for that target month.
# Iterates through eval_months. In each loop: 1. Splits the data into expanding train set (< target_month) and one-month test set (== target_month). 2. Trains a RandomForestClassifier with the globally optimal BEST_PARAMS. 3. Predicts on the test set. 4. Calculates and stores performance metrics and predictions for that month. Uses tqdm for a progress bar.
# Walk-Forward Validation: The core technique of ROE, simulating how a model would be used in production over time. Model Re-training: The practice of periodically re-training a model on all available data to capture the latest trends and data patterns.
# Ref.: Walk-Forward Optimization - Investopedia, Time Series Forecasting (Chapter 5) - Forecasting: Principles and Practice
# DataCamp: Applied Time Series Analysis in Python, Model Validation in Python - LinkedIn: Advanced Machine Learning: Time Series
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
        'train_start': train_data['year_month'].min(),   # Sam: comparing strings not numbers - would like to confirm it will be working as expected
        'train_end': train_data['year_month'].max(),  # Sam: same as above comment
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
""" Rolling Origin Evaluation Loop - End """

"""## ROE Results Analysis"""
""" ROE Results Analysis - Start """ 
# Aggregates and summarizes the performance metrics from the Rolling Origin Evaluation (ROE).
# Converts the list of monthly results (roe_results) into a DataFrame for easy analysis. Calculates the mean and standard deviation (± std) for all key metrics (F1-Score, ROC-AUC, etc.) across all prediction months. Identifies the best and worst-performing months based on F1-Score.
# Converts the list of monthly results (roe_results) into a DataFrame for easy analysis. Calculates the mean and standard deviation (± std) for all key metrics (F1-Score, ROC-AUC, etc.) across all prediction months. Identifies the best and worst-performing months based on F1-Score.
# Model Generalization (Across Time): Assessing how well the model's performance holds up over a long sequence of future prediction periods. Standard Deviation: Measures the monthly variability (consistency) of the model's performance.
# Ref.: Statistical Analysis with Pandas - Documentation, Interpreting Standard Deviation - Statistics LibreTexts
# DataCamp: Statistical Thinking in Python (Part 1 & 2), Data Analysis in Python

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
""" ROE Results Analysis - End """ 

"""## Visualize ROE Performance Over Time"""
""" Visualize ROE Performance - Start """

# Uses matplotlib.pyplot to create a 3×2 grid of subplots (fig, axes = plt.subplots(3, 2)). Each plot uses ax.plot() to show a metric from the roe_metrics_df against the target_month_dt. A horizontal line (ax.axhline) is added to show the mean for context. The last subplot uses a Box Plot (ax.boxplot) to summarize the distribution of all metrics across all time windows.
# Visualizes the model's performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) and training set size over time. This checks for model stability and data drift.
# Rolling Origin Evaluation (ROE) / Time Series Cross-Validation: A robust method for evaluating models on time-dependent data. Performance Metrics (Classification): Accuracy, Precision, Recall, F1-Score, ROC-AUC. Data Visualization: Matplotlib, Box Plot interpretation.
# Ref.: Matplotlib Documentation, Scikit-learn Model Evaluation, Time Series Model Evaluation
# DataCamp: Time Series Analysis in Python, Data Visualization with Matplotlib and Seaborn
fig, axes = plt.subplots(3, 2, figsize=(18, 14))

# Accuracy over time
ax1 = axes[0, 0]
ax1.plot(roe_metrics_df['target_month_dt'], roe_metrics_df['accuracy'],
         marker='o', linewidth=2, markersize=4, color='#3498db', label='Accuracy')
ax1.axhline(y=roe_metrics_df['accuracy'].mean(), color='red', linestyle='--',
           linewidth=2, label=f"Mean = {roe_metrics_df['accuracy'].mean():.3f}")
ax1.set_xlabel('Prediction Month', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Over Time (ROE)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Precision and Recall
ax2 = axes[0, 1]
ax2.plot(roe_metrics_df['target_month_dt'], roe_metrics_df['precision'],
         marker='s', linewidth=2, markersize=4, color='#2ecc71', label='Precision')
ax2.plot(roe_metrics_df['target_month_dt'], roe_metrics_df['recall'],
         marker='^', linewidth=2, markersize=4, color='#e74c3c', label='Recall')
ax2.set_xlabel('Prediction Month', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Precision vs Recall Over Time (ROE)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# F1-Score
ax3 = axes[1, 0]
ax3.plot(roe_metrics_df['target_month_dt'], roe_metrics_df['f1_score'],
         marker='d', linewidth=2, markersize=4, color='#9b59b6')
ax3.axhline(y=roe_metrics_df['f1_score'].mean(), color='red', linestyle='--',
           linewidth=2, label=f"Mean = {roe_metrics_df['f1_score'].mean():.3f}")
ax3.set_xlabel('Prediction Month', fontsize=12)
ax3.set_ylabel('F1-Score', fontsize=12)
ax3.set_title('F1-Score Over Time (ROE)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ROC-AUC
ax4 = axes[1, 1]
ax4.plot(roe_metrics_df['target_month_dt'], roe_metrics_df['roc_auc'],
         marker='o', linewidth=2, markersize=4, color='#f39c12')
ax4.axhline(y=roe_metrics_df['roc_auc'].mean(), color='red', linestyle='--',
           linewidth=2, label=f"Mean = {roe_metrics_df['roc_auc'].mean():.3f}")
ax4.set_xlabel('Prediction Month', fontsize=12)
ax4.set_ylabel('ROC-AUC', fontsize=12)
ax4.set_title('ROC-AUC Over Time (ROE)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Training set size
ax5 = axes[2, 0]
ax5.plot(roe_metrics_df['target_month_dt'], roe_metrics_df['train_size'],
         marker='o', linewidth=2, markersize=4, color='#16a085')
ax5.set_xlabel('Prediction Month', fontsize=12)
ax5.set_ylabel('Training Set Size', fontsize=12)
ax5.set_title('Expanding Training Window Size (ROE)', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Performance distribution
ax6 = axes[2, 1]
metrics_to_plot = roe_metrics_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
bp = ax6.boxplot([metrics_to_plot[col].dropna() for col in metrics_to_plot.columns],
                  labels=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                  patch_artist=True)
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_ylabel('Score', fontsize=12)
ax6.set_title('Performance Metrics Distribution (ROE)', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
""" Visualize ROE Performance - End """

"""## Predict October 2025"""
""" Predict October 2025: Feature Creation - Start """
# Creates the feature matrix for the target month (October 2025) by shifting lag and spatial features from the previous month (September 2025).
# It copies the September 2025 data (sept_2025) to form the October 2025 frame. It then updates all date features (month, year, sin/cos components). Crucially, it shifts the lag features: e.g., fires_lag_1m for Oct uses number_of_fires from Sept. This process imputes the required features based on the last known state.
# Feature Engineering for Time Series: Lag features (Autoregressive component), Trigonometric encoding of time (Seasonality), Spatial lag features. Data Preparation for Forecasting.
# Ref.: Feature Engineering for Time Series Forecasting, Pandas Time Series Functionality
# LinkedIn: Feature Engineering for Machine Learning, Machine Learning and AI Foundations: Predictive Modeling

print("="*80)
print("PREDICTING OCTOBER 2025 (Next Month FORECAST)")
print("="*80)

# We can use September 2025 data to create October features
sept_2025 = df_clean[df_clean['year_month'] == '2025-09'].copy()

print(f"\nFound {len(sept_2025)} cells with September 2025 data")
print("Creating October 2025 features...")

# Sam 1 - below code : why copying data from last column (Sep-2025) to the new column (Oct-2025) to be predicted ? :
# Sam 2 - below code : hard-coded -> fix is to have code that will DYNAMICALLY prepare/generate coming month data, if that is what is required
# October 2025 data                                         
oct_2025 = sept_2025.copy()
oct_2025['year_month'] = '2025-10' 
oct_2025['year_month_dt'] = pd.to_datetime('2025-10-01')
oct_2025['month'] = 10
oct_2025['year'] = 2025
oct_2025['month_sin'] = np.sin(2 * np.pi * 10 / 12)
oct_2025['month_cos'] = np.cos(2 * np.pi * 10 / 12)

# Update lag features (shift by 1 month)
oct_2025['fires_lag_1m'] = sept_2025['number_of_fires'].values
oct_2025['fires_lag_2m'] = sept_2025['fires_lag_1m'].values
oct_2025['fires_lag_3m'] = sept_2025['fires_lag_2m'].values
oct_2025['fires_lag_6m'] = sept_2025['fires_lag_3m'].values  # Approximation
oct_2025['fires_lag_12m'] = sept_2025['fires_lag_6m'].values  # Approximation

oct_2025['target_lag_1m'] = sept_2025['Target_Variable'].values
oct_2025['target_lag_2m'] = sept_2025['target_lag_1m'].values
oct_2025['target_lag_3m'] = sept_2025['target_lag_2m'].values

# Spatial lags (using September's neighbor fires)
oct_2025['neighbors_fires_lag1m'] = sept_2025['number_of_fires'].values
oct_2025['neighbors_target_lag1m'] = sept_2025['Target_Variable'].values

print("\nOctober 2025 features created successfully")
print(f"October 2025 observations: {len(oct_2025)}")

# Train final model on ALL data up to September 2025
print("\nTraining final model on all data up to September 2025...")
""" Predict October 2025: Feature Creation - End """

""" Predict October 2025: Final Training & Prediction - Start """
# Trains the final machine learning model using all available historical data and applies it to the newly created October 2025 features to generate predictions.
# Final Training: Filters df_clean up to September 2025, separates features (X) and target (y), and trains a RandomForestClassifier using BEST_PARAMS (hyperparameters found during the ROE process). Prediction: Uses final_model.predict() for the binary risk class and final_model.predict_proba()[:, 1] for the probability score on the October 2025 feature set (X_oct_2025). Risk Binning: Uses pd.cut() to convert the continuous probability into categorical risk levels (Low, Medium, High, Very High).
# Random Forest Classifier: An ensemble learning method (Bagging of Decision Trees). Model Training/Deployment Pipeline: The process of fitting a final model on complete data before using it for forecasting. Hyperparameter Optimization: BEST_PARAMS is the result of a previous tuning step.
# Ref.: Scikit-learn Random Forest Documentation, Cross-Validation for Hyperparameter Tuning
# DataCamp: Supervised Learning with Scikit-learn, Machine Learning with Tree-Based Models in Python
train_final = df_clean[df_clean['year_month'] <= '2025-09']
X_train_final = train_final[PREDICTOR_FEATURES]
y_train_final = train_final[TARGET]

print(f"Final training set: {len(train_final):,} observations")
print(f"Date range: {train_final['year_month'].min()} to {train_final['year_month'].max()}")

# Train model with tuned hyperparameters
final_model = RandomForestClassifier(**BEST_PARAMS, verbose=1)
final_model.fit(X_train_final, y_train_final)

print("\nFinal model trained successfully")

# Make predictions for October 2025
print("\nGenerating predictions for October 2025...")

X_oct_2025 = oct_2025[PREDICTOR_FEATURES]  # Sam: as a continuity of above comment on why Sep-2025 is copied to Oct-2025 data before prediction, this comment is to highlight a possible data leakege here - clarification required, please ?
oct_2025_pred = final_model.predict(X_oct_2025)
oct_2025_pred_proba = final_model.predict_proba(X_oct_2025)[:, 1]

# Create predictions DataFrame
oct_2025_predictions = oct_2025[['cell_id', 'year_month']].copy()
oct_2025_predictions['predicted_fire_risk'] = oct_2025_pred
oct_2025_predictions['fire_probability'] = oct_2025_pred_proba
oct_2025_predictions['risk_level'] = pd.cut(
    oct_2025_pred_proba,
    bins=[0, 0.3, 0.6, 0.8, 1.0],
    labels=['Low', 'Medium', 'High', 'Very High']
)

print("\nOctober 2025 Predictions Summary:")
print("="*80)
print(f"Total grid cells predicted: {len(oct_2025_predictions)}")
print(f"\nPredicted fire occurrences:")
print(oct_2025_predictions['predicted_fire_risk'].value_counts())
print(f"\nRisk level distribution:")
print(oct_2025_predictions['risk_level'].value_counts())

print("\nTop 10 highest risk cells for October 2025:")   # Sam: How this is currently used
top_risk = oct_2025_predictions.nlargest(10, 'fire_probability')
print(top_risk.to_string(index=False))

print("\nAll October 2025 predictions (sorted by risk):")
print(oct_2025_predictions.sort_values('fire_probability', ascending=False).to_string(index=False))
""" Predict October 2025: Final Training & Prediction - End """

"""## Visualize October 2025 Predictions"""
""" Visualize October 2025 Predictions - Stat """
# Provides a non-spatial visualization of the October 2025 forecast: a histogram of predicted probabilities and a bar chart of the categorical risk level distribution.
# Histogram: Uses ax1.hist() to show the count of grid cells vs. their predicted fire probability, with a vertical line at the default classification threshold of 0.5. Bar Chart: Uses ax2.bar() to show the count of cells falling into each risk category (Low, Medium, High, Very High). This visualization gives an overall view of the forecasted risk magnitude.
# Classification Threshold: The point (usually 0.5) where probability is converted to a binary class (fire/no fire). Data Distribution Analysis. Exploratory Data Analysis (EDA).
# Ref.: Matplotlib Histogram Guide, Probability Calibration in ML
# DataCamp: Introduction to Data Visualization with Matplotlib, Statistical Thinking in Python (Part 1)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Probability distribution
ax1 = axes[0]
ax1.hist(oct_2025_predictions['fire_probability'], bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
ax1.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
ax1.set_xlabel('Fire Probability', fontsize=12)
ax1.set_ylabel('Number of Grid Cells', fontsize=12)
ax1.set_title('October 2025: Fire Risk Probability Distribution', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Risk level bar chart
ax2 = axes[1]
risk_counts = oct_2025_predictions['risk_level'].value_counts().sort_index()
colors_risk = ['#2ecc71', '#f39c12', '#e67e22', '#c0392b']
bars = ax2.bar(range(len(risk_counts)), risk_counts.values, color=colors_risk[:len(risk_counts)], alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(risk_counts)))
ax2.set_xticklabels(risk_counts.index, fontsize=11)
ax2.set_ylabel('Number of Grid Cells', fontsize=12)
ax2.set_title('October 2025: Risk Level Distribution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(risk_counts.values):
    ax2.text(i, v + 0.3, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
""" Visualize October 2025 Predictions - End """

"""### Visualize spatial risk patterns"""
""" Visualize spatial risk patterns - Start """
# Generates a geographical map of the October 2025 predictions, showing the location of fire risk probability (continuous) and categorical risk level (discrete). This is the actionable output of the entire process.
# Spatial Data Handling: Uses geopandas (gpd) to read administrative boundaries and create a regular 2.5km grid (CELL_SIZE = 2500) using Shapely's box function. Spatial Join: Merges the predicted probabilities (oct_2025_predictions) with the grid geometry (grid_gdf) based on the cell_id. Mapping: Uses grid_with_predictions.plot(): The first map uses a sequential color map (cmap='YlOrRd') for continuous probability. The second map plots subsets of the grid using specific, distinct colors (risk_colors) for the categorical risk levels, providing a clear visual risk prioritization.
# Geographic Information Systems (GIS): Geopandas, Shapely, Coordinate Reference Systems (CRS - EPSG:32188). Choropleth Mapping: A thematic map where areas are shaded in proportion to a statistical variable. Spatial Modeling/Gridding.
# Ref.: Geopandas Documentation, Shapely Documentation, Cartography and Map Projections
# LinkedIn: Learning GIS, Geospatial Analysis with Python
# Recreate the grid geometry to visualize spatial risk patterns (or we can save the old grid and reuse them here)
print("Creating spatial grid for visualization...")

# Admin boundaries for Montreal
#admin_bounds = gpd.read_file('Data/EKUE_ASSESSMENTS_DATA/limites-administratives-agglomeration-nad83.geojson')
admin_bounds = gpd.read_file('limites-administratives-agglomeration-nad83 (2).geojson')

admin_bounds['geometry'] = admin_bounds.buffer(0)
admin_bounds = admin_bounds.to_crs(epsg=32188)
boundary_union = unary_union(admin_bounds.geometry)

# Recreate 5km grid (This needs to match whatever we used in preparing data). So, please change the size here to 2.5.
# I used 5km grid just for the POC to reduce computational time.
CELL_SIZE = 2500 #5000  # 5km                   # Sam: consider 2 km grid cell
BUFFER_SIZE = 500  # 500m buffer
poly = boundary_union.buffer(BUFFER_SIZE)
minx, miny, maxx, maxy = poly.bounds

n_cols = int(np.ceil((maxx - minx) / CELL_SIZE))
n_rows = int(np.ceil((maxy - miny) / CELL_SIZE))

# Create grid cells
cells = []
for r in range(n_rows):
    y1 = miny + r * CELL_SIZE
    y2 = y1 + CELL_SIZE
    for c in range(n_cols):
        x1 = minx + c * CELL_SIZE
        x2 = x1 + CELL_SIZE
        cell = box(x1, y1, x2, y2)
        if cell.intersects(poly):
            cell_id = f"{r}_{c}"
            cells.append({'cell_id': cell_id, 'geometry': cell.intersection(poly)})

# GeoDataFrame
grid_gdf = gpd.GeoDataFrame(cells, crs="EPSG:32188")

# Merge with October 2025 predictions
grid_with_predictions = grid_gdf.merge(
    oct_2025_predictions[['cell_id', 'fire_probability', 'risk_level', 'predicted_fire_risk']],
    on='cell_id',
    how='left'
)

print(f"Grid cells created: {len(grid_gdf)}")
print(f"Cells with predictions: {grid_with_predictions['fire_probability'].notna().sum()}")
print(f"Cells without predictions: {grid_with_predictions['fire_probability'].isna().sum()}")

# Spatial map visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Fire Probability
ax1 = axes[0]
grid_with_predictions.plot(
    column='fire_probability',
    ax=ax1,
    cmap='YlOrRd',
    legend=True,
    edgecolor='black',
    linewidth=0.5,
    missing_kwds={'color': 'lightgrey', 'edgecolor': 'black', 'label': 'No data'},
    legend_kwds={'label': 'Fire Probability', 'orientation': 'horizontal', 'shrink': 0.8, 'pad': 0.05}
)
admin_bounds.boundary.plot(ax=ax1, edgecolor='navy', linewidth=2, alpha=0.7)
ax1.set_title('October 2025: Fire Risk Probability Map (Continuous Scale)',
             fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Easting (m)', fontsize=12)
ax1.set_ylabel('Northing (m)', fontsize=12)
ax1.set_aspect('equal')
ax1.ticklabel_format(style='plain')

# Add text annotations for top 5 risk cells
top_5_cells = oct_2025_predictions.nlargest(5, 'fire_probability')
for idx, row in top_5_cells.iterrows():
    cell_geom = grid_with_predictions[grid_with_predictions['cell_id'] == row['cell_id']].geometry.values
    if len(cell_geom) > 0:
        centroid = cell_geom[0].centroid
        ax1.annotate(
            f"{row['fire_probability']:.2f}",
            xy=(centroid.x, centroid.y),
            fontsize=9,
            ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='black')
        )

# Risk Levels
ax2 = axes[1]

# Define color map for risk levels
risk_colors = {
    'Low': '#2ecc71',       # Green
    'Medium': '#f39c12',    # Orange
    'High': '#e67e22',      # Dark Orange
    'Very High': '#c0392b'  # Red
}

# Plot each risk level separately
for risk_level in ['Low', 'Medium', 'High', 'Very High']:
    subset = grid_with_predictions[grid_with_predictions['risk_level'] == risk_level]
    if len(subset) > 0:
        subset.plot(ax=ax2, color=risk_colors[risk_level], edgecolor='black',
                   linewidth=0.5, label=f'{risk_level} ({len(subset)} cells)')

# Plot cells with no data
no_data = grid_with_predictions[grid_with_predictions['risk_level'].isna()]
if len(no_data) > 0:
    no_data.plot(ax=ax2, color='lightgrey', edgecolor='black', linewidth=0.5,
                label=f'No data ({len(no_data)} cells)')

admin_bounds.boundary.plot(ax=ax2, edgecolor='navy', linewidth=2, alpha=0.7, label='Montreal Boundary')
ax2.set_title('October 2025: Fire Risk Level Map (Categorical)',
             fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Easting (m)', fontsize=12)
ax2.set_ylabel('Northing (m)', fontsize=12)
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax2.set_aspect('equal')
ax2.ticklabel_format(style='plain')

plt.tight_layout()
plt.show()

# Summary statistics by risk level
print("\nSpatial Distribution Summary:")
print("="*60)
risk_summary = oct_2025_predictions.groupby('risk_level').agg({
    'cell_id': 'count',
    'fire_probability': ['mean', 'min', 'max']
}).round(3)
risk_summary.columns = ['Count', 'Avg Prob', 'Min Prob', 'Max Prob']
print(risk_summary)

print("\nTop 5 Highest Risk Cells (Spatial):")
top_cells = oct_2025_predictions.nlargest(5, 'fire_probability')[['cell_id', 'fire_probability', 'risk_level']]
print(top_cells.to_string(index=False))
""" Visualize spatial risk patterns - End """

"""## Feature Importance Analysis"""
""" Feature Importance Analysis - Start """
# Determines which input variables (features) the final Random Forest model relies on most to make its predictions. This is crucial for model explainability and domain validation.
# Calculation: Extracts the Gini Importance (also known as Mean Decrease in Impurity, MDI) from the trained final_model.feature_importances_. Visualization: Uses a horizontal bar chart to rank individual features. Features above a 5% threshold are often highlighted. Categorization: Features are grouped (e.g., 'Temporal Lag', 'Spatial Lag') and their importance scores are summed to show the overall impact of each category.
# Feature Importance (Tree-Based Models): Gini Importance/MDI. Model Interpretability (XAI): Understanding model decisions. Domain Knowledge Validation: Checking if the model emphasizes features deemed important by domain experts.
# Ref.: Scikit-learn Feature Importance, Interpretable Machine Learning (Christoph Molnar's Book), LIME and SHAP for advanced interpretability
# DataCamp: Model Validation in Python, Machine Learning Interpretability in Python
# Feature importance from final model
feature_importance = pd.DataFrame({
    'feature': PREDICTOR_FEATURES,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance Rankings (from final Model):")
print(feature_importance.to_string(index=False))

# Plot
plt.figure(figsize=(12, 8))
colors = ['#e74c3c' if imp > 0.05 else '#3498db' for imp in feature_importance['importance']]
plt.barh(range(len(feature_importance)), feature_importance['importance'], color=colors)
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance - Final Model (Trained on All Data)', fontsize=14, fontweight='bold')
plt.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Feature importance by category
def get_feature_category(feature):
    if feature in BUILDING_FEATURES:
        return 'Building'
    elif feature in TEMPORAL_LAG_FEATURES:
        return 'Temporal Lag'
    elif feature in SEASONALITY_FEATURES:
        return 'Seasonality'
    elif feature in SPATIAL_LAG_FEATURES:
        return 'Spatial Lag'
    return 'Other'

feature_importance['category'] = feature_importance['feature'].apply(get_feature_category)
category_importance = feature_importance.groupby('category')['importance'].sum().sort_values(ascending=False)

print("\nImportance by Feature Category:")
print(category_importance)

# Plot category importance
plt.figure(figsize=(10, 6))
category_importance.plot(kind='bar', color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
plt.title('Feature Importance by Category', fontsize=14, fontweight='bold')
plt.xlabel('Feature Category', fontsize=12)
plt.ylabel('Total Importance', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
""" Feature Importance Analysis - End """

"""## Save Results"""
""" Save Results - Start """
# Generates a clear, professional text report summarizing the entire machine learning project, its evaluation, and the final forecast results. This is the project's official communication piece.
# Reporting: Prints formatted strings using Python's f-strings for clear presentation. Summarization: Calculates and displays key statistical summaries, including the mean and standard deviation (std) of performance metrics across the ROE windows. Model Stability: Calculates the Coefficient of Variation (CV) for the F1-Score (CV=std/mean) to quantitatively assess model stability over time.
# Data Science Communication: Presenting technical results to a business audience. Coefficient of Variation (CV): A standardized measure of dispersion of a probability distribution or frequency distribution. Used here to quantify model volatility. Final Project Documentation.
# Effective Data Storytelling, Statistical Reporting Guidelines
# DataCamp: Communicating Data Science Results, Data Science for Business

# Persists all critical artifacts from the entire project to disk. This ensures reproducibility, auditability, and ease of deployment/sharing.
# Data Saving: Uses Pandas' to_csv() to save the ROE metrics (roe_metrics_df), all historical ROE predictions, the October 2025 forecast, and the feature importance table. Model Saving: Uses the Python standard library pickle to serialize and save the actual trained final_model object to a .pkl file.
# Serialization: The process of converting a data structure or object state into a format that can be stored or transmitted (e.g., pickle). MLOps (Machine Learning Operations): Asset versioning and management. Reproducibility.
# Ref.: Pandas CSV Documentation, Python Pickle Documentation, Introduction to MLOps
# LinkedIn: MLOps Foundations, Learning Python File Handling
# Save ROE metrics
# roe_metrics_path = 'Data/roe_metrics.csv'
roe_metrics_path = 'roe_metrics.csv'

roe_metrics_df.to_csv(roe_metrics_path, index=False)
print(f"ROE metrics saved to: {roe_metrics_path}")

# Save all ROE predictions
all_roe_predictions = pd.concat(roe_predictions, ignore_index=True)

#roe_predictions_path = 'Data/roe_all_predictions.csv'
roe_predictions_path = 'roe_all_predictions.csv'

all_roe_predictions.to_csv(roe_predictions_path, index=False)
print(f"All ROE predictions saved to: {roe_predictions_path}")

# Save October 2025 predictions
#oct_2025_path = 'Data/october_2025_predictions.csv'
oct_2025_path = 'october_2025_predictions.csv'

oct_2025_predictions.to_csv(oct_2025_path, index=False)
print(f"October 2025 predictions saved to: {oct_2025_path}")

# Save final model
#model_path = 'Data/final_rf_model_roe.pkl'
model_path = 'final_rf_model_roe.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"Final model saved to: {model_path}")

# Save feature importance
#feature_importance_path = 'Data/feature_importance_roe.csv'
feature_importance_path = 'feature_importance_roe.csv'

feature_importance.to_csv(feature_importance_path, index=False)
print(f"Feature importance saved to: {feature_importance_path}")

print("\nAll results saved successfully!")
""" Save Results - End """

"""## Summary Report"""
""" Summary Report - Start """

print("="*80)
print("ROLLING ORIGIN EVALUATION - FINAL SUMMARY REPORT")
print("="*80)

print("\n EVALUATION APPROACH:")
print(f"   - Method: Rolling Origin Evaluation (ROE)")
print(f"   - Number of predictions: {len(roe_metrics_df)}")
print(f"   - Evaluation period: {EVALUATION_START} to {EVALUATION_END}")
print(f"   - Initial training window: {df_clean['year_month'].min()} to {INITIAL_TRAIN_END}")

print("\n HYPERPARAMETER TUNING:")
print(f"   - Tuning period: {TUNE_TRAIN_END} (train) + {TUNE_VAL_START} to {TUNE_VAL_END} (val)")
print(f"   - Method: Randomized Search with 5-fold Time Series CV")
print(f"   - Best parameters: {BEST_PARAMS}")

print("\n OVERALL ROE PERFORMANCE:")
print(f"   - Mean Accuracy: {roe_metrics_df['accuracy'].mean():.4f} ± {roe_metrics_df['accuracy'].std():.4f}")
print(f"   - Mean Balanced Accuracy: {roe_metrics_df['balanced_accuracy'].mean():.4f} ± {roe_metrics_df['balanced_accuracy'].std():.4f}")
print(f"   - Mean Precision: {roe_metrics_df['precision'].mean():.4f} ± {roe_metrics_df['precision'].std():.4f}")
print(f"   - Mean Recall: {roe_metrics_df['recall'].mean():.4f} ± {roe_metrics_df['recall'].std():.4f}")
print(f"   - Mean F1-Score: {roe_metrics_df['f1_score'].mean():.4f} ± {roe_metrics_df['f1_score'].std():.4f}")
print(f"   - Mean ROC-AUC: {roe_metrics_df['roc_auc'].mean():.4f} ± {roe_metrics_df['roc_auc'].std():.4f}")


print("\n" + "="*80)
print("ROLLING ORIGIN EVALUATION COMPLETE")
print("="*80)

print("=" *80)
print("\n OCTOBER 2025 FORECAST:")
print(f"   - Grid cells predicted: {len(oct_2025_predictions)}")
print(f"   - Cells predicted with fire: {oct_2025_predictions['predicted_fire_risk'].sum()}")
print(f"   - Average fire probability: {oct_2025_predictions['fire_probability'].mean():.4f}")
print(f"   - Highest risk cell: {oct_2025_predictions.loc[oct_2025_predictions['fire_probability'].idxmax(), 'cell_id']}")
print(f"   - Highest risk probability: {oct_2025_predictions['fire_probability'].max():.4f}")

print("=" *80 )

print("="*80)
print("\n TOP 5 MOST IMPORTANT FEATURES:")
for i, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']:30s} {row['importance']:.4f}")

print("="*80)

print("="*80)
print("\n MODEL STABILITY:")
cv_coef = roe_metrics_df['f1_score'].std() / roe_metrics_df['f1_score'].mean()
print(f"   - F1-Score coefficient of variation: {cv_coef:.4f}")
if cv_coef < 0.1:
    print(f"   - Assessment: VERY STABLE (CV < 0.1)")
elif cv_coef < 0.2:
    print(f"   - Assessment: STABLE (CV < 0.2)")
else:
    print(f"   - Assessment: MODERATE VARIABILITY (CV >= 0.2)")

print("="*80)

print("="*80)
print("\n FILES GENERATED:")
print(f"   - {roe_metrics_path}")
print(f"   - {roe_predictions_path}")
print(f"   - {oct_2025_path}")
print(f"   - {model_path}")
print(f"   - {feature_importance_path}")

print("="*80)
""" Summary Report - End """