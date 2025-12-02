Model Approach
- Model: XGBoost Regressor (XGBRegressor) wrapped in a MultiOutputRegressor to handle 6 targets simultaneously.
- Hyperparameters: n_estimators=1000, learning_rate=0.05, max_depth=10, subsample=0.8.
- Loss Function: RMSE (Root Mean Squared Error).

Feature Engineering
1. Heading Mapping: Converted cardinal directions (N, NE, E, etc.) to numerical categories (1-8).
2. Cyclical Time Features: Transformed Hour and Month into sine and cosine components (hour_sin, hour_cos, etc.) to capture temporal continuity.
3. One-Hot Encoding: Applied to the City column (Atlanta, Boston, Chicago, Philadelphia).
4. Target Scaling: Used StandardScaler to normalize the 6 target variables during training.

File Structure
- train.csv / test.csv: Raw input data.
- model.joblib: The trained XGBoost model serialized for future use.
- scaler.joblib: The StandardScaler object used to inverse transform predictions.
- submission.csv: Raw predictions in wide format (one row per intersection).
- submission_formatted.csv: Reshaped long-format submission (ID, Target) required by the competition.
- submission_fixed.csv: Final submission file aligned with sample_submission.csv (filtered to exclude unused test rows).

Requirements
- Python 3.x
- pandas, numpy
- scikit-learn
- xgboost
- joblib

Usage
1. Load the trained model using joblib.load('model.joblib').
2. Preprocess new data using the same feature engineering steps.
3. Predict and inverse transform using scaler.inverse_transform().
