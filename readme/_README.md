# CS412 -- BigQuery Geotab Intersection Congestion

### Combined Code Documentation & Instructions

------------------------------------------------------------------------

## 1. Project Structure

**Files:**

-   **Preprocessing_baselineLR_AblationLSTM.ipynb** Includes full data
    preprocessing pipeline, Baseline Linear Regression model, and the
    complete Ablation Study for the LSTM method.

-   **GCN.py** Implements a second baseline model using a Graph
    Convolutional Network (GCN).

-   **Initial_LSTM.ipynb** Contains the initial implementation of our
    custom LSTM model.

-   **Gating_Rule_LSTM.ipynb** Includes an improved version of our LSTM
    method using probability gating.

-   **xgboost.ipynb** Implements our XGBoost-based method.

------------------------------------------------------------------------

## 2. Required Packages

### Core Python Packages
    pip install numpy pandas scikit-learn matplotlib seaborn tqdm joblib

### XGBoost
    pip install xgboost

### PyTorch & PyTorch Geometric (required for GCN)
Install appropriate version:
    pip install torch torchvision torchaudio
    pip install torch-geometric

------------------------------------------------------------------------

## 3. Data Sources

**BigQuery Geotab Intersection Congestion**
https://www.kaggle.com/competitions/bigquery-geotab-intersection-congestion

Required files: 
    - train.csv 
    - test.csv

Example path setup:

    train_csv_path = "path/to/train.csv"
    test_csv_path  = "path/to/test.csv"

------------------------------------------------------------------------

## 4. How to Run the Code

### 4.1 Preprocessing + Baseline LR + Ablation LSTM
File: Preprocessing_baselineLR_AblationLSTM.ipynb 
    - Open in Jupyter or Colab. 
    - Set correct dataset paths.
    - Run all cells.

### 4.2 GCN
File: GCN.py 
    - Ensure PyTorch + PyTorch Geometric are installed. 
    - Update path to train.csv if needed. 
    - Run: python GCN.py

### 4.3 Initial LSTM
File: Initial_LSTM.ipynb 
    - Open in Jupyter or Colab. 
    - Set correct dataset paths.
    - Run all cells.

### 4.4 Gating Rule LSTM (Improved Method)
File: Gating_Rule_LSTM.ipynb 
    - Open in Jupyter or Colab. 
    - Set correct dataset paths.
    - Run all cells.

### 4.5 XGBoost
File: xgboost.ipynb
    - Open in Jupyter or Colab. 
    - Set correct dataset paths.
    - Run all cells.
