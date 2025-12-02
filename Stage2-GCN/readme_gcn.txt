This code predicts traffic stop times and distances at intersections using a Graph Convolutional Network (GCN).

Dependencies
------------
- Python 3.8+
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- scikit-learn
- tqdm

Data
----
Input file: train.csv
Columns:
- IntersectionId, EntryHeading, ExitHeading
- City, Hour, Month, Weekend
- Latitude, Longitude
- Targets: TotalTimeStopped_p20/p50/p80, DistanceToFirstStop_p20/p50/p80

Steps
-----
1. Preprocess data:
   - Encode headings, one-hot encode City
   - Convert Hour and Month to cyclic features
2. Build a graph using coordinates (radius ~200m)
3. Split data into train/validation
4. Train a 2-layer GCN
5. Evaluate using RMSE per target and global RMSE
6. Predict for all nodes

Usage
-----
Run the main script:
python gcn_traffic.py

Notes
-----
- GPU recommended
- NeighborLoader samples neighbors efficiently for training