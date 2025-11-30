# ------------------------------
# 0. 导入库
# ------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, radius_graph
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

# ------------------------------
# 1. GPU 设置
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ------------------------------
# 2. 读取数据
# ------------------------------
train_csv_path = "train.csv"
df = pd.read_csv(train_csv_path)

# ------------------------------
# 3. 特征处理
# ------------------------------
x_train = df[['IntersectionId','EntryHeading','ExitHeading']]
x_train = pd.concat([x_train, pd.get_dummies(df["City"], dummy_na=False, drop_first=False, dtype=int)], axis=1)

heading_map = {'N':1,'NE':2,'E':3,'SE':4,'S':5, 'SW':6, 'W':7, 'NW':8}
x_train['EntryHeading'] = x_train['EntryHeading'].replace(heading_map)
x_train['ExitHeading'] = x_train['ExitHeading'].replace(heading_map)

# 时间特征
df['hour_sin'] = np.sin(2*np.pi*df['Hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['Hour']/24)
df['month_sin'] = np.sin(2*np.pi*df['Month']/12)
df['month_cos'] = np.cos(2*np.pi*df['Month']/12)

x_train['hour_sin'] = df['hour_sin']
x_train['hour_cos'] = df['hour_cos']
x_train['Month'] = df['Month']
x_train['Weekend'] = df['Weekend']
x_train['Latitude'] = df['Latitude']
x_train['Longitude'] = df['Longitude']
x_train['node_id'] = df['IntersectionId'].astype(str) + "_" + df['EntryHeading']

# ------------------------------
# 4. 目标变量
# ------------------------------
y_train = pd.DataFrame()
y_train['TotalTimeStopped_p20'] = df['TotalTimeStopped_p20']
y_train['TotalTimeStopped_p50'] = df['TotalTimeStopped_p50']
y_train['TotalTimeStopped_p80'] = df['TotalTimeStopped_p80']
y_train['DistanceToFirstStop_p20'] = df['DistanceToFirstStop_p20']
y_train['DistanceToFirstStop_p50'] = df['DistanceToFirstStop_p50']
y_train['DistanceToFirstStop_p80'] = df['DistanceToFirstStop_p80']

# ------------------------------
# 5. 高效构图（使用 radius_graph）
# ------------------------------
coords = torch.tensor(df[['Latitude','Longitude']].values, dtype=torch.float)
threshold_m = 200
threshold_degrees = (threshold_m / 1000) / 111.0 # radius_graph 使用坐标单位，若使用经纬度可当作 km

edge_index = radius_graph(coords, r=threshold_degrees, loop=False)  # 高效稀疏图

# ------------------------------
# 6. 转 tensor 并放 GPU
# ------------------------------
feature_cols = [c for c in x_train.columns if c not in ['node_id','IntersectionId']]
x_tensor = torch.tensor(x_train[feature_cols].values, dtype=torch.float)
y_tensor = torch.tensor(y_train.values, dtype=torch.float)

data = Data(x=x_tensor, y=y_tensor, edge_index=edge_index)

# ------------------------------
# 7. 划分训练/验证
# ------------------------------
train_idx, val_idx = train_test_split(range(len(x_tensor)), test_size=0.2, random_state=42)

train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],  # 每层采样 10 个邻居
    input_nodes=train_idx,
    batch_size=64,
    shuffle=True
)

val_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    input_nodes=val_idx,
    batch_size=64,
    shuffle=False
)

# ------------------------------
# 8. 构建 GCN 模型
# ------------------------------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x

model = GCN(in_channels=x_tensor.shape[1], hidden_channels=64, out_channels=y_tensor.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# ------------------------------
# 9. 训练循环 (mini-batch + GPU + tqdm)
# ------------------------------
# ------------------------------
# 9. 训练循环 (修正切片问题)
# ------------------------------
epochs = 10
print("Start Training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # tqdm用于进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index)
        
        # [重要修正] 只计算 batch_size 大小的中心节点的 loss
        # NeighborLoader 输出的前 batch_size 个节点才是当前 batch 的目标节点
        batch_size = batch.batch_size
        out_target = out[:batch_size]
        y_target = batch.y[:batch_size]
        
        loss = criterion(out_target, y_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        pbar.set_postfix({'loss': loss.item()})
        
    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_idx):.4f}")

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            
            # 同样需要切片
            batch_size = batch.batch_size
            val_loss += criterion(out[:batch_size], batch.y[:batch_size]).item() * batch_size
            
    print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_idx):.4f}")

# ------------------------------
# 10. 预测 (修正切片与排序问题)
# ------------------------------
print("Predicting...")
model.eval()
predictions = []
original_indices = []

# 对所有数据进行预测
all_loader = NeighborLoader(
    data, 
    num_neighbors=[10,10], 
    input_nodes=None, # None 表示遍历所有节点
    batch_size=64, 
    shuffle=False
)

with torch.no_grad():
    for batch in tqdm(all_loader, desc="Inference"):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        
        # [重要修正] 切片获取中心节点的预测值
        batch_size = batch.batch_size
        predictions.append(out[:batch_size].cpu())
        
        # 如果需要保持顺序或者 mapping 回 ID，最好记录 batch.n_id[:batch_size]
        # 这里仅做简单的 concat
        
predictions = torch.cat(predictions, dim=0)
print("Prediction shape:", predictions.shape) # 应该是 (len(df), 6)
print(predictions[0])


from sklearn.metrics import mean_squared_error

# ------------------------------
# 11. 计算 RMSE
# ------------------------------

# 1. 确保数据在 CPU 上并转为 numpy
y_true = y_tensor.cpu().numpy()
y_pred = predictions.cpu().numpy()

# 2. 检查形状是否一致
assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

# 3. 计算全局 RMSE (所有数值混在一起，参考价值有限)
global_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"\n=== Global RMSE: {global_rmse:.4f} ===")

# 4. 计算每一列的 RMSE (更有意义)
target_cols = [
    'TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80',
    'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80'
]

print("\n=== RMSE per Target ===")
rmse_per_col = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))

# 创建一个 DataFrame 展示结果，方便查看
rmse_df = pd.DataFrame({
    'Target': target_cols,
    'RMSE': rmse_per_col
})

print(rmse_df)
