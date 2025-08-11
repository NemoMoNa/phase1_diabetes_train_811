# -*- coding: utf-8 -*-


# phase1_diabetes_train.py
# Diabetes regression (PyTorch MLP) with proper train/val/test, EarlyStopping, LR scheduler, MAE
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib


# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Load dataset (10 numeric features)
ds = load_diabetes()
ds = load_diabetes()
X = ds.data.astype(np.float32)
y = ds.target.astype(np.float32)
feature_names = ds.feature_names  # ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']
print("Features:", feature_names)

# --- Split: train/val/test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)

# --- Scale using ONLY train, then apply to val/test
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_te = scaler.transform(X_te)
joblib.dump(scaler, "scaler_diabetes.joblib")


# --- Tensors + loaders
def to_tensor(a):
    return torch.tensor(a, dtype=torch.float32)

X_tr_t  = to_tensor(X_tr)
y_tr_t  = to_tensor(y_tr).unsqueeze(1)
X_val_t = to_tensor(X_val)
y_val_t = to_tensor(y_val).unsqueeze(1)
X_te_t  = to_tensor(X_te)
y_te_t  = to_tensor(y_te).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256)
test_loader  = DataLoader(TensorDataset(X_te_t,  y_te_t),  batch_size=256)


# --- Model
class MLP(nn.Module):
    def __init__(self, in_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )
    def forward(self, x): 
        return self.net(x)

model = MLP(in_dim=X_tr.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)


# --- Eval helpers
def mse_over(loader):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total += loss.item() * xb.size(0); n += xb.size(0)
    return total / n

def mae_over(loader):
    model.eval(); preds_all, trues_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds_all.append(model(xb).cpu().numpy().ravel()) #ravel() は 多次元配列を一次元に「平らにする」 ための NumPy メソッド
            trues_all.append(yb.numpy().ravel())
    preds = np.concatenate(preds_all); trues = np.concatenate(trues_all)
    return mean_absolute_error(trues, preds)

# --- Train with EarlyStopping (on val MSE)
EPOCHS = 100
patience, min_delta = 8, 1e-4
best_val, wait = float("inf"), 0
best_path = "best_diabetes_model.pt"

hist = {"train_mse": [], "val_mse": [], "train_mae": [], "val_mae": []}

for epoch in range(1, EPOCHS+1):
    model.train()
    running, seen = 0.0, 0  #seen 「これまでに処理したサンプル（行）の累計数」
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0); seen += xb.size(0)

    train_mse = running / seen
    val_mse   = mse_over(val_loader)
    train_mae = mae_over(train_loader)
    val_mae   = mae_over(val_loader)

    hist["train_mse"].append(train_mse); hist["val_mse"].append(val_mse)
    hist["train_mae"].append(train_mae); hist["val_mae"].append(val_mae)

    print(f"Epoch {epoch:03d} | Train MSE {train_mse:.4f} | Val MSE {val_mse:.4f} | "
          f"Train MAE {train_mae:.3f} | Val MAE {val_mae:.3f}")

    scheduler.step(val_mse)

    if best_val - val_mse > min_delta:
        best_val = val_mse; wait = 0
        torch.save(model.state_dict(), best_path)
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val MSE: {best_val:.4f}")
            break

# --- Final test evaluation
model.load_state_dict(torch.load(best_path, map_location=device)) #state_dict と呼ばれる辞書形式）を読み込み
model.eval()
with torch.no_grad():
    preds = model(X_te_t.to(device)).cpu().numpy().ravel()
y_true = y_te

mse = mean_squared_error(y_true, preds)
mae = mean_absolute_error(y_true, preds)
print(f"\nFinal TEST MSE: {mse:.4f} | Final TEST MAE: {mae:.3f}")


#--- Plots
plt.figure()
plt.plot(hist["train_mse"], label="Train MSE")
plt.plot(hist["val_mse"], label="Val MSE")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Learning Curve (MSE)")
plt.grid(True); plt.legend(); plt.show()

plt.figure()
plt.plot(hist["train_mae"], label="Train MAE")
plt.plot(hist["val_mae"], label="Val MAE")
plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.title("Learning Curve (MAE)")
plt.grid(True); plt.legend(); plt.show()

plt.figure()
plt.scatter(y_true, preds, alpha=0.5)
mn, mx = min(y_true.min(), preds.min()), max(y_true.max(), preds.max())
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Actual target")
plt.ylabel("Predicted target")
plt.title("Actual vs Predicted (TEST)")
plt.grid(True); plt.show()

print(f"Saved best model to: {os.path.abspath(best_path)}")
print(f"Saved scaler to   : {os.path.abspath('scaler_diabetes.joblib')}")



