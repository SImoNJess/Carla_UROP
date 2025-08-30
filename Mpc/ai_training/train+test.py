from google.colab import drive
drive.mount('/content/drive')

import os, re, json, math, pickle
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = "/content/drive/MyDrive/UROP"
DT_LOG = 0.1                   # 10 Hz
USE_ACCEL_FEATURES = True      # add body-frame a_long, a_lat
SPEED_MIN = 1.0                # m/s (discard very low-speed rows)
V_DOT_MAX = 8.0                # m/s^2 clamp
BETA_DOT_MAX = 4.0             # rad/s clamp
Q_LO, Q_HI = 0.002, 0.998      # global quantile trim (gentle for ~5k rows)
BATCH_TRAIN, BATCH_VAL = 256, 1024
EPOCHS, LR, WEIGHT_DECAY = 40, 2e-3, 1e-4
H1, H2 = 256, 256

def list_data_files(folder):
    pat = re.compile(r"systemid_data_100ms_.*\.(pkl|data|data\.npy)$")
    return sorted([f for f in os.listdir(folder) if pat.fullmatch(f)])

def load_array(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as fh:
            arr = pickle.load(fh)
    else:
        arr = np.load(path, allow_pickle=True)
    return np.asarray(arr)

def fix_to_19(arr):
    """Return (N,19). If 20 columns, drop the extra one by checking control ranges."""
    if arr.ndim != 2:
        raise ValueError(f"expected 2D, got {arr.ndim}D")
    if arr.shape[1] == 19:
        return arr.astype(np.float32)
    if arr.shape[1] != 20:
        raise ValueError(f"expected 19 or 20 columns, got {arr.shape}")
    best = None
    for drop_idx in range(20):
        A = np.delete(arr, drop_idx, axis=1)
        steer, thr, brk = A[:,8], A[:,9], A[:,10]
        ok = (
            (steer >= -1.25) & (steer <= 1.25) &
            (thr   >= -0.05) & (thr   <= 1.05) &
            (brk   >= -0.05) & (brk   <= 1.05)
        )
        score = ok.mean() - 0.5*((thr < -1e-3).mean() + (brk < -1e-3).mean())
        if best is None or score > best[1]:
            best = (drop_idx, score)
    A = np.delete(arr, best[0], axis=1).astype(np.float32)
    print(f"  -> dropped col {best[0]} to restore 19 cols (score={best[1]:.3f})")
    return A

def speed_beta(vx, vy, yaw_deg):
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    v_long =  vx*c + vy*s
    v_lat  = -vx*s + vy*c
    beta = np.arctan2(v_lat, np.clip(v_long, 1e-6, None))
    return np.hypot(vx, vy), beta

def body_accels(ax, ay, yaw_deg):
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return ax*c + ay*s, -ax*s + ay*c

def ang_wrap(x):
    return ((x + np.pi) % (2*np.pi)) - np.pi

def build_xy_from_array(D):
    """Given a single file's (N,19) array, return filtered X,Y for that file only."""
    pvx, pvy = D[:,2], D[:,3]
    prev_ax, prev_ay = D[:,4], D[:,5]
    psi_p_deg, psi_n_deg = D[:,6], D[:,17]
    omega_z = D[:,7]
    steer, thr, brk = D[:,8], D[:,9], D[:,10]
    nvx, nvy = D[:,13], D[:,14]

    v_p, beta_p = speed_beta(pvx, pvy, psi_p_deg)
    v_n, beta_n = speed_beta(nvx, nvy, psi_n_deg)

    v_dot   = (v_n - v_p)/DT_LOG
    beta_dt = ang_wrap(beta_n - beta_p)/DT_LOG

    mask_speed = (v_p > SPEED_MIN) & (v_n > SPEED_MIN)

    v_dot  = np.clip(v_dot,  -V_DOT_MAX,    V_DOT_MAX)
    beta_dt= np.clip(beta_dt,-BETA_DOT_MAX, BETA_DOT_MAX)

    # features
    feats = [v_p, np.sin(beta_p), np.cos(beta_p), steer, thr, brk, omega_z]
    if USE_ACCEL_FEATURES:
        a_long, a_lat = body_accels(prev_ax, prev_ay, psi_p_deg)
        feats += [a_long, a_lat]
    X = np.stack(feats, axis=1)
    Y = np.stack([v_dot, beta_dt], axis=1)

    m = mask_speed
    return X[m], Y[m]

files = list_data_files(DATA_DIR)
assert files, f"No files found in {DATA_DIR}"
X_list, Y_list = [], []
total_rows = 0

for f in files:
    path = os.path.join(DATA_DIR, f)
    A = load_array(path)
    A19 = fix_to_19(A)
    Xf, Yf = build_xy_from_array(A19)
    X_list.append(Xf.astype(np.float32))
    Y_list.append(Yf.astype(np.float32))
    total_rows += len(A19)
    print(f"Loaded {f}: raw {A.shape} -> used {Xf.shape}")

X = np.vstack(X_list)
Y = np.vstack(Y_list)
print(f"\nAll files combined: X {X.shape}, Y {Y.shape} (raw rows across files: {total_rows})")

v_lo, v_hi = np.quantile(Y[:,0], [Q_LO, Q_HI])
b_lo, b_hi = np.quantile(Y[:,1], [Q_LO, Q_HI])
mask_q = (Y[:,0] >= v_lo) & (Y[:,0] <= v_hi) & (Y[:,1] >= b_lo) & (Y[:,1] <= b_hi)
X, Y = X[mask_q], Y[mask_q]
print(f"After quantile trim: X {X.shape}, Y {Y.shape}")

IN_DIM = X.shape[1]

X_tr, X_va, Y_tr, Y_va = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
xsc = StandardScaler().fit(X_tr)
ysc = StandardScaler().fit(Y_tr)

X_trs = xsc.transform(X_tr).astype(np.float32)
X_vas = xsc.transform(X_va).astype(np.float32)
Y_trs = ysc.transform(Y_tr).astype(np.float32)
Y_vas = ysc.transform(Y_va).astype(np.float32)

def symm(Xs, Ys):
    Xf, Yf = Xs.copy(), Ys.copy()
    Xf[:,1] *= -1.0  # sinβ
    Xf[:,3] *= -1.0  # steer
    Yf[:,1] *= -1.0  # beta_dot
    return np.vstack([Xs, Xf]), np.vstack([Ys, Yf])

X_trs, Y_trs = symm(X_trs, Y_trs)

train_dl = DataLoader(TensorDataset(torch.from_numpy(X_trs), torch.from_numpy(Y_trs)),
                      batch_size=BATCH_TRAIN, shuffle=True)
val_dl   = DataLoader(TensorDataset(torch.from_numpy(X_vas), torch.from_numpy(Y_vas)),
                      batch_size=BATCH_VAL, shuffle=False)

class DynHead(nn.Module):
    def __init__(self, in_dim=IN_DIM, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, H1), nn.Tanh(),
            nn.Linear(H1, H2),     nn.Tanh(),
            nn.Linear(H2, out_dim)
        )
        ll = self.net[-1]
        if isinstance(ll, nn.Linear):
            nn.init.uniform_(ll.weight, -1e-3, 1e-3)
            nn.init.zeros_(ll.bias)
    def forward(self, x): return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
net = DynHead().to(device)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.SmoothL1Loss()

best = float("inf")
for epoch in range(1, EPOCHS+1):
    net.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    sched.step()

    net.eval(); tot=0; n=0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            L = loss_fn(net(xb), yb).item()
            tot += L*xb.size(0); n += xb.size(0)
    val = tot/max(n,1)
    print(f"epoch {epoch:02d}  val {val:.6f}")
    if val < best:
        best = val
        torch.save(net.state_dict(), "/content/dyn_head.pt")

os.makedirs("/content/artifacts", exist_ok=True)
torch.save(net.state_dict(), "/content/artifacts/dyn_head.pt")
joblib.dump(xsc, "/content/artifacts/x_scaler.pkl")
joblib.dump(ysc, "/content/artifacts/y_scaler.pkl")
print("Saved artifacts to /content/artifacts/: dyn_head.pt, x_scaler.pkl, y_scaler.pkl")

def val_mae_real():
    net.eval(); mae = np.zeros(2, dtype=np.float64); n=0
    with torch.no_grad():
        for xb, yb in val_dl:
            y_pred_std = net(xb.to(device)).cpu().numpy()
            y_true_std = yb.cpu().numpy()
            y_pred = ysc.inverse_transform(y_pred_std)
            y_true = ysc.inverse_transform(y_true_std)
            mae += np.abs(y_pred - y_true).sum(axis=0); n += y_pred.shape[0]
    return mae / max(n,1)

mae_vdot, mae_bdot = val_mae_real()
print(f"Val MAE  v_dot: {mae_vdot:.4f} m/s^2   beta_dot: {mae_bdot:.4f} rad/s")

with torch.no_grad():
    y_pred_std = net(torch.from_numpy(X_vas[:5]).to(device)).cpu().numpy()
true_real = ysc.inverse_transform(Y_vas[:5])
pred_real = ysc.inverse_transform(y_pred_std)
print("True (v_dot, beta_dot):\n", np.round(true_real, 4))
print("Pred (v_dot, beta_dot):\n", np.round(pred_real, 4))
