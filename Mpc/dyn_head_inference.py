# dyn_head_inference.py (compat-friendly, no sklearn required locally)
import os, json, numpy as np, torch, torch.nn as nn

def _wrap(a): return (a + np.pi) % (2*np.pi) - np.pi
def _speed_beta_world(vx, vy, yaw_deg):
    yaw = np.deg2rad(yaw_deg); c, s = np.cos(yaw), np.sin(yaw)
    v_long =  vx*c + vy*s; v_lat = -vx*s + vy*c
    return float(np.hypot(vx,vy)), float(np.arctan2(v_lat, np.clip(v_long,1e-6,None)))
def _acc_body(ax, ay, yaw_deg):
    yaw = np.deg2rad(yaw_deg); c, s = np.cos(yaw), np.sin(yaw)
    return float(ax*c + ay*s), float(-ax*s + ay*c)

class _DynNet(nn.Module):
    def __init__(self, in_dim, out_dim=2, h1=256, h2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2),     nn.Tanh(),
            nn.Linear(h2, out_dim)
        )
        ll = self.net[-1]
        if isinstance(ll, nn.Linear):
            nn.init.uniform_(ll.weight, -1e-3, 1e-3); nn.init.zeros_(ll.bias)
    def forward(self, x): return self.net(x)

class _ScalerCompat:
    """Manual standardize / inverse using saved mean & scale arrays."""
    def __init__(self, x_mean, x_scale, y_mean, y_scale):
        self.x_mean = x_mean.astype(np.float32)
        self.x_scale= x_scale.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_scale= y_scale.astype(np.float32)
        self.n_features_in_ = self.x_mean.shape[0]
    def x_transform(self, X):
        return ((X - self.x_mean) / (self.x_scale + 1e-12)).astype(np.float32)
    def y_inverse(self, Ystd):
        return (Ystd * self.y_scale + self.y_mean).astype(np.float32)

class LearnedDynHead:
    """
    Predicts (v_dot [m/s^2], beta_dot [rad/s]) in REAL units.
    Uses artifacts/compat/* if present (no sklearn needed). Falls back to sklearn pickles if you have them.
    """
    def __init__(self,
                 ckpt="dyn_head.pt",
                 x_scaler_pkl="x_scaler.pkl",
                 y_scaler_pkl="y_scaler.pkl",
                 compat_dir="compat",
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Prefer compat arrays
        if os.path.isdir(compat_dir) and all(os.path.isfile(os.path.join(compat_dir,f))
                                             for f in ["x_mean.npy","x_scale.npy","y_mean.npy","y_scale.npy","meta.json"]):
            x_mean = np.load(os.path.join(compat_dir,"x_mean.npy"))
            x_scale= np.load(os.path.join(compat_dir,"x_scale.npy"))
            y_mean = np.load(os.path.join(compat_dir,"y_mean.npy"))
            y_scale= np.load(os.path.join(compat_dir,"y_scale.npy"))
            meta = json.load(open(os.path.join(compat_dir,"meta.json"),"r"))
            self.in_dim = int(meta["in_dim"])
            self.use_acc = bool(meta.get("use_accel_features", self.in_dim==9))
            self.scaler = _ScalerCompat(x_mean, x_scale, y_mean, y_scale)
        else:
            # Fallback (requires sklearn installed). Better to use compat arrays on Python 3.7.
            import joblib
            xsc = joblib.load(x_scaler_pkl)
            ysc = joblib.load(y_scaler_pkl)
            self.in_dim = int(getattr(xsc, "n_features_in_", len(xsc.mean_)))
            self.use_acc = (self.in_dim == 9)
            self.scaler = _ScalerCompat(xsc.mean_.astype(np.float32),
                                        xsc.scale_.astype(np.float32),
                                        ysc.mean_.astype(np.float32),
                                        ysc.scale_.astype(np.float32))

        self.net = _DynNet(in_dim=self.in_dim).to(self.device)
        self.net.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.net.eval()

    def _predict_std(self, xs_np):
        with torch.no_grad():
            y_std = self.net(torch.from_numpy(xs_np).to(self.device)).cpu().numpy()
        return y_std

    def predict_from_raw(self, prev_state, action):
        vx, vy = float(prev_state["vx"]), float(prev_state["vy"])
        ax, ay = float(prev_state.get("ax",0.0)), float(prev_state.get("ay",0.0))
        yaw_deg = float(prev_state["yaw_deg"])
        omega_z = float(prev_state.get("yawrate_z",0.0))
        if isinstance(action, dict):
            steer, thr, brk = float(action["steer"]), float(action["throttle"]), float(action["brake"])
        else:
            steer, thr, brk = map(float, action)

        v, beta = _speed_beta_world(vx, vy, yaw_deg)
        feats = [v, np.sin(beta), np.cos(beta), steer, thr, brk, omega_z]
        if self.use_acc:
            a_long, a_lat = _acc_body(ax, ay, yaw_deg); feats += [a_long, a_lat]

        X = np.array(feats, np.float32)[None,:]
        Xs = self.scaler.x_transform(X.astype(np.float64))
        y_std = self._predict_std(Xs)
        y = self.scaler.y_inverse(y_std)
        return float(y[0,0]), float(y[0,1])

    def predict_from_state(self, v, beta, steer, thr, brk, lr_mean=1.5389):
        omega_z = float(v*np.sin(beta)/max(lr_mean,1e-6))
        feats = [float(v), float(np.sin(beta)), float(np.cos(beta)),
                 float(steer), float(thr), float(brk), omega_z]
        if self.use_acc:
            feats += [0.0, 0.0]
        X = np.array(feats, np.float32)[None,:]
        Xs = self.scaler.x_transform(X.astype(np.float64))
        y_std = self._predict_std(Xs)
        y = self.scaler.y_inverse(y_std)
        return float(y[0,0]), float(y[0,1])
