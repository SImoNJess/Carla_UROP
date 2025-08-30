# ilqr_numpy_blackbox.py
import numpy as np
from dyn_head_inference import LearnedDynHead

def wrap_angle(a): return (a + np.pi) % (2*np.pi) - np.pi

# ---- Discrete dynamics: x = [x, y, v, phi, beta], u = [steer, thr, brk] ----
class BlackBoxDynamics:
    def __init__(self, dt=0.1, lr_mean=1.5389, dyn_head=None):
        self.dt = dt
        self.lr_mean = lr_mean
        self.dyn = dyn_head or LearnedDynHead(device="cpu")

    def step(self, x, u):
        px, py, v, phi, beta = x
        steer, thr, brk = u
        # learned parts
        v_dot, b_dot = self.dyn.predict_from_state(v, beta, steer, thr, brk, lr_mean=self.lr_mean)
        # kinematics
        dx = v * np.cos(phi + beta)
        dy = v * np.sin(phi + beta)
        phi_dot = v * np.sin(beta) / max(self.lr_mean, 1e-6)
        # integrate
        x_next = np.array([
            px + self.dt * dx,
            py + self.dt * dy,
            max(0.0, v + self.dt * v_dot),
            wrap_angle(phi + self.dt * phi_dot),
            wrap_angle(beta + self.dt * b_dot),
        ], dtype=np.float32)
        return x_next

# ---- Cost (similar spirit to repo) ----
class CostFunction:
    def __init__(self, waypoints, dp=1.0, speed_ref=8.0, w_pos=0.04, w_speed=0.002, w_u=0.0005):
        self.route = np.asarray(waypoints, dtype=np.float32)  # shape (M,2)
        self.dp = float(dp)
        self.speed_ref = float(speed_ref)
        self.w_pos, self.w_speed, self.w_u = w_pos, w_speed, w_u
        self.target_ratio = len(self.route) / (6*np.pi) if len(self.route)>0 else 1.0

    def pos_cost_softmin(self, x):
        # soft-min over future points (log-sum-exp on negative squared distances)
        dx = x[0] - self.route[:,0]
        dy = x[1] - self.route[:,1]
        neg_d = -(dx*dx + dy*dy)/(self.dp*self.dp)
        # soft-min ~ -logsumexp
        m = neg_d.max()
        return -(m + np.log(np.exp(neg_d - m).sum() + 1e-9))

    def stage(self, x, u, time_steps_ratio=1.2):  # TIME_STEPS ~ 60 → ratio ~ 60/50
        steering, thr, brk = u
        c_pos = self.pos_cost_softmin(x)
        c_speed = (x[2] - self.speed_ref)**2
        c_u = (steering**2 + thr**2 + brk**2 + thr*brk)
        return (self.w_pos*c_pos + self.w_speed*c_speed + self.w_u*c_u) / time_steps_ratio

    def terminal(self, x):
        if len(self.route)==0: return 0.0
        dx = x[0] - self.route[-1,0]
        dy = x[1] - self.route[-1,1]
        c_pos = (dx*dx + dy*dy) / (self.target_ratio**2)
        return c_pos

# ---- Finite-difference helpers ----
def fd_jacobian_f(dyn: BlackBoxDynamics, x, u, eps_x=1e-4, eps_u=1e-4):
    nX, nU = 5, 3
    f0 = dyn.step(x, u)
    fx = np.zeros((nX, nX), dtype=np.float32)
    fu = np.zeros((nX, nU), dtype=np.float32)
    for i in range(nX):
        dx = np.zeros(nX); dx[i] = eps_x
        f_plus  = dyn.step(x+dx, u)
        f_minus = dyn.step(x-dx, u)
        fx[:, i] = (f_plus - f_minus) / (2*eps_x)
    for j in range(nU):
        du = np.zeros(nU); du[j] = eps_u
        f_plus  = dyn.step(x, u+du)
        f_minus = dyn.step(x, u-du)
        fu[:, j] = (f_plus - f_minus) / (2*eps_u)
    return fx, fu

def fd_cost_grads(cost: CostFunction, x, u, eps_x=1e-4, eps_u=1e-4):
    # returns l_x, l_u, l_xx, l_uu, l_ux
    l0 = cost.stage(x, u)
    nX, nU = 5, 3
    l_x = np.zeros(nX); l_u = np.zeros(nU)
    l_xx = np.zeros((nX,nX)); l_uu = np.zeros((nU,nU)); l_ux = np.zeros((nU,nX))
    # gradients
    for i in range(nX):
        dx = np.zeros(nX); dx[i]=eps_x
        l_plus  = cost.stage(x+dx, u)
        l_minus = cost.stage(x-dx, u)
        l_x[i] = (l_plus - l_minus)/(2*eps_x)
    for j in range(nU):
        du = np.zeros(nU); du[j]=eps_u
        l_plus  = cost.stage(x, u+du)
        l_minus = cost.stage(x, u-du)
        l_u[j] = (l_plus - l_minus)/(2*eps_u)
    # Hessians (symmetric) via FD on gradients
    for i in range(nX):
        dx = np.zeros(nX); dx[i]=eps_x
        lp = np.zeros(nX); lm = np.zeros(nX)
        for k in range(nX):
            dk = np.zeros(nX); dk[k]=eps_x
            lp[k] = (cost.stage(x+dx+dk,u) - cost.stage(x+dx-dk,u))/(2*eps_x)
            lm[k] = (cost.stage(x-dx+dk,u) - cost.stage(x-dx-dk,u))/(2*eps_x)
        l_xx[:,i] = (lp - lm)/(2*eps_x)
    for j in range(nU):
        du = np.zeros(nU); du[j]=eps_u
        lp = np.zeros(nU); lm = np.zeros(nU)
        for k in range(nU):
            dk = np.zeros(nU); dk[k]=eps_u
            lp[k] = (cost.stage(x,u+du+dk) - cost.stage(x,u+du-dk))/(2*eps_u)
            lm[k] = (cost.stage(x,u-du+dk) - cost.stage(x,u-du-dk))/(2*eps_u)
        l_uu[:,j] = (lp - lm)/(2*eps_u)
    for j in range(nU):
        du = np.zeros(nU); du[j]=eps_u
        for i in range(nX):
            dx = np.zeros(nX); dx[i]=eps_x
            l_up = cost.stage(x+dx, u+du) - cost.stage(x+dx, u-du)
            l_um = cost.stage(x-dx, u+du) - cost.stage(x-dx, u-du)
            l_ux[j,i] = (l_up - l_um)/(4*eps_x*eps_u)
    return l_x, l_u, l_xx, l_uu, l_ux

def fd_cost_final(cost: CostFunction, x, eps_x=1e-4):
    nX=5
    phi = cost.terminal(x)
    l_x = np.zeros(nX); l_xx = np.zeros((nX,nX))
    for i in range(nX):
        dx = np.zeros(nX); dx[i]=eps_x
        l_x[i] = (cost.terminal(x+dx) - cost.terminal(x-dx))/(2*eps_x)
        for k in range(nX):
            dk = np.zeros(nX); dk[k]=eps_x
            lp = (cost.terminal(x+dx+dk) - cost.terminal(x+dx-dk))/(2*eps_x)
            lm = (cost.terminal(x-dx+dk) - cost.terminal(x-dx-dk))/(2*eps_x)
            l_xx[k,i] = (lp - lm)/(2*eps_x)
    return l_x, l_xx

# ---- iLQR core ----
def ilqr_optimize(dyn:BlackBoxDynamics, cost:CostFunction, x0, u_trj, max_iter=60, reg=1.0):
    T = u_trj.shape[0]        # stages
    nX, nU = 5, 3
    alphas = [1.0, 0.7, 0.4, 0.2, 0.1]
    # rollout
    x_trj = np.zeros((T+1, nX), dtype=np.float32)
    x_trj[0] = x0
    for t in range(T):
        x_trj[t+1] = dyn.step(x_trj[t], u_trj[t])

    def total_cost(x_trj, u_trj):
        c = 0.0
        for t in range(T):
            c += cost.stage(x_trj[t], u_trj[t])
        c += cost.terminal(x_trj[-1])
        return c

    for it in range(max_iter):
        # backward pass
        V_x, V_xx = fd_cost_final(cost, x_trj[-1])
        k_trj = np.zeros((T, nU), dtype=np.float32)
        K_trj = np.zeros((T, nU, nX), dtype=np.float32)

        diverged = False
        for t in reversed(range(T)):
            l_x, l_u, l_xx, l_uu, l_ux = fd_cost_grads(cost, x_trj[t], u_trj[t])
            f_x, f_u = fd_jacobian_f(dyn, x_trj[t], u_trj[t])

            Q_x  = l_x  + f_x.T @ V_x
            Q_u  = l_u  + f_u.T @ V_x
            Q_xx = l_xx + f_x.T @ V_xx @ f_x
            Q_ux = l_ux + f_u.T @ V_xx @ f_x
            Q_uu = l_uu + f_u.T @ V_xx @ f_u

            # regularization for positive-definite
            Q_uu_reg = Q_uu + reg*np.eye(nU)
            try:
                inv = np.linalg.inv(Q_uu_reg)
            except np.linalg.LinAlgError:
                diverged = True; break

            k = -inv @ Q_u
            K = -inv @ Q_ux
            k_trj[t] = k
            K_trj[t] = K
            V_x  = Q_x  + K.T @ Q_u + Q_ux.T @ k + K.T @ Q_uu @ k
            V_xx = Q_xx + K.T @ Q_ux + Q_ux.T @ K + K.T @ Q_uu @ K

        if diverged:
            reg *= 10.0
            continue

        # forward line-search
        cost_prev = total_cost(x_trj, u_trj)
        accepted = False
        for a in alphas:
            x_new = np.zeros_like(x_trj)
            u_new = np.zeros_like(u_trj)
            x_new[0] = x_trj[0]
            for t in range(T):
                du = a*k_trj[t] + K_trj[t] @ (x_new[t] - x_trj[t])
                u_new[t] = u_trj[t] + du
                # clamp controls to valid range
                u_new[t,0] = np.clip(u_new[t,0], -1.0, 1.0)
                u_new[t,1] = np.clip(u_new[t,1],  0.0, 1.0)
                u_new[t,2] = np.clip(u_new[t,2],  0.0, 1.0)
                x_new[t+1] = dyn.step(x_new[t], u_new[t])
            c_new = total_cost(x_new, u_new)
            if c_new < cost_prev:
                x_trj, u_trj = x_new, u_new
                accepted = True
                reg = max(1e-6, reg/2.0)
                break
        if not accepted:
            reg *= 10.0
        # stopping criterion: small improvement
        if accepted and abs(cost_prev - c_new) < 1e-3:
            break

    return x_trj, u_trj
