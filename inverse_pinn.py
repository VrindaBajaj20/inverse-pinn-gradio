# inverse_pinn.py
"""
Inverse PINN: jointly learn V(S,t) (option price) and sigma(S,t) (local vol)
from observed noisy option prices V_obs at points (S,t).

Usage pattern:
  model = InversePINN(hidden_V=[64,64], hidden_sigma=[64,64])
  trainer = Trainer(model, data_dict, device='cuda')
  trainer.train(num_epochs=10000)
"""

import torch
import torch.nn as nn
import numpy as np
import os

# small MLP
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(64,64), act=nn.Tanh):
        super().__init__()
        layers = []
        dims = [in_dim] + list(hidden_dims)
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(act())
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)
        # initialize
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

class InversePINN(nn.Module):
    def __init__(self, hidden_V=(64,64), hidden_sigma=(64,64)):
        super().__init__()
        self.netV = MLP(2,1,hidden_V)         # inputs: (S_scaled, t_scaled) -> price
        self.netsigma = MLP(2,1,hidden_sigma) # inputs -> sigma (positive via softplus)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: (N,2) where col0 S, col1 t
        V = self.netV(x)
        sigma = self.softplus(self.netsigma(x)) + 1e-6
        return V.squeeze(-1), sigma.squeeze(-1)

# PDE residual: Black-Scholes with local vol sigma(S,t)
def pde_residual(model, x, r):
    # x: (N,2) requires grad for autograd
    x = x.clone().requires_grad_(True)
    V, sigma = model(x)
    S = x[:,0:1]
    t = x[:,1:2]
    # time derivative: ∂V/∂t
    V_t = torch.autograd.grad(V.sum(), x, create_graph=True)[0][:,1:2]
    # first derivative wrt S
    V_S = torch.autograd.grad(V.sum(), x, create_graph=True)[0][:,0:1]
    # second derivative: use grad of V_S wrt x
    V_S2 = torch.autograd.grad(V_S.sum(), x, create_graph=True)[0][:,0:1]

    # Black-Scholes PDE residual (note sign conv): V_t + 0.5 * sigma^2 * S^2 * V_SS + r*S*V_S - r*V = 0
    res = V_t + 0.5 * (sigma**2).unsqueeze(-1) * (S**2) * V_S2 + r * S * V_S - r * V.unsqueeze(-1)
    return res.squeeze(-1)

# Trainer helper
class Trainer:
    def __init__(self, model, data_npz, device='cpu',
                 lambda_data=1.0, lambda_pde=1.0, lambda_reg=1e-4):
        self.model = model.to(device)
        self.device = device
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_reg = lambda_reg
        self.data_npz = data_npz 
        # load data
        data = np.load(data_npz)
        S_grid = data['S_grid']
        t_grid = data['t_grid']
        V_noisy = data['V_noisy']
        self.K = float(data['K']) if 'K' in data else 100.0
        self.r = float(data['r']) if 'r' in data else 0.01
        self.T = float(data['T']) if 'T' in data else 1.0

        # Flatten grid into observation points (use all points as observed for now)
        Sg, tg = np.meshgrid(S_grid, t_grid, indexing='xy')
        X = np.vstack([Sg.ravel(), tg.ravel()]).T.astype(np.float32)
        y = V_noisy.ravel().astype(np.float32)
        # scale S and t to ~1 for numerical stability
        self.S_mean = Sg.mean()
        self.S_std = Sg.std()
        self.t_mean = tg.mean()
        self.t_std = tg.std()
        Xs = np.copy(X)
        Xs[:,0] = (Xs[:,0] - self.S_mean) / (self.S_std + 1e-9)
        Xs[:,1] = (Xs[:,1] - self.t_mean) / (self.t_std + 1e-9)

        self.X = torch.tensor(Xs, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

        # collocation points for PDE residual - sample uniformly in domain
        N_coll = min(2000, len(self.X))
        perm = np.random.choice(len(self.X), N_coll, replace=False)
        self.X_coll = self.X[perm].detach().clone()

    def loss_fn(self):
        V_pred, sigma_pred = self.model(self.X)
        data_loss = nn.MSELoss()(V_pred, self.y)
        # PDE residual loss
        res = pde_residual(self.model, self.X_coll, self.r)
        pde_loss = torch.mean(res**2)
        # regularizer on sigma smoothness (finite diff on collocation)
        sigma_reg = torch.mean((sigma_pred[1:] - sigma_pred[:-1])**2)
        loss = self.lambda_data * data_loss + self.lambda_pde * pde_loss + self.lambda_reg * sigma_reg
        return loss, data_loss.detach().cpu().item(), pde_loss.detach().cpu().item(), sigma_reg.detach().cpu().item()

    def train(self, num_epochs=5000, lr=1e-3, save_path='models', print_every=200):
        os.makedirs(save_path, exist_ok=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        history = {'loss':[], 'data_loss':[], 'pde_loss':[], 'reg':[]}
        for ep in range(1, num_epochs+1):
            opt.zero_grad()
            loss, dloss, ploss, rloss = self.loss_fn()
            loss.backward()
            opt.step()
            history['loss'].append(loss.item())
            history['data_loss'].append(dloss)
            history['pde_loss'].append(ploss)
            history['reg'].append(rloss)
            if ep % print_every == 0 or ep==1:
                print(f"Epoch {ep}/{num_epochs} Loss {loss.item():.4e} Data {dloss:.4e} PDE {ploss:.4e} Reg {rloss:.4e}")
        # Save model
        torch.save(self.model.state_dict(), os.path.join(save_path, 'inverse_pinn.pt'))
        print("Model saved.")
        return history

    def predict_grid(self, S_grid, t_grid):
        # scale with training mean/std
        Sg, tg = np.meshgrid(S_grid, t_grid, indexing='xy')
        X = np.vstack([Sg.ravel(), tg.ravel()]).T.astype(np.float32)
        Xs = np.copy(X)
        Xs[:,0] = (Xs[:,0] - self.S_mean) / (self.S_std + 1e-9)
        Xs[:,1] = (Xs[:,1] - self.t_mean) / (self.t_std + 1e-9)
        Xtorch = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            Vp, sigmap = self.model(Xtorch)
        Vp = Vp.cpu().numpy().reshape(len(t_grid), len(S_grid))
        sigmap = sigmap.cpu().numpy().reshape(len(t_grid), len(S_grid))
        return Vp, sigmap
    def predict(self):
        """
        Convenience wrapper: Predict on the same grid used in training,
        and return only sigma(S,t).
        """
        # recover grids used during training from npz file
        data = np.load(self.data_npz)  # we already have this file path
        S_grid = data['S_grid']
        t_grid = data['t_grid']
        _, sigmap = self.predict_grid(S_grid, t_grid)
        return sigmap
