# train_inverse_pinn.py
import matplotlib.pyplot as plt
import numpy as np
import os
from simulate_data import generate_dataset
from inverse_pinn import InversePINN, Trainer

def main():
    out_dir = 'data'
    fname = generate_dataset(nS=80, nT=50, scenario='regime_switch', noise_std=0.005, out_path=out_dir)
    model = InversePINN(hidden_V=(128,128), hidden_sigma=(128,128))
    trainer = Trainer(model, data_npz=fname, device='cpu', lambda_data=1.0, lambda_pde=1.0, lambda_reg=1e-5)
    history = trainer.train(num_epochs=2000, lr=1e-3, save_path='models', print_every=200)

    # predict on grid and compute errors if ground-truth available
    data = np.load(fname)
    S_grid = data['S_grid']; t_grid = data['t_grid']
    Vp, sigmap = trainer.predict_grid(S_grid, t_grid)

    sigma_true = data['sigma_true']
    V_true = data['V_true']

    # simple error metrics
    mse_sigma = np.mean((sigmap - sigma_true)**2)
    mse_price = np.mean((Vp - V_true)**2)
    print(f"MSE sigma: {mse_sigma:.4e} | MSE price: {mse_price:.4e}")

    # plots
    os.makedirs('results', exist_ok=True)
    # heatmap sigma error
    plt.figure(figsize=(8,5))
    plt.imshow(np.abs(sigmap - sigma_true), origin='lower', aspect='auto',
               extent=[S_grid.min(), S_grid.max(), t_grid.min(), t_grid.max()])
    plt.colorbar(); plt.title('Absolute Error |sigma_pred - sigma_true|'); plt.xlabel('S'); plt.ylabel('t')
    plt.savefig('results/sigma_error.png', dpi=150)
    plt.close()

    # plot slices at three times
    for ti in [5, 25, 45]:
        plt.figure()
        plt.plot(S_grid, sigma_true[ti,:], label='true')
        plt.plot(S_grid, sigmap[ti,:], label='pred')
        plt.title(f'Sigma slice at t index {ti}, t={t_grid[ti]:.3f}')
        plt.legend()
        plt.savefig(f'results/sigma_slice_t{ti}.png', dpi=150)
        plt.close()

    # save outputs
    np.savez_compressed('results/prediction.npz', S_grid=S_grid, t_grid=t_grid, Vp=Vp, sigmap=sigmap)
    print("Saved results to results/")

if __name__ == '__main__':
    main()
