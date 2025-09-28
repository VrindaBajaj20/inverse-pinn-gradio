import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

from inverse_pinn import InversePINN, Trainer
import simulate_data
from detect_regimes import compute_features_from_sigma, run_kmeans
from load_real_data import build_grid_from_yfinance

# --------- Volatility Inference ---------
def volatility_inference(mode, dataset_scenario=None):
    """Run PINN volatility inference"""
    results_text = ""
    if mode == "Synthetic":
        fname = simulate_data.generate_dataset(
            nS=64, nT=40, scenario=dataset_scenario, noise_std=0.005, out_path='data'
        )
        model = InversePINN(hidden_V=(128,128), hidden_sigma=(128,128))
        trainer = Trainer(model, data_npz=fname, device='cpu', lambda_data=1.0, lambda_pde=1.0, lambda_reg=1e-5)
        trainer.data_npz = fname
        trainer.train(num_epochs=500, lr=1e-3, save_path='models', print_every=100)

        data = np.load(fname)
        S_grid = data['S_grid']
        t_grid = data['t_grid']

        sigmap = trainer.predict()
        os.makedirs("results", exist_ok=True)
        np.savez("results/prediction.npz", S_grid=S_grid, t_grid=t_grid, sigmap=sigmap)

        results_text = f"Training finished. Saved results in 'results/prediction.npz'. Shape: {sigmap.shape}"
        return results_text

    return "Mode not implemented for Gradio yet."


# --------- Regime Detection ---------
def regime_detection(n_clusters):
    """Run regime detection on previously saved results"""
    if not os.path.exists('results/prediction.npz'):
        return "No prediction results found. Run Volatility Inference first."
    
    data = np.load('results/prediction.npz')
    S_grid = data['S_grid']
    sigmap = data['sigmap']

    features = compute_features_from_sigma(sigmap, S_grid)
    labels, _ = run_kmeans(features, k=n_clusters)

    # Generate a plot of mean sigma and std sigma
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(features[:,0], label='Mean Sigma')
    ax.plot(features[:,1], label='Std Sigma')
    ax.set_title("Sigma Features per Time Step")
    ax.legend()
    plt.tight_layout()

    return labels.tolist(), fig


# --------- Real Market Data ---------
def real_market_data(ticker, expiries, strikes):
    S_grid, t_grid, V_grid, S_now = build_grid_from_yfinance(
        ticker=ticker,
        expiries=expiries,
        strikes_per_expiry=strikes
    )

    df = pd.DataFrame(V_grid, index=[f"t={t:.3f}" for t in t_grid],
                      columns=[f"S={s:.2f}" for s in S_grid])
    
    fig, ax = plt.subplots(figsize=(8,5))
    c = ax.imshow(V_grid, origin='lower', aspect='auto',
                  extent=[S_grid.min(), S_grid.max(), t_grid.min(), t_grid.max()])
    ax.set_xlabel("Underlying Price (S)")
    ax.set_ylabel("Time to Maturity (t)")
    ax.set_title(f"Option Price Surface for {ticker}")
    fig.colorbar(c, ax=ax, label="Call Price")
    plt.tight_layout()

    return f"Current spot price: {S_now:.2f}", df, fig


# --------- Gradio Interface ---------
with gr.Blocks() as demo:
    gr.Markdown("# Inverse PINNs for Volatility & Regimes")

    with gr.Tab("Volatility Inference"):
        mode_input = gr.Radio(["Synthetic"], label="Mode")
        scenario_input = gr.Dropdown(["regime_switch", "smooth", "heston_like"], label="Scenario")
        run_button = gr.Button("Run Inference")
        output_text = gr.Textbox(label="Status")
        run_button.click(fn=volatility_inference, inputs=[mode_input, scenario_input], outputs=output_text)

    with gr.Tab("Regime Detection"):
        cluster_slider = gr.Slider(2,5, step=1, label="Number of clusters (KMeans)")
        run_button2 = gr.Button("Run Detection")
        output_labels = gr.Textbox(label="Regime Labels")
        output_fig = gr.Plot(label="Feature Plot")
        run_button2.click(fn=regime_detection, inputs=cluster_slider, outputs=[output_labels, output_fig])

    with gr.Tab("Real Market Data"):
        ticker_input = gr.Textbox(value="SPY", label="Ticker")
        expiries_slider = gr.Slider(1,5,step=1,value=3,label="Number of Expiries")
        strikes_slider = gr.Slider(5,50,step=1,value=25,label="Strikes per Expiry")
        run_button3 = gr.Button("Fetch Data")
        output_text2 = gr.Textbox(label="Spot Price")
        output_df = gr.Dataframe(label="Price Matrix")
        output_fig2 = gr.Plot(label="Price Surface")
        run_button3.click(fn=real_market_data,
                          inputs=[ticker_input, expiries_slider, strikes_slider],
                          outputs=[output_text2, output_df, output_fig2])

demo.launch()
