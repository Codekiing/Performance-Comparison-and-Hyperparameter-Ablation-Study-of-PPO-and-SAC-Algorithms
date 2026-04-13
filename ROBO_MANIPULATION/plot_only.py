import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import ts2xy, load_results

def plot_training(log_dirs, algos, title="Training Curves"):
    plt.figure(figsize=(8, 6))
    for log_dir, algo in zip(log_dirs, algos):
        try:
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            window = min(len(y), 50)
            if window > 0:
                y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
                x_smooth = x[len(x)-len(y_smooth):]
                plt.plot(x_smooth, y_smooth, label=algo)
            else:
                plt.plot(x, y, label=algo)
        except Exception as e:
            print(f"Could not load/plot from {log_dir}: {e}")
            
    plt.title(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.legend()
    plt.grid()
    plt.savefig("training_curve.png")
    print("Saved training_curve.png")

plot_training(["./logs/PPO_FetchReach-v4", "./logs/SAC_FetchReach-v4"], ["PPO", "SAC"])
