import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(env_id="FetchReach-v4", log_dir=None):
    def _init():
        env = gym.make(env_id, max_episode_steps=50)
        if log_dir is not None:
            env = Monitor(env, log_dir)
        return env
    return _init

def run_experiment(algo_class, algo_name, kwargs={}, timesteps=300000):
    env_id = "FetchReach-v4"
    log_dir = f"./logs_advanced/{algo_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    env = DummyVecEnv([make_env(env_id, log_dir)])
    eval_env = DummyVecEnv([make_env(env_id)])
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)
    
    model = algo_class("MultiInputPolicy", env, verbose=0, tensorboard_log=f"./tb_logs_advanced/{algo_name}", **kwargs)
    print(f"Starting training for {algo_name}...")
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=False)
    model.save(f"{log_dir}/final_model")
    return log_dir

def plot_all_metrics(log_dirs, names):
    # Plot 1: Training Cumulative Rewards (from monitor)
    plt.figure(figsize=(8, 6))
    for log_dir, name in zip(log_dirs, names):
        try:
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            window = min(len(y), 100)
            if window > 0:
                y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
                x_smooth = x[len(x)-len(y_smooth):]
                plt.plot(x_smooth, y_smooth, label=name)
        except Exception as e:
            pass
    plt.title("Training Curve (Cumulative Rewards)")
    plt.xlabel('Timesteps')
    plt.ylabel('Ep_Rew_Mean')
    plt.legend()
    plt.grid()
    plt.savefig("eval_training_rewards.png")
    
    # Plot 2: Success Rates
    plt.figure(figsize=(8, 6))
    for log_dir, name in zip(log_dirs, names):
        data_path = f"{log_dir}/evaluations.npz"
        if os.path.exists(data_path):
            data = np.load(data_path)
            if 'successes' in data.files:
                sr = np.mean(data['successes'], axis=1)
                plt.plot(data['timesteps'], sr, label=name)
    plt.title("Evaluating Success Rate")
    plt.xlabel('Timesteps')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid()
    plt.savefig("eval_success_rate.png")
    
    print("Saved eval_training_rewards.png and eval_success_rate.png")

if __name__ == "__main__":
    TIMESTEPS = 300000
    experiments = [
        (PPO, "PPO_Baseline", {}), 
        (SAC, "SAC_Baseline", {}),
        # Ablation 1: tuned PPO
        (PPO, "PPO_Optimized_batch256", {"batch_size": 256, "learning_rate": 1e-3}),
        # Ablation 2: tuned SAC (larger batch size, different learning rate)
        (SAC, "SAC_Optimized_batch512", {"batch_size": 512, "learning_rate": 1e-3}),
    ]
    
    logs = []
    names = []
    
    for algo_class, name, kwargs in experiments:
        log_dir = run_experiment(algo_class, name, kwargs, TIMESTEPS)
        logs.append(log_dir)
        names.append(name)
        
    print("Plotting comparative results...")
    plot_all_metrics(logs, names)
    print("All tasks finished successfully.")
