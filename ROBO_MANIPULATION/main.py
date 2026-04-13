import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import plot_results, ts2xy, load_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import imageio

def make_env(env_id="FetchReach-v4", log_dir=None):
    def _init():
        env = gym.make(env_id, max_episode_steps=50, render_mode="rgb_array")
        if log_dir is not None:
            env = Monitor(env, log_dir)
        return env
    return _init

def train_model(algo="PPO", env_id="FetchReach-v4", total_timesteps=200000):
    log_dir = f"./logs/{algo}_{env_id}"
    os.makedirs(log_dir, exist_ok=True)
    
    env = DummyVecEnv([make_env(env_id, log_dir)])
    
    eval_env = DummyVecEnv([make_env(env_id)])
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)
    
    if algo == "PPO":
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./tb_logs/{algo}")
    elif algo == "SAC":
        model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./tb_logs/{algo}")
    else:
        raise ValueError("Unsupported algorithm")
        
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    model.save(f"{log_dir}/final_model")
    return log_dir

def plot_training(log_dirs, algos, title="Training Curves"):
    plt.figure(figsize=(8, 6))
    for log_dir, algo in zip(log_dirs, algos):
        try:
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            
            # Subsample or smooth
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

def evaluate_and_record(algo, env_id="FetchReach-v4", log_dir=None):
    video_folder = f"./videos/{algo}"
    os.makedirs(video_folder, exist_ok=True)
    
    if algo == "PPO":
        model = PPO.load(f"{log_dir}/best_model")
    else:
        model = SAC.load(f"{log_dir}/best_model")
        
    env = make_env(env_id=env_id)()
    obs, info = env.reset()
    frames = []
    
    for _ in range(150):
        # Render environment
        img = env.render()
        frames.append(img)
        
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            
    env.close()
    
    # Save as gif
    imageio.mimsave(f"{video_folder}/{algo}_FetchReach.gif", frames, fps=30)
    print(f"Saved recording to {video_folder}/{algo}_FetchReach.gif")

if __name__ == "__main__":
    env_id = "FetchReach-v4"
    timesteps = 100000 
    
    print("Training PPO...")
    log_ppo = train_model("PPO", env_id, timesteps)
    
    print("Training SAC...")
    log_sac = train_model("SAC", env_id, timesteps)
    
    print("Plotting results...")
    plot_training([log_ppo, log_sac], ["PPO", "SAC"])
    
    print("Evaluating and recording PPO...")
    evaluate_and_record("PPO", env_id, log_ppo)
    
    print("Evaluating and recording SAC...")
    evaluate_and_record("SAC", env_id, log_sac)
    print("Done!")
