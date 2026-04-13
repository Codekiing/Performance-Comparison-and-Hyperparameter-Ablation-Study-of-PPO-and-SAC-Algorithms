import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def extract_tb_data(logdir, scalar_key):
    # Find event file
    event_files = [os.path.join(logdir, f) for f in os.listdir(logdir) if f.startswith('events.out.tfevents')]
    if not event_files:
        return [], []
    event_file = event_files[0] # assuming one event file per algo run
    
    ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    
    if scalar_key not in ea.Tags()['scalars']:
        return [], []
        
    events = ea.Scalars(scalar_key)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def plot_success_rates():
    plt.figure(figsize=(8, 6))
    for algo in ['PPO', 'SAC']:
        data = np.load(f'logs/{algo}_FetchReach-v4/evaluations.npz')
        if 'successes' in data.files:
            success_rate = np.mean(data['successes'], axis=1)
            timesteps = data['timesteps']
            plt.plot(timesteps, success_rate, label=f'{algo}')
            print(f"{algo} Final Success Rate: {success_rate[-1] * 100:.2f}%")
            
    plt.title("Evaluation Success Rate over Timesteps")
    plt.xlabel('Timesteps')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid()
    plt.savefig("success_rate_curve.png")
    print("Saved success_rate_curve.png")

def plot_losses():
    # PPO
    ppo_dirs = [d for d in os.listdir('tb_logs/PPO') if 'PPO' in d]
    if ppo_dirs:
        ppo_dir = os.path.join('tb_logs/PPO', sorted(ppo_dirs)[-1]) # gets latest run
        ppo_steps, ppo_policy_loss = extract_tb_data(ppo_dir, 'train/policy_gradient_loss')
        _, ppo_value_loss = extract_tb_data(ppo_dir, 'train/value_loss')
        
        if ppo_steps:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(ppo_steps, ppo_policy_loss, label='PPO Policy/Actor Loss')
            plt.title('PPO Policy Loss')
            plt.xlabel('Timesteps')
            plt.grid()
            
            plt.subplot(1, 2, 2)
            plt.plot(ppo_steps, ppo_value_loss, label='PPO Value Loss')
            plt.title('PPO Value Loss')
            plt.xlabel('Timesteps')
            plt.grid()
            plt.savefig("ppo_loss_curves.png")
            print("Saved ppo_loss_curves.png")

    # SAC
    sac_dirs = [d for d in os.listdir('tb_logs/SAC') if 'SAC' in d]
    if sac_dirs:
        sac_dir = os.path.join('tb_logs/SAC', sorted(sac_dirs)[-1])
        sac_steps, sac_actor_loss = extract_tb_data(sac_dir, 'train/actor_loss')
        _, sac_critic_loss = extract_tb_data(sac_dir, 'train/critic_loss')
        
        if sac_steps:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(sac_steps, sac_actor_loss, label='SAC Actor Loss', color='orange')
            plt.title('SAC Actor Loss')
            plt.xlabel('Timesteps')
            plt.grid()
            
            plt.subplot(1, 2, 2)
            plt.plot(sac_steps, sac_critic_loss, label='SAC Critic Loss', color='orange')
            plt.title('SAC Critic Loss')
            plt.xlabel('Timesteps')
            plt.grid()
            plt.savefig("sac_loss_curves.png")
            print("Saved sac_loss_curves.png")

if __name__ == "__main__":
    plot_success_rates()
    plot_losses()
