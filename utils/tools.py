import os
from os import path
import torch
from nnmodels.mlp_critic import Value
from nnmodels.mlp_policy import Policy
import pickle
from core.ppo import ppo_step, estimate_advantages
import numpy as np
from argparse import Namespace
from utils.torchpy import to_device

def assets_dir():
    # Returns the absolute path to the assets directory.
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))

def initialize_networks(args, action_space_size, dtype, device):
    if args.model_path is None:
        policy_network = Policy(args.hist_length, action_space_size, log_std=args.log_std)
        value_network = Value(args.hist_length)
    else:
        policy_network, value_network, _ = pickle.load(open(args.model_path, "rb"))
    policy_network.to(device).to(dtype)
    value_network.to(device).to(dtype)
    return policy_network, value_network

def epoch_prepare(imgs_depth, goals, rays, hist_actions, actions, returns, advantages, fixed_log_probs, device):
    # Prepare the data for a training epoch by shuffling and returning the data in batches.
    perm = np.arange(imgs_depth.shape[0])
    np.random.shuffle(perm)
    perm = torch.LongTensor(perm).to(device)
    return [x[perm].clone() for x in [imgs_depth, goals, rays, hist_actions, actions, returns, advantages, fixed_log_probs]]

def iter_prepare(data, start_index, batch_size):
    # Slices the input data into batches of size `batch_size`, starting from `start_index`.
    end_index = min(start_index + batch_size, len(data[0]))
    return [x[start_index:end_index] for x in data]

def update_networks(batch: object, iteration: int, policy_network: Policy, value_network: Value, optimizer_policy: torch.optim.Optimizer, optimizer_value: torch.optim.Optimizer, args: Namespace, dtype: torch.dtype, device: torch.device, optim_epochs: int = 10, optim_batch_size: int = 64) -> object:
    # Process and convert batch data to tensors
    imgs_depth = torch.from_numpy(np.stack(batch.img_depth)).to(dtype).to(device)
    goals = torch.from_numpy(np.stack(batch.goal)).to(dtype).to(device)
    rays = torch.from_numpy(np.stack(batch.ray)).to(dtype).to(device)
    hist_actions = torch.from_numpy(np.stack(batch.hist_action)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    # Compute values and log probabilities
    with torch.no_grad():
        values = value_network(imgs_depth, goals, rays, hist_actions)
        fixed_log_probs = policy_network.get_log_prob(imgs_depth, goals, rays, hist_actions, actions)
    # Estimate advantages
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)
    # Mini-batch PPO update
    optim_iter_num = int(np.ceil(imgs_depth.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        imgs_depth, goals, rays, hist_actions, actions, returns, advantages, fixed_log_probs = epoch_prepare(imgs_depth, goals, rays, hist_actions, actions, returns, advantages, fixed_log_probs, device)
        for i in range(optim_iter_num):
            start_index = i * optim_batch_size
            batch_data = iter_prepare([imgs_depth, goals, rays, hist_actions, actions, returns, advantages, fixed_log_probs], start_index, optim_batch_size)
            (imgs_depth_b, goals_b, rays_b, hist_actions_b, actions_b, returns_b, advantages_b, fixed_log_probs_b) = batch_data
            ppo_step(policy_network, value_network, optimizer_policy, optimizer_value, 1, imgs_depth_b, goals_b, rays_b, hist_actions_b, actions_b, returns_b, advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)
    # Logging the update process (optional)
    if iteration % args.log_interval == 0:
        print(f"Iteration {iteration}: Policy and value networks updated.")

def print_and_save(args, iteration, log, update_start_time, update_end_time, log_eval, eval_end_time, policy_network, value_network, running_state, device):
    print(f"Iteration {iteration}: Printing and saving data.")
    # Print the time and reward statistics
    if iteration % args.log_interval == 0:
        eval_str = f"\teval_R_avg {log_eval['avg_reward']:.2f}" if args.eval_batch_size > 0 else ""
        log_str = f"\tT_eval {eval_end_time - update_end_time:.4f}{eval_str}"
        print(f"{iteration}\tT_sample {log['sample_time']:.4f}\tT_update {update_end_time - update_start_time:.4f}{log_str}\ttrain_R_min {log['min_reward']:.2f}\ttrain_R_max {log['max_reward']:.2f}\ttrain_R_avg {log['avg_reward']:.2f}")
    # Save the training statistics
    filename = f"{args.env_name}_ppo_{'adapt' if args.adaptation else 'rand' if args.randomization else 'norand'}.txt"
    my_open = open(os.path.join(assets_dir(), f"learned_models/{filename}"), "a")
    data = [f"{iteration} {log['avg_reward']} {log['num_episodes']} {log['ratio_success']} {log['avg_steps_success']} {log['avg_last_reward']}\n"]
    my_open.writelines(data)
    my_open.close()
    # Save the learned models
    if args.save_model_interval > 0 and (iteration + 1) % args.save_model_interval == 0:
        to_device(torch.device("cpu"), policy_network, value_network)
        filename = "learned_models/{}_ppo_adapt.p".format(args.env_name) if args.adaptation == 1 else "learned_models/{}_ppo_rand.p".format(args.env_name) if args.randomization == 1 else "learned_models/{}_ppo_norand.p".format(args.env_name)
        pickle.dump((policy_network, value_network, running_state), open(os.path.join(assets_dir(), filename), "wb"))
        to_device(device, policy_network, value_network)
    print(f"Iteration {iteration}: Printing and saving data completed.")