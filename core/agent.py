import multiprocessing
from utils.replay_memory import Memory
from utils.torchpy import *
from utils.tools import *
import math
import time
import os
import sys
import signal
from core.env import UnderwaterNavigation

os.environ["OMP_NUM_THREADS"] = "1"

class Agent:
    """
    The Agent class represents an agent that interacts with an environment using a policy.

    Args:
        env (object): The environment object.
        policy (object): The policy object.
        device (object): The device object.
        custom_reward (callable, optional): A custom reward function. Defaults to None.
        running_state (object, optional): The running state object. Defaults to None.
        num_threads (int, optional): The number of worker threads. Defaults to 1.
        training (bool, optional): Whether the agent is in training mode. Defaults to True.
    """

    def __init__(self, env, policy, device, custom_reward=None, running_state=None, num_threads=1, training=True):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.running_state = running_state
        self.num_threads = num_threads
        self.training = training

    def collect_samples(self, min_batch_size, mean_action=False, render=False):
        """
        Collects samples from the environment using the agent's policy.

        Args:
            min_batch_size (int): The minimum batch size of samples to collect.
            mean_action (bool, optional): Whether to use the mean action. Defaults to False.
            render (bool, optional): Whether to render the environment. Defaults to False.

        Returns:
            tuple: A tuple containing the collected batch of samples and a log dictionary.
        """
        if min_batch_size == 0:
            return None, None
        t_start = time.time()
        to_device(torch.device("cpu"), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = self._start_workers(queue, thread_batch_size, mean_action)
        memory, log = samples(0, None, self.env[0], self.policy, self.custom_reward, mean_action, render, self.running_state, thread_batch_size, training=self.training)
        worker_results = [queue.get() for _ in workers]
        worker_memories, worker_logs = zip(*[(res[1], res[2]) for res in worker_results])
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log = self._merge_log([log] + list(worker_logs))
        to_device(self.device, self.policy)
        t_end = time.time()
        log["sample_time"] = t_end - t_start
        log.update(self._calculate_action_stats(batch))
        return batch, log

    def _start_workers(self, queue, thread_batch_size, mean_action):
        """
        Starts the worker threads for collecting samples.

        Args:
            queue (object): The multiprocessing queue.
            thread_batch_size (int): The batch size for each worker thread.
            mean_action (bool): Whether to use the mean action.

        Returns:
            list: A list of worker processes.
        """
        workers = []
        for i in range(self.num_threads - 1):
            worker_args = (
                i + 1,
                queue,
                self.env[i + 1],
                self.policy,
                self.custom_reward,
                mean_action,
                False,  # render is always False for worker threads
                self.running_state,
                thread_batch_size,
                self.training,
            )
            worker = multiprocessing.Process(target=samples, args=worker_args)
            workers.append(worker)
            worker.start()
        return workers

    def _calculate_action_stats(self, batch):
        """
        Calculates statistics of the actions in the batch.

        Args:
            batch (object): The batch of samples.

        Returns:
            dict: A dictionary containing the action statistics.
        """
        actions = np.vstack(batch.action)
        return {
            "action_mean": np.mean(actions, axis=0),
            "action_min": np.min(actions, axis=0),
            "action_max": np.max(actions, axis=0)
        }

    def _merge_log(self, log_list):
        """
        Merges the log dictionaries from multiple worker threads.

        Args:
            log_list (list): A list of log dictionaries.

        Returns:
            dict: A merged log dictionary.
        """
        log = dict()
        log["total_reward"] = sum([x["total_reward"] for x in log_list])
        log["num_episodes"] = sum([x["num_episodes"] for x in log_list])
        log["num_steps"] = sum([x["num_steps"] for x in log_list])
        log["avg_reward"] = log["total_reward"] / log["num_episodes"]
        log["max_reward"] = max([x["max_reward"] for x in log_list])
        log["min_reward"] = min([x["min_reward"] for x in log_list])
        if "total_c_reward" in log_list[0]:
            log["total_c_reward"] = sum([x["total_c_reward"] for x in log_list])
            log["avg_c_reward"] = log["total_c_reward"] / log["num_steps"]
            log["max_c_reward"] = max([x["max_c_reward"] for x in log_list])
            log["min_c_reward"] = min([x["min_c_reward"] for x in log_list])
        return log

def samples(pid, queue, env: UnderwaterNavigation, policy, custom_reward, mean_action, render, running_state, min_batch_size, training=True):
    """
    Collects a batch of experiences from the environment using the given policy.

    Args:
        pid (int): The process ID.
        queue (Queue): A multiprocessing queue to store the collected experiences.
        env (gym.Env): The environment to collect experiences from.
        policy (Policy): The policy to use for selecting actions.
        custom_reward (function): A function to calculate the custom reward.
        mean_action (bool): Whether to use the mean action or sample from the policy.
        render (bool): Whether to render the environment.
        running_state (RunningStat): The running state of the environment.
        min_batch_size (int): The minimum number of experiences to collect.
        training (bool): Whether to update the policy.

    Returns:
        tuple: A tuple containing the collected experiences and the log data.
    """
    initialize_env(env, pid)
    log, memory = {}, Memory()
    num_episodes, num_steps, num_episodes_success, num_steps_episodes, total_reward, total_c_reward, reward_done = 0, 0, 0, 0, 0, 0, 0
    min_reward, max_reward, min_c_reward, max_c_reward = 1e6, -1e6, 1e6, -1e6
    while num_steps < min_batch_size:
        state = process_state(*env.reset(), running_state)
        reward_episode = 0
        for t in range(10000):
            signal.signal(signal.SIGINT, signal_handler)
            action = select_action(policy, *state, mean_action)
            next_state, reward, done = step_environment(env, action, running_state)
            reward, total_c_reward, min_c_reward, max_c_reward, reward_episode, reward_done, num_episodes_success, num_steps_episodes = process_reward(reward, *state, action, custom_reward, total_c_reward, min_c_reward, max_c_reward, reward_episode, done, reward_done, t, num_episodes_success, num_steps_episodes)
            next_img_depth, next_goal, _, next_hist_action = next_state
            memory.push(*state, action, 0 if done else 1, next_img_depth, next_goal, next_hist_action, reward)
            if render: env.render()
            if done: break
            state = next_state
        num_steps += t + 1
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)
        if not training: write_reward(reward_episode, num_episodes)
    log = update_log(custom_reward, log, num_episodes, num_steps, num_episodes_success, num_steps_episodes, total_reward, total_c_reward, reward_done, min_reward, max_reward, min_c_reward, max_c_reward)
    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log
    
def signal_handler(sig, frame):
    sys.exit(0)

def initialize_env(env, pid):
    """
    Initializes the environment with a random seed based on the process ID.

    Args:
        env (gym.Env): The environment to initialize.
        pid (int): The process ID.

    Returns:
        int: The environment seed.
    """
    if pid > 0:
        torch.manual_seed(torch.randint(0, 5000, (1,)) * pid) # seed torch with a random seed based on the process ID

        # seed numpy with a random seed based on the process ID
        if hasattr(env, "np_random"):
            env.np_random.seed(env.np_random.randint(5000) * pid)
        if hasattr(env, "env") and hasattr(env.env, "np_random"):
            env.env.np_random.seed(env.env.np_random.randint(5000) * pid)

def process_state(img_depth, goal, ray, hist_action, running_state):
    """
    Preprocesses the input state for the agent.

    Args:
        img_depth (numpy.ndarray): The depth image of the current state.
        goal (numpy.ndarray): The goal position of the agent.
        ray (numpy.ndarray): The ray vector of the agent.
        hist_action (numpy.ndarray): The previous action taken by the agent.
        running_state (function): A function that transforms the input state.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: The preprocessed state.
    """
    if running_state is not None:
        _, goal, ray = running_state(img_depth, goal, ray)
        img_depth = np.float64((img_depth - 0.5) / 0.5)
        hist_action = np.float64(hist_action)
    else:
        img_depth, goal, ray, hist_action = (
            img_depth.astype(np.float64),
            goal.astype(np.float64),
            ray.astype(np.float64),
            hist_action.astype(np.float64),
        )
    return img_depth, goal, ray, hist_action

def select_action(policy: Policy, img_depth, goal, ray, hist_action, mean_action):
    """
    Selects an action based on the given policy and input data.

    Args:
        policy (MyPolicy): The policy to use for selecting the action.
        img_depth (numpy.ndarray): The depth image input.
        goal (numpy.ndarray): The goal input.
        ray (numpy.ndarray): The ray input.
        hist_action (numpy.ndarray): The historical action input.
        mean_action (bool): Whether to use the mean action from the policy.

    Returns:
        int or float: The selected action.
    """
    img_depth_var = tensor(img_depth).unsqueeze(0)
    goal_var = tensor(goal).unsqueeze(0)
    ray_var = tensor(ray).unsqueeze(0)
    hist_action_var = tensor(hist_action).unsqueeze(0)
    with torch.no_grad():
        if mean_action:
            # Use the mean action from the policy
            action = policy(img_depth_var, goal_var, ray_var, hist_action_var)[0][0].numpy()
        else:
            # Select action using policy (for stochastic policies)
            action = policy.select_action(img_depth_var, goal_var, ray_var, hist_action_var)[0].numpy()
    action = int(action) if policy.is_disc_action else action.astype(np.float64)
    return action

def step_environment(env: UnderwaterNavigation, action, running_state):
    """
    Takes a step in the environment using the given action and returns the next state, reward, and done flag.

    Args:
        env (gym.Env): The environment to step in.
        action (int): The action to take in the environment.
        running_state (RunningState): The running state of the agent.

    Returns:
        Tuple containing:
            - next_img_depth (np.ndarray): The next image depth array.
            - next_goal (np.ndarray): The next goal array.
            - next_ray (np.ndarray): The next ray array.
            - next_hist_action (np.ndarray): The next history action array.
            - reward (float): The reward received from the environment.
            - done (bool): Whether the episode is done or not.
    """
    next_img_depth, next_goal, next_ray, next_hist_action, reward, done, _ = env.step(action)
    next_img_depth, next_goal, next_ray, next_hist_action = process_state(next_img_depth, next_goal, next_ray, next_hist_action, running_state)
    next_state = (next_img_depth, next_goal, next_ray, next_hist_action)
    return next_state, reward, done

def write_reward(reward, num_episodes):
    """
    Write the given reward to a file and exit the program if the number of episodes
    is greater than or equal to 5.

    Args:
        reward (float): The reward to write to the file.
        num_episodes (int): The number of episodes that have been completed.

    Returns:
        None
    """
    asset_dir = path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))
    with open(os.path.join(asset_dir, "learned_models/test_rewards.txt"), "a") as f:
        f.write(f"{reward}\n\n")
    if num_episodes >= 5:
        sys.exit()

def process_reward(reward, img_depth, goal, ray, action, hist_action, custom_reward, total_c_reward, min_c_reward, max_c_reward, reward_episode, done, reward_done, t, num_episodes_success, num_steps_episodes):
    """
    Processes the reward obtained by the agent after taking an action in the environment.

    Args:
        reward (float): The reward obtained by the agent after taking an action.
        img_depth (numpy.ndarray): The depth image of the current observation.
        goal (numpy.ndarray): The goal position in the environment.
        ray (numpy.ndarray): The ray casted from the agent to the goal position.
        action (numpy.ndarray): The action taken by the agent.
        hist_action (numpy.ndarray): The previous action taken by the agent.
        custom_reward (function): A custom reward function to be used instead of the default reward.
        total_c_reward (float): The total custom reward obtained by the agent.
        min_c_reward (float): The minimum custom reward obtained by the agent.
        max_c_reward (float): The maximum custom reward obtained by the agent.
        reward_episode (float): The total reward obtained by the agent in the current episode.
        done (bool): Whether the episode has ended or not.
        reward_done (float): The total reward obtained by the agent in the current episode if it has ended.
        t (int): The current timestep in the episode.
        num_episodes_success (int): The number of episodes in which the agent has reached the goal.
        num_steps_episodes (int): The total number of timesteps taken by the agent in the episodes in which it has reached the goal.

    Returns:
        Tuple[float, float, float, float, float, float, int, int]: A tuple containing the updated values of reward, total_c_reward, min_c_reward, max_c_reward, reward_episode, reward_done, num_episodes_success, and num_steps_episodes.
    """
    if custom_reward is not None:
        reward = custom_reward(img_depth, goal, ray, action)
        total_c_reward += reward
        min_c_reward = min(min_c_reward, reward)
        max_c_reward = max(max_c_reward, reward)
    reward_episode += reward
    if done:
        reward_done += reward
        if reward > 0 and t < 499:
            num_episodes_success += 1
            num_steps_episodes += t
    return reward, total_c_reward, min_c_reward, max_c_reward, reward_episode, reward_done, num_episodes_success, num_steps_episodes

def update_log(custom_reward, log, num_episodes, num_steps, num_episodes_success, num_steps_episodes, total_reward, total_c_reward, reward_done, min_reward, max_reward, min_c_reward, max_c_reward):
    """
    Update the log with the given parameters.

    Args:
        custom_reward (float): The custom reward.
        log (dict): The log to update.
        num_episodes (int): The number of episodes.
        num_steps (int): The number of steps.
        num_episodes_success (int): The number of successful episodes.
        num_steps_episodes (int): The number of steps per episode.
        total_reward (float): The total reward.
        total_c_reward (float): The total custom reward.
        reward_done (float): The reward for the last episode.
        min_reward (float): The minimum reward.
        max_reward (float): The maximum reward.
        min_c_reward (float): The minimum custom reward.
        max_c_reward (float): The maximum custom reward.

    Returns:
        dict: The updated log.
    """
    log["num_steps"] = num_steps
    log["num_episodes"] = num_episodes
    log["total_reward"] = total_reward
    log["avg_reward"] = total_reward / num_episodes
    log["max_reward"] = max_reward
    log["min_reward"] = min_reward
    log["num_episodes"] = num_episodes
    log["ratio_success"] = float(num_episodes_success) / float(num_episodes)
    log["avg_last_reward"] = reward_done / num_episodes
    if num_episodes_success != 0:
        log["avg_steps_success"] = float(num_steps_episodes) / float(num_episodes_success)
    else:
        log["avg_steps_success"] = 0
    if custom_reward is not None:
        log["total_c_reward"] = total_c_reward
        log["avg_c_reward"] = total_c_reward / num_steps
        log["max_c_reward"] = max_c_reward
        log["min_c_reward"] = min_c_reward
    return log