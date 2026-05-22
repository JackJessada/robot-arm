"""Weighted denoising reward: R = 0.1*r8 + 0.2*r9 + 0.7*r10."""

import multiprocessing as mp
import os
import torch
import numpy as np
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv


def execute_and_score(raw_env, actions_np):
    """
    Execute an action chunk in a LIBERO OffScreenRenderEnv and return binary success {0, 1}.
    """
    done = False
    success = 0
    for act in actions_np:
        if done:
            break
        try:
            obs, reward, done, info = raw_env.step(act)
        except ValueError:
            break
        if info.get("success", False):
            success = 1
            break
    return success


# Global worker state
_WORKER_ENV = None
_WORKER_SUITE = None


def init_worker(task_suite_name):
    """Initialize LIBERO environment once per worker process."""
    global _WORKER_ENV, _WORKER_SUITE
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "osmesa")
    bd = benchmark.get_benchmark_dict()
    _WORKER_SUITE = bd[task_suite_name]()
    # Dummy task_id for init, will be changed per request
    _WORKER_ENV = LiberoEnv(
        task_suite=_WORKER_SUITE,
        task_id=0,
        task_suite_name=task_suite_name,
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
    )


def compute_single_reward_worker(args):
    """Worker function to compute reward for one trajectory."""
    task_id, init_state_idx, actions_8_np, actions_9_np, actions_10_np = args
    global _WORKER_ENV, _WORKER_SUITE

    # 1. Switch task if needed (LiberoEnv handles this by recreating raw env if id changes)
    if _WORKER_ENV.task_id != task_id:
        _WORKER_ENV.close()
        _WORKER_ENV = LiberoEnv(
            task_suite=_WORKER_SUITE,
            task_id=task_id,
            task_suite_name=_WORKER_ENV.task_suite_name,
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
        )

    # 2. Reset to specific init state
    # We use init_state_idx to ensure all workers start from the same seed
    init_states = _WORKER_SUITE.get_task_init_states(task_id)
    raw_env = _WORKER_ENV._env
    raw_env.reset()
    raw_env.set_init_state(init_states[init_state_idx % len(init_states)])
    
    # 3. Stabilize (match LiberoEnv.reset logic)
    for _ in range(10):
        raw_env.step([0.0] * 7)
    raw_env.use_delta = True

    # 4. Save stabilized state
    sim_state = raw_env.sim.get_state()
    saved_timestep = raw_env.env.timestep

    def run_horizon(actions_np):
        raw_env.sim.set_state(sim_state)
        raw_env.sim.forward()
        raw_env.env.timestep = saved_timestep
        raw_env.env.done = False
        raw_env.env._obs_cache = {}
        return execute_and_score(raw_env, actions_np)

    r8  = run_horizon(actions_8_np)
    r9  = run_horizon(actions_9_np)
    r10 = run_horizon(actions_10_np)

    return 0.1 * r8 + 0.2 * r9 + 0.7 * r10


class ParallelRewardRunner:
    def __init__(self, task_suite_name, n_workers=8):
        self.n_workers = n_workers
        self.pool = mp.Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(task_suite_name,)
        )

    def compute_batch_rewards(self, task_id, init_state_idx, group_data, postprocessor):
        """
        Compute rewards for a whole GRPO group in parallel.
        """
        # Pre-denormalize actions on main process (with GPU)
        tasks = []
        for traj in group_data:
            with torch.no_grad():
                a8_np  = postprocessor(traj["actions_8"]).squeeze(0).cpu().numpy()
                a9_np  = postprocessor(traj["actions_9"]).squeeze(0).cpu().numpy()
                a10_np = postprocessor(traj["actions_10"]).squeeze(0).cpu().numpy()
            
            tasks.append((task_id, init_state_idx, a8_np, a9_np, a10_np))

        return self.pool.map(compute_single_reward_worker, tasks)

    def close(self):
        self.pool.close()
        self.pool.join()


def compute_weighted_reward(raw_env, group_trajectory, postprocessor):
    """
    Keep original function for backward compatibility (non-parallel evaluation).
    """
    sim_state = raw_env.sim.get_state()
    saved_timestep = raw_env.env.timestep

    def run_horizon(actions_tensor):
        raw_env.sim.set_state(sim_state)
        raw_env.sim.forward()
        raw_env.env.timestep = saved_timestep
        raw_env.env.done = False
        raw_env.env._obs_cache = {}

        actions_denorm = postprocessor(actions_tensor)
        actions_np = actions_denorm.squeeze(0).cpu().float().numpy()
        return execute_and_score(raw_env, actions_np)

    r8  = run_horizon(group_trajectory["actions_8"])
    r9  = run_horizon(group_trajectory["actions_9"])
    r10 = run_horizon(group_trajectory["actions_10"])

    return 0.1 * r8 + 0.2 * r9 + 0.7 * r10
