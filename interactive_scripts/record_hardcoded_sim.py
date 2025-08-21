import pyrallis
import argparse
from envs.common_mj_env import MujocoEnvConfig
import numpy as np
import os
from dataset_utils.annotate_modes import relabel_demo, load_demo
import pickle
import pdb
import time

def wait_for_file_complete(path, timeout=35):
    start = time.time()
    last_size = -1

    while time.time() - start < timeout:
        if os.path.exists(path):
            current_size = os.path.getsize(path)
            if current_size > 0 and current_size == last_size:
                return  # File exists and is stable
            last_size = current_size
        time.sleep(0.1)

    raise TimeoutError(f"Timed out waiting for file to be fully written: {path}")

def generate_cube_positions(env_name):
    def noise(x_range, y_range):
        return np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            0
        ])
    if env_name == "cube_wbc_longhorizon.yaml":
        base_pos2 = np.array([1.0,  0.4, 0.0]) 
        goal_pos2 = np.array([1.0,  0.08, 0.0])
        noise2 = np.array([
            np.random.uniform(low=-0.2, high=0.2),  
            np.random.uniform(low=-0.2, high=0.2)    
        ])
        noise_g2 = np.array([
            np.random.uniform(low=-0.05, high=0.05),  
            np.random.uniform(low=-0.05, high=0.05)    
        ])

        cube2 = base_pos2.copy()
        cube2[:2] += noise2
        goal2 = goal_pos2.copy()
        goal2[:2] += noise_g2

        return [cube2, goal2]
    if env_name == "cube_wbc_longhorizon_2cubes.yaml":
        cube1 = np.array([1.0,  0.4, 0.0]) + noise((-0.05, 0.05), (-0.2, 0.2))
        cube2 = np.array([1.0,  -0.4, 0.0]) + noise((-0.05, 0.05), (-0.2, 0.2))
        goal1 = np.array([1.0,  0.08, 0.0]) + noise((-0.05, 0.05), (-0.05, 0.05))
        goal2 = np.array([1.0,  -0.08, 0.0]) + noise((-0.05, 0.05), (-0.05, 0.05))

        return [cube1, goal1, cube2, goal2]

    if env_name == "drawer.yaml":
        cube1 = np.array([1.3, 0.2, 0.75]) + noise((-0.05, 0.05), (-0.2, 0.2))
        cube2 = np.array([1.3, -0.2, 0.75]) + noise((-0.05, 0.05), (-0.2, 0.2))
        goal1 = np.array([1.15, 0.1, 0.7]) + noise((-0.05, 0.05), (-0.05, 0.05))
        goal2 = np.array([1.15, -0.1, 0.7]) + noise((-0.05, 0.05), (-0.05, 0.05))

        return [cube1, goal1, cube2, goal2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/cube_wbc_longhorizon_2cubes.yaml")
    args = parser.parse_args()
    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))
    env_name = args.env_cfg.split('/')[-1]
    demo_dir = 'dev1'
    relabel_dir = 'dev1_relabeled'

    if env_cfg.wbc:
        from envs.mj_env_wbc import MujocoEnv
    else:
        from envs.mj_env_base_arm import MujocoEnv

    while True:
        # Re-randomize per episode
        cube_positions = generate_cube_positions(env_name)
        env = MujocoEnv(env_cfg, cube_positions, show_images=False)
        env.reset()
        annotations, episode_fn = env.hardcoded_episode(cube_positions, env_name)

        wait_for_file_complete(episode_fn)

        demo = load_demo(episode_fn)
        demo_relabeled = relabel_demo(demo, annotations)


        output_path = episode_fn.replace(demo_dir, relabel_dir)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(demo_relabeled, f)

        print('episode length: %d, max steps: %d' % (env.num_step, env.max_num_step))
        env.close()
