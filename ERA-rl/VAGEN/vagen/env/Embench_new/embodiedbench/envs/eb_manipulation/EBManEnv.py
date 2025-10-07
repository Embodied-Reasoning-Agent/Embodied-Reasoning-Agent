#Modified From the rlbench: https://github.com/stepjam/RLBench
from typing import Union, Dict, Tuple
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.errors import ConfigurationError, ConfigurationPathError
import gymnasium as gym
from gymnasium import spaces
from amsolver.environment import Environment
from amsolver.action_modes import ArmActionMode, ActionMode
from amsolver.observation_config import ObservationConfig
from amsolver.task_environment import TTMS_FOLDER   
import numpy as np
from amsolver.backend.utils import task_file_to_task_class
from pathlib import Path
from amsolver.utils import name_to_task_class
from vagen.env.Embench_new.embodiedbench.envs.eb_manipulation.eb_man_utils import get_continous_action_from_discrete
import os
import time
from PIL import Image
from vagen.env.Embench_new.embodiedbench.main import logger

EVAL_SETS = {
    'base': ['pick_cube_shape', 'stack_cubes_color', 'place_into_shape_sorter_color', 'wipe_table_direction'],
    'common_sense': ['pick_cube_shape', 'stack_cubes_color', 'place_into_shape_sorter_color', 'wipe_table_direction'],
    'complex': ['pick_cube_shape', 'stack_cubes_color', 'place_into_shape_sorter_color', 'wipe_table_direction'],
    'spatial': ['pick_cube_relative', 'stack_cubes_relative', 'place_into_shape_sorter_relative', 'wipe_table_relative'],
    'visual': ['pick_cube_shape', 'stack_cubes_color', 'place_into_shape_sorter_color']
}

ValidEvalSets = ['base', 'common_sense', 'complex', 'spatial', 'visual']

class EBManEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode='human', img_size=(500, 500), log_path = None):

        obs_config = ObservationConfig()
        obs_config.set_all(True)
        obs_config.set_image_size(img_size)

        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)        
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        self._render_mode = render_mode

        # Load dataset for all eval sets
        self.task_class = None
        # print("start load dataset")
        self.dataset = self._load_dataset(None, None, None)  # 参数在新的实现中不再使用
        # print("dataset loaded")
        # print(len(self.dataset[0]))
        self.task = None
        self.current_task_variation = None

        # Episode tracking
        # self.number_of_episodes = len(self.dataset)
        self._reset = False
        self._current_episode_num = 0
        self._current_step = 0
        
        # change for era testing
        self._max_episode_steps = 20

        self._episode_start_time = 0
        self.episode_log = []

        # Task-related attributes
        self.episode_language_instruction = ''
        self.episode_data = None
        self.last_frame_obs = None

        self.action_space = spaces.Box(
            low=0.0, high=100.0, shape=(self.env.action_size,))

        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

        if log_path is None:
            self.log_path = 'running/eb_manipulation/all'
        else:
            self.log_path = log_path
    
    def load_test_config(self, data_folder, task_name):
        episode_list = []
        for path in data_folder.rglob('configs*'):
            t_name = path.parents[3].name
            if t_name == task_name:
                episode_list.append(path.parent)

        def extract_variation_episode(path):
            import re
            match = re.search(r'variation(\d+)/episodes/episode(\d+)', str(path))
            if match:
                variation = int(match.group(1))
                episode = int(match.group(2))
                return (variation, episode)
            return (float('inf'), float('inf'))

        episode_list = sorted(episode_list, key=extract_variation_episode)
        return episode_list
    
    def _load_dataset(self, eval_tasks, data_folder, task_files):
        dataset = {}
        for eval_set_idx, eval_set in enumerate(ValidEvalSets):
            dataset[eval_set_idx] = []
            current_task_files = EVAL_SETS[eval_set]
            current_eval_tasks = [task_file_to_task_class(t, parent_folder = 'vlm') for t in current_task_files]
            current_data_folder = Path(os.path.join(TTMS_FOLDER, f'data/{eval_set}/eval')) # "../env/Embench_new/embodiedbench/envs/eb_manipulation
            for i, task_to_use in enumerate(current_eval_tasks):
                e_path = self.load_test_config(current_data_folder, current_task_files[i])
                for num, e in enumerate(e_path):
                    task_base = str(e/"task_base.ttm")
                    waypoint_sets = str(e/"waypoint_sets.ttm")
                    config = str(e/"configs.pkl")
                    dataset[eval_set_idx].append((task_to_use, task_base, waypoint_sets, config, current_task_files[i]))
        return dataset
    
    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        return {
            "state": obs.get_low_dim_data(),
            "left_shoulder_rgb": obs.left_shoulder_rgb,
            "right_shoulder_rgb": obs.right_shoulder_rgb,
            "wrist_rgb": obs.wrist_rgb,
            "front_rgb": obs.front_rgb,
            "overhead_rgb": obs.overhead_rgb
        }

    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            return self._gym_cam.capture_rgb()

    def reset(self, eval_set_idx=0, episode_id=None):
        """
        Reset the environment for a new episode.
        
        Args:
            eval_set_idx (int): Index of the evaluation set (0-4)
            episode_id (int, optional): Specific episode ID to reset to. If None, uses current episode counter.
        
        Returns:
            observation
        """
        if eval_set_idx not in self.dataset:
            raise ValueError(f"Invalid eval_set_idx: {eval_set_idx}. Must be between 0-4.")
            
        if episode_id is not None:
            if episode_id >= len(self.dataset[eval_set_idx]):
                raise ValueError(f"episode_id {episode_id} exceeds dataset length {len(self.dataset[eval_set_idx])}")
            self._current_episode_num = episode_id
        else:
            assert self._current_episode_num <= len(self.dataset[eval_set_idx]), "All episodes have been completed."
            self._current_episode_num += 1

        self._current_step = 0
        self._reset = True
        self._episode_log = []
        self._episode_start_time = time.time()

        # print(len(self.dataset[eval_set_idx]))
        
        current_data = self.dataset[eval_set_idx][self._current_episode_num]
        self.task = self.env.get_task(current_data[0])
        self.current_task_variation = current_data[-1]
        self.task_class = self.current_task_variation.split('_')[0]
        self.task_file = current_data[-1]
        descriptions, obs, color_info = self.task.load_config(current_data[1], current_data[2], current_data[3])
        self.episode_language_instruction = descriptions[0]
        self.last_frame_obs = vars(obs)
        
        return descriptions[0], vars(obs), self.task_class, color_info
    
    def step(self, discrete_action):
        assert self._reset, "Reset the environment before stepping."
        info = {}
        self._current_step += 1
        action_success = False
        try:
            action = get_continous_action_from_discrete(discrete_action)
            obs, reward, terminate = self.task.step(action)
            if self.current_task_variation.startswith('stack'):
                if terminate:
                    if action[-1] == 0.0:
                        reward = 0.0
                        terminate = False
                        logger.debug("wrong success condition for stack, setting reward to 0 and terminate to False ...")
                    elif action[-1] == 1.0:
                        action[2] += 0.03
                        logger.debug("checking if the object is stacked properly ...")
                        obs, reward, terminate = self.task.step(action)
                        if terminate and reward == 1.0:
                            logger.debug("stacking is successful ...")
                            reward = 1.0
                            terminate = True
                        else:
                            logger.debug("stacking is unsuccessful ...")
                            reward = 0.0
                            terminate = False
            self.last_frame_obs = vars(obs)
            action_success = True
        except Exception as e:
            print(f"*** An unexpected error occurred: {e}")
            obs, reward, terminate = self.last_frame_obs, -1, False
            action_success = e
        env_feedback = self.get_env_feedback(action_success, reward)
        info['env_feedback'] = env_feedback
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
        info['episode_num'] = self._current_episode_num
        info['action'] = discrete_action
        if action_success == True:
            info['action_success'] = 1.0
        else:
            info['action_success'] = 0.0
        if terminate and reward == 1.0:
            info['task_success'] = 1.0
        else:
            info['task_success'] = 0.0
        if self._current_step >= self._max_episode_steps:
            # print(f"self._current_step: {self._current_step}")
            # print(f"self._max_episode_steps: {self._max_episode_steps}")
            terminate = True
        self.episode_log.append(info)

        return self.last_frame_obs, reward, terminate, info

    def close(self) -> None:
        self.env.shutdown()
    
    def save_image(self, key=['front_rgb']) -> str:
        log_path = self.log_path + '/images/' + f"episode_{self._current_episode_num}"
        if not os.path.exists(log_path):
            os.makedirs(log_path) 
        image_path_list=[]
        for cam_view in key:
            single_image = Image.fromarray(self.last_frame_obs[cam_view])
            time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
            image_path = 'episode_{}_step_{}_{}.png'.format(self._current_episode_num, self._current_step, cam_view)
            image_path = os.path.join(log_path, 'episode_{}_step_{}_{}.png'.format(self._current_episode_num, self._current_step, cam_view))
            single_image.save(image_path)
            image_path_list.append(image_path)
        return image_path_list
    
    def get_env_feedback(self, task_success, reward=None):
        """
        Generate feedback message for the current step.
        Args:
            info (dict): Action execution information
        Returns:
            str: Descriptive message about step outcome
        """
        msg = ''
        if task_success == True:
            msg += 'Last action was successful.'
        else:
            msg += 'Last action was not successful.'
        return msg
         
if __name__ == '__main__':
    """
    Example usage of the EB-Manipulation environment.
    Demonstrates environment interaction with random actions.
    """
    import os, sys
    os.environ['DISPLAY'] = ':1'

    print("\n=== 检查Python路径 ===")
    print("Python路径:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    # 检查amsolver包的源码路径
    print("\n=== 检查amsolver包信息 ===")
    try:
        import amsolver
        print(f"amsolver包路径: {amsolver.__file__}")
        print(f"amsolver包版本: {getattr(amsolver, '__version__', '未知')}")
        
        # 检查主要模块的路径
        modules_to_check = [
            'amsolver.environment',
            'amsolver.action_modes', 
            'amsolver.observation_config',
            'amsolver.task_environment',
            'amsolver.backend.utils',
            'amsolver.utils'
        ]
        
        for module_name in modules_to_check:
            try:
                module = __import__(module_name, fromlist=[''])
                print(f"{module_name}: {module.__file__}")
            except ImportError as e:
                print(f"{module_name}: 导入失败 - {e}")
                
    except ImportError as e:
        print(f"无法导入amsolver包: {e}")
        
    # 检查Environment类的具体信息
    print("\n=== 检查Environment类信息 ===")
    try:
        from amsolver.environment import Environment
        import inspect
        
        print(f"Environment类文件: {inspect.getfile(Environment)}")
        print(f"Environment类模块: {Environment.__module__}")
        
        # 列出Environment类的主要方法
        methods = [method for method in dir(Environment) if not method.startswith('_')]
        print(f"Environment类的主要方法: {methods[:10]}...")  # 只显示前10个方法
        
    except Exception as e:
        print(f"检查Environment类时出错: {e}")


    test_env = EBManEnv(render_mode='human')
    description, obs, task_class, color_info = test_env.reset()
    test_env.save_image()
    print("testing the EB-Manipulation environment ...")
    print("ignore errors like could not create path or target is outside of workspace as actions are randomly sampled ...")
    # for _ in range(3):
    #     action = test_env.action_space.sample()
    #     action[-1] = 1.0
    #     obs, reward, terminate, info = test_env.step(action)
    #     test_env.save_image()
    test_env.close()
    print("testing completed!")