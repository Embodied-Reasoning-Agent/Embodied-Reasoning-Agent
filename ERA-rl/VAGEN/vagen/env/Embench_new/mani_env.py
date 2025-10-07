# from vagen.env.base.base_env import BaseEnv
from vagen.env.Embench_new.embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
# from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
import json
import re
from PIL import Image
import copy
import numpy as np

from vagen.env.Embench_new.embodiedbench.envs.eb_manipulation.eb_man_utils import form_object_coord_for_input

from vagen.env.Embench_new.embodiedbench.evaluator.config.eb_manipulation_example import vlm_examples_baseline

from vagen.env.Embench_new.mani_env_utils import *


import os
os.environ['DISPLAY'] = ':1'

def filter_thinking_without_reflection(thinking_text):
    """
    Remove the reasoning_and_reflection part from thinking text while keeping other fields.
    
    Args:
        thinking_text (str): Original thinking text containing visual_description, reasoning_and_reflection, and language_plan
    
    Returns:
        str: Filtered thinking text without reasoning_and_reflection
    """
    if not thinking_text or thinking_text == "[No think block found]":
        return thinking_text
    
    # Remove reasoning_and_reflection section using regex
    # Pattern matches "reasoning_and_reflection:" followed by any content until the next field or end
    pattern = r'reasoning_and_reflection:\s*.*?(?=\s+\w+:|$)'
    filtered_thinking = re.sub(pattern, '', thinking_text, flags=re.DOTALL).strip()
    
    # Clean up any extra whitespace
    filtered_thinking = re.sub(r'\s+', ' ', filtered_thinking).strip()
    
    return filtered_thinking


def seed_to_config(seed):
    if isinstance(seed, str):
        seed = int(seed)
    # eval_sets = ['base', 'common_sense', 'complex', 'spatial', 'visual']
    return seed//100, seed%100 #TODO: check if this is correct

class EBManipulationEnv():
    def __init__(self, *args, **kwargs):

        # super(EBManipulationEnv, self).__init__()
        self.env = EBManEnv(render_mode='human', *args, **kwargs)
        self.system_prompt = ""
        # self.renew_system_prompt()
        self.all_steps_history = []
        self.bufferwithoutreflection = []
        self.bufferwithoutthinking = []
        self.instruction = self.env.episode_language_instruction
        self.total_reward = 0
        self.gamma = 0.9
        self.camera_views = ['front_rgb']
        
        # 1. for target object reward
        self.target_objects_approached = {}
        
        # 2. for visual reward
        self.last_observation = {}
        self.object2property = {}  # {object_name: (object_type, object_color)}
        self.parsed_visual_description = []  # Store parsed visual description from think text
        self.color_info = {}
        self.task_class = ""
        self.global_step = 0
        
    def get_system_prompt(self):
        return manipulation_system_prompt
        #.format("Example: " + vlm_examples_baseline[self.env.current_task_variation.split('_')[0]][0])
        
        
    def reset(self, seed, global_step): 

        self.global_step = global_step

        eval_set, episode_id = seed_to_config(seed)
        _, observation, self.task_class, self.color_info = self.env.reset(eval_set, episode_id)

        self.last_observation = observation
        
        # 1. Create tracking dictionary
        self.target_objects_approached = self._extract_target_objects(observation)
        # print(self.target_objects_approached)
        
        # 2. Create object to property mapping for visual perception reward
        self.object2property = self._create_object2property_mapping(observation)
        # print("Object to property mapping:", self.object2property)
        
        image = Image.fromarray(observation['front_rgb'])
        
        # self.renew_system_prompt()
        self.all_steps_history = []
        self.bufferwithoutreflection = []
        self.bufferwithoutthinking = []
        self.instruction = self.env.episode_language_instruction
        camera_views = ['front_rgb']
        avg_obj_coord, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list = form_object_coord_for_input(copy.deepcopy(observation), self.env.task_class, camera_views)
        self.avg_obj_coord = avg_obj_coord
        self.total_reward = 0
        user_prompt = (
                        "<image>\n " + "instruction: " + self.instruction + " \n " +
                        "interaction_history: []\n" +
                        "additional_info:" + str(avg_obj_coord) + "\n" +
                        # "Your response should include visual description of the current scene, reasoning and reflection, language plan and action." +
                        "Based on the above information, please provide the action for the next step to complete the task. Think, then act."
                    )
        obs = {
            "obs_str": user_prompt,
            "multi_modal_data": {
                "<image>": [image]
            }
        }
        metrics = {
            "turn_metrics": {
                "task_success": False,
                # "visual_accuracy": 0.0,
            },
            "traj_metrics": {
                "task_success": False,
            }
        }

        info = {
            "metrics": metrics,
            "llm_raw_response": None,
            "llm_response": None
        }
        return obs, info

    def step(self, llm_raw_response):

        action_list, think_text, format_correct = output_to_action(llm_raw_response)

        obs, reward, done, info = self.env.step(action_list)

        avg_obj_coord, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list = form_object_coord_for_input(copy.deepcopy(obs), self.env.task_class, self.camera_views)

        if self.global_step <= 20:
            # 0. Calculate regular reward
            reward = 0
            # if format_correct:
            #     reward += 0.5

            if info["task_success"]:
                reward += 3

            # 1. target object reward
            target_approached = self._check_target_objects_approached(obs)

            if target_approached:
                reward += 1

            visual_accuracy = 0
            self.parsed_visual_description = parse_visual_description(think_text)
            if self.parsed_visual_description: # if the visual description is not empty
                correct_visual_description = self._get_objects_left_to_right(self.last_observation)
                visual_accuracy = self._check_visual_perception_accuracy(correct_visual_description, self.parsed_visual_description)
                if visual_accuracy > 0.75:
                    reward += 0.5
                if visual_accuracy < 0.25:
                    reward -= 0.5
            else:
                reward += 0

        else:
            reward = 0
            if info["task_success"]:
                reward += 3

            visual_accuracy = 0 # for consistency

        # 2. visual reward
        # visual_accuracy = None
        # self.parsed_visual_description = parse_visual_description(think_text)
        # # print("Parsed visual description:", self.parsed_visual_description)
        # if self.parsed_visual_description: # if the visual description is not empty
        #     correct_visual_description = self._get_objects_left_to_right(self.last_observation)
        #     # print("Correct visual description:", correct_visual_description)
        #     visual_accuracy = self._check_visual_perception_accuracy(correct_visual_description, self.parsed_visual_description)
        #     # if visual_accuracy > 0.8:  # High accuracy threshold
        #     #     reward += 3
        #     # elif visual_accuracy > 0.5:  # Medium accuracy threshold
        #     #     reward += 1
        #     if visual_accuracy > 0.5:
        #         reward += 1
        #     elif visual_accuracy < 0.2:
        #         reward -= 1
        # else:
        #     reward += 0

        self.total_reward = self.total_reward * self.gamma + reward

        self.last_observation = obs

        step_history_entry = {
            "step_id": info['env_step'], #TODO: check if this is correct!!!!! should start from 0
            "thinking": think_text,
            "action": action_list,
            "env_feedback": info["env_feedback"]
        }
        self.all_steps_history.append(step_history_entry)
        
        # Create entry for buffer without reflection
        step_history_entry_without_reflection = {
            "step_id": info['env_step'],
            "thinking": filter_thinking_without_reflection(think_text),
            "action": action_list,
            "env_feedback": info["env_feedback"]
        }
        step_history_entry_without_thinking = {
            "step_id": info['env_step'],
            "action": action_list,
            "env_feedback": info["env_feedback"]
        }
        self.bufferwithoutthinking.append(step_history_entry_without_thinking)
        self.bufferwithoutreflection.append(step_history_entry_without_reflection)
        
        interaction_history = self.all_steps_history.copy()[-1:]
        interaction_history_without_thinking = self.bufferwithoutthinking.copy()[-3:]
        interaction_history_without_reflection = self.bufferwithoutreflection.copy()[-1:]


        user_prompt = (
                    "<image>\n"  +
                    "instruction: " + self.instruction + " \n " +
                    'interaction_history:' + str(interaction_history_without_thinking) + " \n " +
                    "additional_info:" + str(avg_obj_coord) + "\n" +
                    # "Your response should include visual description of the current scene, reasoning and reflection, language plan and action." +
                    "Based on the above information, please provide the action for the next step to complete the task. Think, then act."
                )

        image = Image.fromarray(obs['front_rgb'])
        metrics = {
            "turn_metrics": {
                "task_success": info["task_success"],
                "visual_accuracy": 0 #visual_accuracy
            },
            "traj_metrics": {
                "task_success": info["task_success"],
            }
        }
        obs = {
            "obs_str": user_prompt,
            "multi_modal_data": {
                "<image>": [image]
            }
        }
        info = {
            "metrics": metrics,
            "llm_raw_response": llm_raw_response
            # "llm_response": "" #Not using it currently, but we follow the original format to avoid errors
        }

        return obs, reward, done, info



    
    def system_prompt(self):
        return self.system_prompt
    
    def compute_reward(self):
        return self.total_reward

    def close(self):
        self.env.close()



    def _get_objects_left_to_right(self, observation):
        """
        Get objects ordered from left to right based on y coordinate (smaller y is left)
        Args:
            observation (dict): Environment observation data
        Returns:
            list: List of (object_name, object_type, object_color) tuples ordered left to right
        """
        if 'object_informations' not in observation:
            return []
            
        # Collect objects with their y coordinates
        objects_with_y = []
        for obj_name, obj_info in observation['object_informations'].items():
            if obj_name in self.object2property and isinstance(obj_info, dict) and 'pose' in obj_info:
                try:
                    pose = obj_info['pose']
                    if isinstance(pose, str):
                        # Parse string format "[x y z qx qy qz qw]"
                        pose_values = pose.strip('[]').split()
                        y_coord = float(pose_values[1])
                    else:
                        y_coord = float(pose[1])
                    
                    object_type, object_color = self.object2property[obj_name]
                    objects_with_y.append((y_coord, obj_name, object_type, object_color))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing pose for {obj_name}: {e}")
                    continue
        
        # Sort by y coordinate (left to right)
        objects_with_y.sort(key=lambda x: x[0])
        
        # Return list of (object_type, object_color) tuples
        return [(obj_type, obj_color) for _, _, obj_type, obj_color in objects_with_y]
    
    def _check_visual_perception_accuracy(self, current_scene_objects, parsed_objects):
        """
        Check visual perception accuracy by comparing current scene objects with parsed objects
        Args:
            current_scene_objects (list): List of (object_type, object_color) tuples from left to right
            parsed_objects (list): List of (object_type, object_color) tuples from visual description
        Returns:
            float: Accuracy score (0.0 to 1.0)
        """
        
        # Simple accuracy: check if the objects match in order
        correct_matches = 0
        correct_length = len(current_scene_objects)
        
        for i in range(min(len(current_scene_objects), len(parsed_objects))):
            current_obj = current_scene_objects[i]
            parsed_obj = parsed_objects[i]
            
            # Check if type matches (color can be None in parsed)
            type_match = current_obj[0] == parsed_obj[1]  # Compare object types
            color_match = parsed_obj[0] is None or current_obj[1] == parsed_obj[0]  # Color can be None
            
            if type_match: # and color_match
                correct_matches += 1
        
        return correct_matches / correct_length

    def _check_target_objects_approached(self, observation):
        """
        Check if any target objects were approached in the given observation
        Args:
            observation (dict): Environment observation data
        Returns:
            bool: True if any target objects were approached, False otherwise
        """
        # Get gripper pose (first 3 coordinates: x, y, z)
        if 'gripper_pose' not in observation:
            return False
            
        try:
            gripper_pose = observation['gripper_pose']
            if isinstance(gripper_pose, str):
                # Parse string format like "[x y z roll pitch yaw gripper_state]"
                gripper_values = gripper_pose.strip('[]').split()
                gripper_coords = np.array([float(gripper_values[0]), float(gripper_values[1]), float(gripper_values[2])])
            else:
                gripper_coords = np.array(gripper_pose[:3])
        except (ValueError, IndexError) as e:
            print(f"Error parsing gripper pose: {e}")
            return False
        
        # Check each target object that hasn't been approached yet
        if 'object_informations' not in observation:
            return False
            
        for target_obj_name, approached in self.target_objects_approached.items():
            if approached == 0:  # Only check objects that haven't been approached
                # Find this target object in object_informations
                if target_obj_name in observation['object_informations']:
                    obj_info = observation['object_informations'][target_obj_name]
                    if isinstance(obj_info, dict) and 'pose' in obj_info:
                        try:
                            obj_pose = obj_info['pose']
                            if isinstance(obj_pose, str):
                                # Parse string format "[x y z qx qy qz qw]"
                                obj_values = obj_pose.strip('[]').split()
                                obj_coords = np.array([float(obj_values[0]), float(obj_values[1]), float(obj_values[2])])
                            else:
                                obj_coords = np.array(obj_pose[:3])
                            
                            # Calculate L2 norm of the difference
                            distance = np.linalg.norm(gripper_coords - obj_coords)
                            
                            # Debug print for distance checking
                            # print(f"Checking {target_obj_name}: gripper at {gripper_coords}, object at {obj_coords}, distance: {distance:.4f}")
                            
                            if distance <= 0.2:
                                # Mark this object as approached
                                self.target_objects_approached[target_obj_name] = 1
                                # print(f"Target object '{target_obj_name}' was approached! Distance: {distance:.4f}")
                                return True
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing pose for {target_obj_name}: {e}")
                            continue

        # print("No target objects were approached")
        return False

    def _create_object2property_mapping(self, observation):
        """
        Create object to property mapping based on observation
        Args:
            observation (dict): Environment observation data
        Returns:
            dict: {object_name: (object_type, object_color)}
        """
        object2property = {}
        
        # Get task-specific object dict
        task_dict = None
        if self.task_class == "pick":
            task_dict = pick
        elif self.task_class == "stack":
            task_dict = stack
        elif self.task_class == "place":
            task_dict = place
        elif self.task_class == "wipe":
            task_dict = wipe
        
        if task_dict is None:
            print(f"Warning: Unknown task class {self.task_class}")
            return object2property
        
        # Traverse object_informations
        if 'object_informations' not in observation:
            return object2property
            
        for obj_name, obj_info in observation['object_informations'].items():
            if obj_name in task_dict:
                object_type = task_dict[obj_name]
                object_color = self.color_info.get(obj_name, "unknown")
                object2property[obj_name] = (object_type, object_color)
                
        return object2property

    def _extract_target_objects(self, observation):
        """
        Extract all unique target objects from observation and create tracking dictionary
        Args:
            observation (dict): Environment observation data
        Returns:
            dict: {target_object_name: 0/1} dictionary, initial values are all 0
        """
        target_objects = set()
        
        # Traverse all waypoints in object_informations
        if 'object_informations' in observation:
            for key, value in observation['object_informations'].items():
                if key.startswith('waypoint') and isinstance(value, dict):
                    if 'target_obj_name' in value:
                        target_objects.add(value['target_obj_name'])
        
        # Create tracking dictionary, initial values are all 0 (not approached)
        target_objects_dict = {obj_name: 0 for obj_name in target_objects}
        
        return target_objects_dict



if __name__ == "__main__":

    env = EBManipulationEnv()
    obs, info = env.reset(seed=0)
    env.env.save_image()
    
    # Test step with a dummy action to see if target object checking works
    dummy_action = """
    <|think_start|>
    visual_description: I can see a purple cylinder at [10, 20, 30], a white container at [40, 50, 60], a teal cube at [70, 80, 90], a silver container at [15, 25, 35], a red star at [45, 55, 65], a maroon moon at [75, 85, 95], and a teal triangular at [20, 30, 40].
    
    reasoning_and_reflection: Based on the instruction to pick up the star and place it into the blue container, I need to first approach the red star.
    <|think_end|>
    <|action_start|>[20,30, 20, 10, 20, 10, 0]<|action_end|>
    """
    obs, reward, done, info = env.step(dummy_action)
    env.env.save_image()
    print(f"After step: reward={reward}, target_objects={env.target_objects_approached}")
    
    env.close()
    print("Test completed!")
