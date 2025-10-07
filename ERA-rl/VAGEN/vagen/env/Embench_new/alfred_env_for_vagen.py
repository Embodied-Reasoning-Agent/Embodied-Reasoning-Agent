from vagen.env.base.base_env import BaseEnv
from vagen.env.Embench_new.embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
# from vagen.env.Embench_new.prompt_utils_alfred import get_system_prompt
import json
import re
from PIL import Image
import os
from datetime import datetime

alfred_system_prompt = '''## You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions to achieve the desired outcome.

## Action Descriptions and Validity Rules
• Find: Parameterized by the name of the receptacle to navigate to. So long as the object is present in the scene, this skill is always valid
• Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding another object, and the object is not inside a closed receptacle.
• Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is holding an object.
• Drop: Parameterized by the name of the object to put down. It is different from Put down action, as this does not guarantee the held object will be put into a specified receptacle. 
• Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is close to the receptacle.
• Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is close to the receptacle.
• Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is close to the object.
• Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is close to the object.
• Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to the object.


## The available action id (0 ~ {}) and action names are: {}.

## Guidelines
1. **Output Plan**: Avoid generating empty plan. Each plan should include no more than 20 actions.
2. **Visibility**: Always locate a visible object by the 'find' action before interacting with it.
3. **Action Guidelines**: Make sure match the action name and its corresponding action id in the output.\n Avoid performing actions that do not meet the defined validity criteria. For instance, if you want to put object in a receptacle, use 'put down' rather than 'drop' actions. 
4. **Prevent Repeating Action Sequences**: Do not repeatedly execute the same action or sequence of actions.\n Try to modify the action sequence because previous actions do not lead to success.
5. **Multiple Instances**: There may be multiple instances of the same object, distinguished by an index following their names, e.g., Cabinet_2, Cabinet_3. You can explore these instances if you do not find the desired object in the current receptacle.
6. **Reflection on History and Feedback**: Use interaction history and feedback from the environment to refine and improve your current plan.\n If the last action is invalid, reflect on the reason, such as not adhering to action rules or missing preliminary actions, and adjust your plan accordingly.

** Generation Guide **\n    - Include the thinking process between <|think_start|> and <|think_end|>\n    - Include only the target action in <|action_start|> and <|action_end|>, i.e. the content inside <|action_start|> and <|action_end|> should be nothing more than [action_id, 'action_name'], where the action id is an integer and the action name is the corresponding name. Do not include any other thing, such as '\"'.\n
'''

# !!! change back to normal after testing direct RL

# alfred_system_prompt = '''## You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions to achieve the desired outcome.

# ## Action Descriptions and Validity Rules
# • Find: Parameterized by the name of the receptacle to navigate to. So long as the object is present in the scene, this skill is always valid
# • Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding another object, and the object is not inside a closed receptacle.
# • Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is holding an object.
# • Drop: Parameterized by the name of the object to put down. It is different from Put down action, as this does not guarantee the held object will be put into a specified receptacle. 
# • Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is close to the receptacle.
# • Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is close to the receptacle.
# • Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is close to the object.
# • Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is close to the object.
# • Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to the object.


# ## The available action id (0 ~ {}) and action names are: {}.

# ## Guidelines
# 1. **Output Plan**: Avoid generating empty plan. Each plan should include no more than 20 actions.
# 2. **Visibility**: Always locate a visible object by the 'find' action before interacting with it.
# 3. **Action Guidelines**: Make sure match the action name and its corresponding action id in the output.\n Avoid performing actions that do not meet the defined validity criteria. For instance, if you want to put object in a receptacle, use 'put down' rather than 'drop' actions. 
# 4. **Prevent Repeating Action Sequences**: Do not repeatedly execute the same action or sequence of actions.\n Try to modify the action sequence because previous actions do not lead to success.
# 5. **Multiple Instances**: There may be multiple instances of the same object, distinguished by an index following their names, e.g., Cabinet_2, Cabinet_3. You can explore these instances if you do not find the desired object in the current receptacle.
# 6. **Reflection on History and Feedback**: Use interaction history and feedback from the environment to refine and improve your current plan.\n If the last action is invalid, reflect on the reason, such as not adhering to action rules or missing preliminary actions, and adjust your plan accordingly.

# ** Generation Guide **\n    - Include the thinking process between <think> and </think>\n    - Include only the target action in <action> and </action>, i.e. the content inside <action> and </action> should be nothing more than [action_id, 'action_name'], where the action id is an integer and the action name is the corresponding name. Do not include any other thing, such as '\"'.\n
# '''


# def parse(llm_raw_response):
#     pattern = r"<\|action_start\|\>\s*\[\s*(\d+)"
#     m = re.search(pattern, llm_raw_response)
#     if m:
#         print(f"debug: action is {m.group(1)}")
#         return int(m.group(1)), None, None
#     return None, None, None

def output_to_action(output_text):

    # Extract raw string between think_start and think_end
    think_match = re.search(r"<\|think_start\|\>(.*?)<\|think_end\|\>", output_text, re.DOTALL)
    if think_match:
        think_format_correct = True
        think_text = think_match.group(1).strip()
    else:
        think_format_correct = False
        think_text = "[No think block found]"   

    # Extract the first action - support both quoted and unquoted formats
    # First try with quotes (single or double)
    action_match = re.search(r"<\|action_start\|\>\[(\d+),\s*['\"](.*?)['\"]\]<\|action_end\|\>", output_text, re.DOTALL)
    
    if not action_match:
        # If no quotes found, try without quotes
        action_match = re.search(r"<\|action_start\|\>\[(\d+),\s*([^\]]*?)\]<\|action_end\|\>", output_text, re.DOTALL)
    
    if action_match:
        action_format_correct = True
        action_index = int(action_match.group(1))
        action_detail = action_match.group(2).strip()     
        action_content = f"[{action_index}, '{action_detail}']"
    else:
        action_format_correct = False
        action_index = 1
        action_detail = ""
        action_content = "[No action block found]"

    return action_index, action_detail, action_content, think_text, think_format_correct and action_format_correct

# def output_to_action(output_text):

#     # 提取 think
#     think_match = re.search(r"<think>(.*?)<\\think>", output_text, re.DOTALL)
#     if think_match:
#         think_format_correct = True
#         think_text = think_match.group(1).strip()
#     else:
#         think_format_correct = False
#         think_text = "[No think block found]"

#     # 提取第一个 action（先匹配带引号，其次匹配不带引号）
#     action_match = re.search(
#         r"<action>\[(\d+),\s*['\"](.*?)['\"]\]<\\action>", output_text, re.DOTALL
#     )
#     if not action_match:
#         action_match = re.search(
#             r"<action>\[(\d+),\s*([^\]]*?)\]<\\action>", output_text, re.DOTALL
#         )

#     if action_match:
#         action_format_correct = True
#         action_index = int(action_match.group(1))
#         action_detail = action_match.group(2).strip()
#         action_content = f"[{action_index}, '{action_detail}']"
#     else:
#         action_format_correct = False
#         action_index = 1
#         action_detail = ""
#         action_content = "[No action block found]"

#     return action_index, action_detail, action_content, think_text, think_format_correct and action_format_correct

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

def filter_thinking_only_reflection(thinking_text):
    """
    Extract only the reasoning_and_reflection part from thinking text.
    
    Args:
        thinking_text (str): Original thinking text containing visual_description, reasoning_and_reflection, and language_plan
    
    Returns:
        str: Filtered thinking text with only reasoning_and_reflection
    """
    if not thinking_text or thinking_text == "[No think block found]":
        return thinking_text
    
    # Extract reasoning_and_reflection section using regex
    # Pattern matches "reasoning_and_reflection:" followed by content until the next field or end
    pattern = r'reasoning_and_reflection:\s*(.*?)(?=\s+\w+:|$)'
    match = re.search(pattern, thinking_text, flags=re.DOTALL)
    
    if match:
        reflection_content = match.group(1).strip()
        return f"reasoning_and_reflection: {reflection_content}"
    
    # If no reasoning_and_reflection found, return empty
    return ""

def filter_thinking_reflection_and_plan(thinking_text):
    """
    Keep only the reasoning_and_reflection and language_plan parts from thinking text.

    Args:
        thinking_text (str): Original thinking text containing visual_description, reasoning_and_reflection, and language_plan

    Returns:
        str: Filtered thinking text with only reasoning_and_reflection and language_plan
    """
    if not thinking_text or thinking_text == "[No think block found]":
        return thinking_text

    reflection_match = re.search(r'reasoning_and_reflection:\s*(.*?)(?=\s+\w+:|$)', thinking_text, flags=re.DOTALL)
    plan_match = re.search(r'language_plan:\s*(.*?)(?=\s+\w+:|$)', thinking_text, flags=re.DOTALL)

    parts = []
    if reflection_match:
        parts.append(f"reasoning_and_reflection: {reflection_match.group(1).strip()}")
    if plan_match:
        parts.append(f"language_plan: {plan_match.group(1).strip()}")

    if not parts:
        return ""

    filtered = " ".join(parts)
    filtered = re.sub(r'\s+', ' ', filtered).strip()
    return filtered

def seed_to_config(seed):
    if isinstance(seed, str):
        seed = int(seed)
    eval_sets = ['base', 'spatial', 'common_sense', 'complex_instruction', 'visual_appearance', 'long_horizon']
    return eval_sets[seed//100], seed%100 #TODO: check if this is correct

class AlfredEnv(BaseEnv):
    def __init__(self, image_mode='always', image_interval=1, log_dir='logs', *args, **kwargs):
        """
        image_mode: 
            - 'none': 不包含图像
            - 'periodic': 每 image_interval 步提供一次图像
            - 'always': 每步都提供图像
        image_interval: 当 image_mode 为 'periodic' 时，每隔多少步提供一次图像
        log_dir: 日志文件存储目录
        """
        super(AlfredEnv, self).__init__()
        self.env = EBAlfEnv(resolution=500, *args, **kwargs)
        self.system_prompt = ""
        # self.renew_system_prompt()
        self.all_steps_history = []
        self.bufferwithoutthinking = []
        self.bufferwithoutreflection = []
        self.bufferonlyreflection = []
        self.bufferreflectionandplan = []
        self.instruction = self.env.episode_language_instruction
        self.total_reward = 0
        self.gamma = 0.9
        self.history_window = 5
        self.image_mode = image_mode
        self.image_interval = image_interval
        self.step_counter = 0

        self.task_progress = 0
        self.global_step = 0
        # 日志相关
        self.log_dir = log_dir
        self.log_file = None
        os.makedirs(log_dir, exist_ok=True)

        
    def get_system_prompt(self):
        # 将 language_skill_set 转换为包含索引的格式
        indexed_skill_set = [[i, skill] for i, skill in enumerate(self.env.language_skill_set)]
        return alfred_system_prompt.format(len(self.env.language_skill_set)-1, indexed_skill_set)
        
    def should_include_image(self):
        """判断当前步骤是否应该包含图像"""
        if self.image_mode == 'none':
            return False
        elif self.image_mode == 'always':
            return True
        elif self.image_mode == 'periodic':
            return self.step_counter % self.image_interval == 0
        return False

    def step(self, llm_raw_response):
        # print("--------------------------------")
        # print("llm_raw_response: ", llm_raw_response)
        # print("--------------------------------")

        action_id, action_detail, action_content, thinking, format_correct = output_to_action(llm_raw_response)
        self.step_counter += 1

        obs, reward, done, info = self.env.step(action_id, action_detail, llm_raw_response)

        # # 记录本步骤信息到日志文件
        # with open(self.log_file, 'a') as f:
        #     f.write(f"\nStep {self.step_counter}\n")
        #     f.write("-"*30 + "\n")
        #     f.write(f"LLM Raw Response:\n{llm_raw_response}\n\n")
        #     f.write(f"Parsed Action:\n")
        #     f.write(f"- Action ID: {action_id}\n")
        #     f.write(f"- Action Detail: {action_detail}\n")
        #     f.write(f"- Action Content: {action_content}\n")
        #     f.write(f"- Thinking: {thinking}\n")
        #     f.write(f"- Format Correct: {format_correct}\n\n")
        #     # f.write(f"Environment Info:\n{json.dumps(info, indent=2)}\n")
        #     f.write("-"*30 + "\n")

        # TODO: calculate reward
        if self.global_step <= 22:
            
            reward = 0

            # if format_correct:
            #     reward += 0.3

            if info['last_action_success'] < 1:
                reward -= 0.5

            if info['task_progress'] > self.task_progress:
                self.task_progress = info['task_progress']
                reward += 1

            if info["task_success"]:
                reward += 3

        else:
            reward = 0
            if info["task_success"]:
                reward += 3

        # reward = 0
        # if info["task_success"]:
        #     reward += 3

        self.total_reward = self.total_reward * self.gamma + reward

        step_history_entry = {
            "step_id": info['env_step']-1,
            "thinking": thinking,
            "action": action_content,
            "env_feedback": info["env_feedback"]
        }
        self.all_steps_history.append(step_history_entry)
        
        # Create entry for buffer without reflection
        step_history_entry_without_reflection = {
            "step_id": info['env_step']-1,
            "thinking": filter_thinking_without_reflection(thinking),
            "action": action_content,
            "env_feedback": info["env_feedback"]
        }
        step_history_entry_without_thinking = {
            "step_id": info['env_step']-1,
            "action": action_content,
            "env_feedback": info["env_feedback"]
        }
        step_history_entry_only_reflection = {
            "step_id": info['env_step']-1,
            "thinking": filter_thinking_only_reflection(thinking),
            "action": action_content,
            "env_feedback": info["env_feedback"]
        }
        self.bufferwithoutthinking.append(step_history_entry_without_thinking)
        self.bufferwithoutreflection.append(step_history_entry_without_reflection)
        self.bufferonlyreflection.append(step_history_entry_only_reflection)
        step_history_entry_reflection_and_plan = {
            "step_id": info['env_step']-1,
            "thinking": filter_thinking_reflection_and_plan(thinking),
            "action": action_content,
            "env_feedback": info["env_feedback"]
        }
        self.bufferreflectionandplan.append(step_history_entry_reflection_and_plan)
        
        interaction_history = self.all_steps_history.copy()[-1:]
        interaction_history_without_thinking = self.bufferwithoutthinking.copy()[-5:]
        interaction_history_without_reflection = self.bufferwithoutreflection.copy()[-1:]
        interaction_history_only_reflection = self.bufferonlyreflection.copy()[-1:]
        interaction_history_reflection_and_plan = self.bufferreflectionandplan.copy()[-1:]

        # 决定是否在prompt中包含图像
        should_include_image = self.should_include_image()
        user_prompt = (
                        ("<image>\n" if should_include_image else "") +
                        "instruction: " + self.instruction + " \n " +
                        'interaction_history:' + str(interaction_history_reflection_and_plan) + " \n " +
                        # "Your response should include visual description of the current scene, reasoning and reflection, language plan and action." +
                        "Based on the above information, please provide the action for the next step to complete the task. Think, then act."
                    )

        image = Image.fromarray(obs['head_rgb'])
        metrics = {
            "turn_metrics": {
                "task_progress": info["task_progress"],
                "task_success": info["task_success"],
            },
            "traj_metrics": {
                "task_success": info["task_success"],
            }
        }
        
        # 根据配置决定是否包含图像数据
        multi_modal_data = {"<image>": [image]} if should_include_image else {}
        
        obs = {
            "obs_str": user_prompt,
            "multi_modal_data": multi_modal_data
        }
        info = {
            "metrics": metrics,
            "llm_raw_response": llm_raw_response
        }

        return obs, reward, done, info
        
    def close(self):
        self.env.close()
        
    def reset(self, seed, global_step):

        self.global_step = global_step

        self.task_progress = 0

        eval_set, episode_id = seed_to_config(seed)
        image = Image.fromarray(self.env.reset(eval_set, episode_id)['head_rgb'])

        # 创建新的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"{timestamp}_env{eval_set}_{episode_id}.log")
        
        # 记录初始信息
        with open(self.log_file, 'w') as f:
            f.write(f"Episode Start Time: {timestamp}\n")
            f.write(f"Eval Set: {eval_set}\n")
            f.write(f"Episode ID: {episode_id}\n")
            f.write(f"Instruction: {self.instruction}\n")
            f.write("="*50 + "\n\n")

        self.all_steps_history = []
        self.bufferwithoutreflection = []
        self.bufferwithoutthinking = []
        self.bufferonlyreflection = []
        self.bufferreflectionandplan = []
        self.instruction = self.env.episode_language_instruction
        self.total_reward = 0
        self.step_counter = 0

        # 在reset时，如果不是'none'模式，则显示第一张图片
        should_include_image = self.image_mode != 'none'
        user_prompt = (
                        ("<image>\n " if should_include_image else "") + 
                        "instruction: " + self.instruction + " \n " +
                        "interaction_history: []\n" +
                        # "Your response should include visual description of the current scene, reasoning and reflection, language plan and action." +
                        "Based on the above information, please provide the action for the next step to complete the task. Think, then act."
                    )
                    
        # 根据配置决定是否包含图像数据
        multi_modal_data = {"<image>": [image]} if should_include_image else {}
        
        obs = {
            "obs_str": user_prompt,
            "multi_modal_data": multi_modal_data
        }
        metrics = {
            "turn_metrics": {
                "task_progress": 0,
                "task_success": False,
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
    
    def system_prompt(self):
        return self.system_prompt
    
    def compute_reward(self):
        return self.total_reward
    
    # def renew_system_prompt(self):
    #     episode_num = self.env._current_episode_num
    #     eval_set = self.env._eval_set
    #     self.system_prompt = get_system_prompt(eval_set, episode_num)

if __name__ == "__main__":
    env = AlfredEnv()
    env.reset(100)