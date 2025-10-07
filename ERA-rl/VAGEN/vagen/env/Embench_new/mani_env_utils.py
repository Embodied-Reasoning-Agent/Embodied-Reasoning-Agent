import re

pick = {
    "small_container0": "container",
    "small_container1": "container",
    "star_normal_visual0": "star",
    "star_normal_visual1": "star",
    "cylinder_normal0": "cylinder",
    "cylinder_normal1": "cylinder",
    "triangular_normal0": "triangular",
    "triangular_normal1": "triangular",
    "cube_basic0": "cube",
    "cube_basic1": "cube",
    "moon_normal_visual0": "moon",
    "moon_normal_visual1": "moon",
}


stack = {
    "star_normal_visual0": "star",
    "star_normal_visual1": "star",
    "star_normal_visual2": "star",
    "star_normal_visual3": "star",
    "cylinder_normal0": "cylinder",
    "cylinder_normal1": "cylinder",
    "cylinder_normal2": "cylinder",
    "cylinder_normal3": "cylinder",
    "triangular_normal0": "triangular",
    "triangular_normal1": "triangular",
    "triangular_normal2": "triangular",
    "triangular_normal3": "triangular",
    "cube_basic0": "cube",
    "cube_basic1": "cube",
    "cube_basic2": "cube",
    "cube_basic3": "cube",
    "moon_normal_visual0": "moon",
    "moon_normal_visual1": "moon",
    "moon_normal_visual2": "moon",
    "moon_normal_visual3": "moon",
}

place = {
    "star_normal_visual0": "star",
    "star_normal_visual1": "star",
    "star_normal_visual2": "star",
    "star_normal_visual3": "star",
    "cylinder_normal0": "cylinder",
    "cylinder_normal1": "cylinder",
    "cylinder_normal2": "cylinder",
    "cylinder_normal3": "cylinder",
    "triangular_normal0": "triangular",
    "triangular_normal1": "triangular",
    "triangular_normal2": "triangular",
    "triangular_normal3": "triangular",
    "cube_basic0": "cube",
    "cube_basic1": "cube",
    "cube_basic2": "cube",
    "cube_basic3": "cube",
    "moon_normal_visual0": "moon",
    "moon_normal_visual1": "moon",
    "moon_normal_visual2": "moon",
    "moon_normal_visual3": "moon",
    "shape_sorter_visual": "shape sorter"
}

wipe = {
    "rectangle": "rectangle area",
    "rectangle0": "rectangle area",
    "round": "round area",
    "round0": "round area",
    "triangle": "triangle area",
    "triangle0": "triangle area",
    "star": "star area",
    "star0": "star area",
    "sponge_visual0": "sponge",
}


manipulation_system_prompt = """## You are a Franka Panda robot with a parallel gripper. You can perform various tasks and output a sequence of gripper actions to accomplish a given task with images of your status. The input space, output action space and color space are defined as follows:

** Input Space **
- Each input object is represented as a 3D discrete position in the following format: [X, Y, Z]. 
- There is a red XYZ coordinate frame located in the top-left corner of the table. The X-Y plane is the table surface. 
- The allowed range of X, Y, Z is [0, 100]. 
- Objects are ordered by Y in ascending order.

** Output Action Space **
- Each output action is represented as a 7D discrete gripper action in the following format: [X, Y, Z, Roll, Pitch, Yaw, Gripper state].
- X, Y, Z are the 3D discrete position of the gripper in the environment. It follows the same coordinate system as the input object coordinates.
- The allowed range of X, Y, Z is [0, 100].
- Roll, Pitch, Yaw are the 3D discrete orientation of the gripper in the environment, represented as discrete Euler Angles. 
- The allowed range of Roll, Pitch, Yaw is [0, 120] and each unit represents 3 degrees.
- Gripper state is 0 for close and 1 for open.

** Color space **
- Each object can be described using one of the colors below:
  ["red", "maroon", "lime", "green", "blue", "navy", "yellow", "cyan", "magenta", "silver", "gray", "olive", "purple", "teal", "azure", "violet", "rose", "black", "white"],

** Generation Guide **
- Include the thinking process between <|think_start|> and <|think_end|>
- Include only the target action in <|action_start|> and <|action_end|>, i.e. the content inside <|action_start|> and <|action_end|> should be nothing more than the 7-DoF vector. Do not include any other thing, such as '"'.

"""

# manipulation_system_prompt = """## You are a Franka Panda robot with a parallel gripper. You can perform various tasks and output a sequence of gripper actions to accomplish a given task with images of your status. The input space, output action space and color space are defined as follows:

# ** Input Space **
# - Each input object is represented as a 3D discrete position in the following format: [X, Y, Z]. 
# - There is a red XYZ coordinate frame located in the top-left corner of the table. The X-Y plane is the table surface. 
# - The allowed range of X, Y, Z is [0, 100]. 
# - Objects are ordered by Y in ascending order.

# ** Output Action Space **
# - Each output action is represented as a 7D discrete gripper action in the following format: [X, Y, Z, Roll, Pitch, Yaw, Gripper state].
# - X, Y, Z are the 3D discrete position of the gripper in the environment. It follows the same coordinate system as the input object coordinates.
# - The allowed range of X, Y, Z is [0, 100].
# - Roll, Pitch, Yaw are the 3D discrete orientation of the gripper in the environment, represented as discrete Euler Angles. 
# - The allowed range of Roll, Pitch, Yaw is [0, 120] and each unit represents 3 degrees.
# - Gripper state is 0 for close and 1 for open.

# ** Color space **
# - Each object can be described using one of the colors below:
#   ["red", "maroon", "lime", "green", "blue", "navy", "yellow", "cyan", "magenta", "silver", "gray", "olive", "purple", "teal", "azure", "violet", "rose", "black", "white"],

# ** Generation Guide **
# - Include the thinking process between <think> and <\think>
# - Include only the target action in <action> and <\action>, i.e. the content inside <action> and <\action> should be nothing more than the 7-DoF vector. Do not include any other thing, such as '"'.

# """

reflection_system_prompt = """
** Importance of self reflection **
You are recommended to reflect on your progress based on interaction history if you have reached the 8th step without success. Specifically, when you find yourself unable to make progress after several attempts, use the keyword "wait" to initiate a reflection process. 
You should reflect on: 
1) Whether you have correctly identified the target objects (should you reconsider and find the correct targets?), and 
2) Whether your manipulation steps are correct (should you derive a new set of manipulation steps?). 

Include the "wait" and reflection process between <|think_start|> and <|think_end|> if you reached the 8th step without success.
"""

def output_to_action(output_text):
    # Extract raw string between think_start and think_end
    think_match = re.search(r"<\|think_start\|\>(.*?)<\|think_end\|\>", output_text, re.DOTALL)
    if think_match:
        try:
            think_text = think_match.group(1).strip()
            think_format_correct = True
        except:
            think_format_correct = False
            think_text = "[No think block found]"
    else:
        think_format_correct = False
        think_text = "[No think block found]"

    # Extract the first action
    action_match = re.search(r"<\|action_start\|\>\[(.*?)\]<\|action_end\|\>", output_text)
    if action_match:
        try:
            action_list = [int(x.strip()) for x in action_match.group(1).split(',')]
            action_format_correct = True
        except:
            action_format_correct = False
            action_list = []
    else:
        action_format_correct = False
        action_list = []

    return action_list, think_text, think_format_correct and action_format_correct


# def output_to_action(output_text):
#     # 提取 think
#     think_match = re.search(r"<think>(.*?)<\\think>", output_text, re.DOTALL)
#     if think_match:
#         try:
#             think_text = think_match.group(1).strip()
#             think_format_correct = True
#         except:
#             think_format_correct = False
#             think_text = "[No think block found]"
#     else:
#         think_format_correct = False
#         think_text = "[No think block found]"

#     # 提取 action（保持原逻辑：不使用 DOTALL，方括号内若换行会匹配失败）
#     action_match = re.search(r"<action>\[(.*?)\]<\\action>", output_text)
#     if action_match:
#         try:
#             action_list = [int(x.strip()) for x in action_match.group(1).split(',')]
#             action_format_correct = True
#         except:
#             action_format_correct = False
#             action_list = []
#     else:
#         action_format_correct = False
#         action_list = []

#     return action_list, think_text, think_format_correct and action_format_correct


def parse_visual_description(think_text):
    """
    from think_text, parse the visual description part, extract the color and type of the object
    
    Args:
        think_text (str): think text
        
    Returns:
        list: [(color, type), ...] list, color could be None
    """
    # 首先查找visual_description部分
    visual_desc_match = re.search(r"visual_description:\s*(.*?)(?:reasoning_and_reflection:|action|$)", 
                                  think_text, re.DOTALL | re.IGNORECASE)
    
    if not visual_desc_match:
        # 如果没有找到visual_description，直接返回空列表
        return []
    
    visual_desc_text = visual_desc_match.group(1).strip()
    
    # 定义颜色和物体类型列表
    colors = ["red", "maroon", "lime", "green", "blue", "navy", "yellow", "cyan", 
              "magenta", "silver", "gray", "olive", "purple", "teal", "azure", 
              "violet", "rose", "black", "white"]
    
    object_types = ["container", "star", "cylinder", "triangular", "cube", "moon", 
                   "shape sorter", "rectangle area", "round area", "triangle area", 
                   "star area", "sponge"]
    
    # 使用原有的解析逻辑：查找 "a/an ... at [coordinates]" 模式
    pattern = r"\b(?:a|an)\s+((?:\w+\s+)*?\w+?)\s+at\s+\[[\d\s,.-]+\]"
    matches = re.findall(pattern, visual_desc_text, re.IGNORECASE)
    
    result = []
    
    for match in matches:
        # split the matched text into words
        words = match.strip().split()
        
        if len(words) == 0:
            continue
        elif len(words) == 1:
            # only one word, no color, this word is the type
            result.append((None, words[0]))
        else:
            # multiple words, find the color word
            color_found = None
            color_index = -1
            
            # find the color word from all the words
            for i, word in enumerate(words):
                if word.lower() in colors:
                    color_found = word
                    color_index = i
                    break
            
            if color_found is not None:
                # found the color, extract the non-color words as the type
                type_words = words[:color_index] + words[color_index+1:]
                object_type = ' '.join(type_words) if type_words else words[0]
                result.append((color_found, object_type))
            else:
                # no color found, all words are the type
                result.append((None, ' '.join(words)))
    
    return result






















# def env_worker(pipe):
#     env = EBManipulationEnv()
#     try:
#         while True:
#             cmd, data = pipe.recv()
#             if cmd == "reset":
#                 print(type(data))
#                 obs, info = env.reset(int(data))  # data is seed
#                 pipe.send((obs, info))
#             elif cmd == "step":
#                 result = env.step(data)  # data is action
#                 pipe.send(result)
#             elif cmd == "close":
#                 env.close()
#                 break
#     except (EOFError, KeyboardInterrupt):
#         env.close()

# class EnvManager:
#     def __init__(self, num_envs=2):
#         self.num_envs = num_envs
#         self.processes = []
#         self.pipes = []
#         self._create_envs()
        
#     def _create_envs(self):
#         for _ in range(self.num_envs):
#             parent_pipe, child_pipe = mp.Pipe()
#             process = mp.Process(target=env_worker, args=(child_pipe,))
#             process.start()
#             self.pipes.append(parent_pipe)
#             self.processes.append(process)
            
#     def reset_env(self, env_id, seed):
#         if not 0 <= env_id < self.num_envs:
#             raise ValueError(f"Invalid environment id: {env_id}")
#         pipe = self.pipes[env_id]
#         pipe.send(("reset", seed))
#         return pipe.recv()
        
#     def step_env(self, env_id, action):
#         if not 0 <= env_id < self.num_envs:
#             raise ValueError(f"Invalid environment id: {env_id}")
#         pipe = self.pipes[env_id]
#         pipe.send(("step", action))
#         return pipe.recv()
        
#     def close_all(self):
#         for pipe in self.pipes:
#             try:
#                 pipe.send(("close", None))
#             except (IOError, OSError):
#                 pass
        
#         for process in self.processes:
#             process.join(timeout=1)
#             if process.is_alive():
#                 process.terminate()
        
#         self.pipes = []
#         self.processes = []
