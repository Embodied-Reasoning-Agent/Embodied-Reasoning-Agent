import json

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# System Message
grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
alfred_system_message = "\n## You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions to\nachieve the desired outcome.\n## Action Descriptions and Validity Rules\n• Find: Parameterized by the name of the receptacle to navigate to. So long as the object is present in the scene, this\nskill is always valid.\n• Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding\nanother object, and the object is not inside a closed receptacle.\n• Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is\nholding an object.\n• Drop: Parameterized by the name of the object to put down. It is different from the Put down action, as this does\nnot guarantee the held object will be put into a specified receptacle.\n• Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is\nclose to the receptacle.\n• Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is\nclose to the receptacle.\n• Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is\nclose to the object.\n• Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is\nclose to the object.\n• Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to\nthe object.\n\n## You are supposed to output in json. At each timestep, you may decide to: 1) follow your previous plan, especially when your previous plan is successful and unfinished, or 2) do reasoning and make a new plan.\nFor reasoning, you need to output a reasoning message, in which you should describe the current visual state from the image, output your reasoning steps, and plan. \nAt the end, you should output an action message, which should include the action id from the available actions to execute and its corresponding description.\n"

# Chat Template
chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

# assistant_template = "{% for message in messages %}{{'<|im_start|>' + message['role']}}{% if 'recipient' in message %}<|recipient|>{{ message['recipient'] }}{% endif %}{{'\n' + message['content'][0]['text']}}{% if 'end_turn' in message and message['end_turn'] %}{{'<|diff_marker|>\n'}}{% else %}{{'<|im_end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|recipient|>' }}{% endif %}"

assistant_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] }}{{ '\n' }}{% if message['think_or_action'] == 0 %}{{ '<|think_start|>' + message['content'][0]['text'] + '<|think_end|>' }}{% elif message['think_or_action'] == 1 %}{{ '<|action_start|>' + message['content'][0]['text'] + '<|action_end|>' }}{% else %}{{ message['content'][0]['text'] }}{% endif %}{% if 'end_turn' in message and message['end_turn'] %}{{ '<|diff_marker|>\n' }}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant' }}{% endif %}"


# Special Tokens
additional_special_tokens = [
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|diff_marker|>",
    "<|think_start|>",
    "<|think_end|>",
    "<|action_start|>",
    "<|action_end|>"
]

# Plugin Functions
select_option_func = {
    "name": "browser.select_option",
    "description": "Select an option from a dropdown menu",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the dropdown menu",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the dropdown menu",
            },
            "value": {
                "type": "string",
                "description": "The value of the option to select",
            },
        },
        "required": ["x", "y", "value"],
    },
}

swipe_func = {
    "name": "mobile.swipe",
    "description": "Swipe on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "from_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The starting coordinates of the swipe",
            },
            "to_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The ending coordinates of the swipe",
            },
        },
        "required": ["from_coord", "to_coord"],
    },
}

home_func = {"name": "mobile.home", "description": "Press the home button"}

back_func = {"name": "mobile.back", "description": "Press the back button"}

wait_func = {
    "name": "mobile.wait",
    "description": "wait for the change to happen",
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "The seconds to wait",
            },
        },
        "required": ["seconds"],
    },
}

long_press_func = {
    "name": "mobile.long_press",
    "description": "Long press on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the long press",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the long press",
            },
        },
        "required": ["x", "y"],
    },
}

open_app_func = {
    "name": "mobile.open_app",
    "description": "Open an app on the device",
    "parameters": {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "The name of the app to open",
            },
        },
        "required": ["app_name"],
    },
}

agent_system_message = f"""You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

You have access to the following functions:
- {json.dumps(swipe_func)}
- {json.dumps(home_func)}
- {json.dumps(back_func)}
- {json.dumps(wait_func)}
- {json.dumps(long_press_func)}
- {json.dumps(open_app_func)}
"""

user_instruction = """Please generate the next move according to the ui screenshot, instruction and previous actions.

Instruction: {overall_goal}

Previous actions:
{previous_actions}
"""

until = ["<|diff_marker|>"]
