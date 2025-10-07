import torch
from typing import Optional
from io import BytesIO
from PIL import Image
import requests
import os

# If these come from another module, make sure to import them correctly.
# For demonstration, we assume they are available in this script.
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from embodiedbench.aguvis_constants import (
    agent_system_message,
    grounding_system_message,
    chat_template,
    until,
    user_instruction,
)

def load_image(image_file: str) -> Image.Image:
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_pretrained_model(model_path: str, device: str = "cuda"):
    # # 展开用户目录中的 ~ 符号并获取绝对路径
    # model_path = os.path.expanduser(model_path)
    # model_path = os.path.abspath(model_path)
    
    # 使用local_files_only=True来强制从本地加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        # local_files_only=True,
        # trust_remote_code=True
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        # local_files_only=True,
        # trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    model.to(device)
    model.tie_weights()
    return model, processor, tokenizer

def generate_response(
    model,
    processor,
    tokenizer,
    messages: list,
    # image: Optional[Image.Image],
    # instruction: str,
    # previous_actions: Optional[str] = None,
    # low_level_instruction: Optional[str] = None,
    obs: Optional[Image.Image] = None,
    mode: str = "self-plan",
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
):
    # if image is None:
    #     raise ValueError("An image is required for this model pipeline.")

    # system_message = {
    #     "role": "system",
    #     "content": grounding_system_message if mode == "grounding" else agent_system_message,
    # }

    # # If previous_actions is a list, turn it into a single string
    # if isinstance(previous_actions, list):
    #     previous_actions = "\n".join(previous_actions)
    # if not previous_actions:
    #     previous_actions = "None"

    # user_message = {
    #     "role": "user",
    #     "content": [
    #         {"type": "image", "image": image},
    #         {
    #             "type": "text",
    #             "text": user_instruction.format(
    #                 overall_goal=instruction,
    #                 previous_actions=previous_actions,
    #                 low_level_instruction=low_level_instruction,
    #             ),
    #         }
    #     ],
    # }

    # if low_level_instruction:
    #     # If low-level instruction is provided, we enforce using that instruction directly
    #     recipient_text = f"<|im_start|>assistant<|recipient|>all\nAction: {low_level_instruction}\n"
    # elif mode == "grounding":
    #     recipient_text = "<|im_start|>assistant<|recipient|>os\n"
    # elif mode == "self-plan":
    #     recipient_text = "<|im_start|>assistant<|recipient|>"
    # elif mode == "force-plan":
    #     recipient_text = "<|im_start|>assistant<|recipient|>all\nThought: "
    # else:
    #     raise ValueError(f"Invalid mode: {mode}")

    # Prepare input for the model
    # messages = [system_message, user_message]

    messages[1]['content'][0]['image'] = obs

    text = processor.apply_chat_template(
        # messages,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template
    )
    # text += recipient_text
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate
    generated = model.generate(
        **inputs,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )

    # Convert tokens to text
    cont_toks = generated.tolist()[0][len(inputs.input_ids[0]) :]
    text_outputs = tokenizer.decode(cont_toks, skip_special_tokens=False).strip()

    # Truncate at any stop strings in 'until'
    for term in until:
        if term:
            text_outputs = text_outputs.split(term)[0]
    return text_outputs

class AguvisModel:
    """
    Wrapper class that loads the Qwen2VL model and provides a 'respond' method
    that returns the text response given a prompt and an image (obs).
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        # Load the model, processor, and tokenizer once in the constructor
        self.model, self.processor, self.tokenizer = load_pretrained_model(model_path, device)
        self.model.eval()
        self.device = device

    def respond(
        self,
        messages: list,
        # prompt: str,
        # obs: Optional[str] = None,
        # previous_actions: Optional[str] = None,
        # low_level_instruction: Optional[str] = None,
        mode: str = "self-plan",
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Generate a response given a text prompt and an optional image path (obs).

        :param prompt: The high-level instruction or user goal.
        :param obs: Path to an image file (local or a URL).
        :param previous_actions: Any previous actions to feed into the model as context.
        :param low_level_instruction: A forced instruction for the model to execute.
        :param mode: The generation mode ("self-plan", "force-plan", or "grounding").
        :param temperature: Softmax temperature for generation.
        :param max_new_tokens: The maximum number of newly generated tokens.
        :return: The generated response as a string.
        """

        # image = None
        # if obs is not None:
        #     image = load_image(obs)
        obs = messages[1]['content'][0]['image']
        obs = load_image(obs)

        with torch.no_grad():
            response = generate_response(
                model=self.model,
                processor=self.processor,
                tokenizer=self.tokenizer,
                messages=messages,
                obs=obs,
                # image=image,
                # instruction=prompt,
                # previous_actions=previous_actions,
                # low_level_instruction=low_level_instruction,
                mode=mode,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
        return response
