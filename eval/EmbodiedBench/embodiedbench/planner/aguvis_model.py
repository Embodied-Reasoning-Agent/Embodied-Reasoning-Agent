import torch
from typing import Optional
from io import BytesIO
from PIL import Image
import requests
import os
import time

# If these come from another module, make sure to import them correctly.
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from embodiedbench.aguvis_constants import (
    agent_system_message,
    grounding_system_message,
    chat_template,
    until,
    user_instruction,
)
from peft import PeftModel
from transformers import AutoModelForCausalLM

def load_image(image_file: str) -> Image.Image:
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def get_model_device(model):
    """Get the device of the first parameter in the model"""
    return next(model.parameters()).device

def load_pretrained_model(model_path: str, device: str = "cuda"):
    # Load base model with auto device mapping
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        #ignore_mismatched_sizes=True
        # device_map="auto"  
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    tokenizer = processor.tokenizer
    model.resize_token_embeddings(len(tokenizer))

    new_eos_id = tokenizer.convert_tokens_to_ids("<|diff_marker|>")
    model.generation_config.eos_token_id = new_eos_id
    model.generation_config.im_end_id    = new_eos_id
    model.tie_weights()
    
    return model, processor, tokenizer

def generate_response(
    model,
    processor,
    tokenizer,
    messages: list,
    obs: Optional[Image.Image] = None,
    mode: str = "self-plan",
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
):
    # Prepare input for the model
    messages[1]['content'][0]['image'] = obs

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    # Get the device of the model's first parameter
    model_device = get_model_device(model)
    print(f"Moving inputs to model device: {model_device}")
    
    # Move inputs to the same device as the model
    inputs = inputs.to(model_device)

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
        
        # Print device mapping for debugging
        print("Model device mapping:")
        if hasattr(self.model, 'hf_device_map'):
            print(self.model.hf_device_map)

    def respond(
        self,
        messages: list,
        mode: str = "self-plan",
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Generate a response given a text prompt and an optional image path (obs).
        """

        obs = messages[1]['content'][0]['image']
        obs = load_image(obs)

        with torch.no_grad():
            start = time.time()
            response = generate_response(
                model=self.model,
                processor=self.processor,
                tokenizer=self.tokenizer,
                messages=messages,
                obs=obs,
                mode=mode,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
        return response
