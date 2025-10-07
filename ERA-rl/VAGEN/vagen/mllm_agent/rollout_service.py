from typing import List, Union, Optional, Dict
import copy
from collections import defaultdict
import torch
import numpy as np
from transformers import PreTrainedTokenizer, ProcessorMixin
from dataclasses import dataclass, field
import PIL
import re

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import process_image, collate_fn
import vagen.env
from vagen.env import REGISTERED_ENV
from vagen.server.client import BatchEnvClient

import math

chat_template = """{% for message in messages -%}
{% if message.role == "system" or message.role == "user" -%}
<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n
{% elif message.role == "assistant" -%}
<|im_start|>{{ message.role }}\n{{ message.content }}\n
{% endif %}
{%- endfor %}"""
    
class QwenVLRolloutManagerService():
    def __init__(self,
                 actor_rollout_wg,
                 config,
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 split="train",
                 ):
        self.split=split
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.recorder= None # defaultdict(list) env_id:record
        self.envs = None # dict env_id:env_config_instance
        self.system_prompts = None # dict env_id:str
        self.env_states = None # dict
        self.batch_idx_to_env_id = None # dict
        self.env_client = BatchEnvClient(base_url=self.config.base_url,timeout=self.config.timeout,max_workers=self.config.max_workers)

    @torch.no_grad()
    def _handle_special_tokens(self, llm_raw_response: str, prep_for_loss_mask: bool) -> str:
        """
        1. Filter out special tokens: <image> and special tokens marking environment observation in the llm generated response
        2. prep_for_loss_mask: if true, add special tokens to the beginning and end of the response if compute_loss_mask is True
        """
        llm_raw_response = re.sub(r'<image>', '', llm_raw_response)
        if prep_for_loss_mask:
            sptk_b = self.config.special_token_for_loss_mask[0]
            sptk_e = self.config.special_token_for_loss_mask[1]
            llm_raw_response = llm_raw_response.replace(sptk_e, '')
            llm_raw_response = llm_raw_response.replace(sptk_b, '')
            llm_raw_response = sptk_b + llm_raw_response + sptk_e
        return llm_raw_response
    
    @torch.no_grad()
    def _handle_multi_modal_data(
            self, 
            prompt_template: str, 
            row_dict: Dict,
            image_data: List[PIL.Image.Image],
            do_embedding: bool = True,
        ) -> str:
        """Handle multi-modal data in the prompt template

        - For do_embedding=False(vllm), replace <image> with <|vision_start|><|image_pad|><|vision_end|> -> raw_prompt
        - For do_embedding=True, replace <image> with <|vision_start|>{image_token}<|vision_end|> -> prompt_template
            - where {image_token} is the length of image embedding
        """
        assert len(image_data) == prompt_template.count('<image>'), 'Number of images does not match number of <image> in the prompt template'
        raw_prompt = prompt_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        row_dict['multi_modal_data'] = {'image': image_data}
        image_grid_thw = None
        if do_embedding:
            image_inputs = self.processor.image_processor(image_data, return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            # print(f"[DEBUG] number of image_data in rollout: {len(image_data)}")
        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            index = 0
            while '<image>' in prompt_template:
                prompt_template = prompt_template.replace(
                    '<image>',
                    '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                    '<|vision_end|>',
                    1,
                )
                index += 1

            prompt_template = prompt_template.replace('<|placeholder|>',
                                                        self.processor.image_token)
            # print(f"[DEBUG] number of image_data in final trajectory: {len(image_data)}")
            # number_of_image_tokens=prompt_template.count(self.processor.image_token)
            # print(f"[DEBUG] number_of_image_tokens: {number_of_image_tokens}")
        return prompt_template, row_dict, image_grid_thw, raw_prompt
    
    @torch.no_grad()
    def _compute_loss_mask(self, input_ids, attention_mask):
        """
        Compute loss mask for the input ids and attention mask
        We only do loss for the tokens in input_ids that are wrapped by special tokens (by defualt they're <|box_start|> and <|box_end|>)
        
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
    
        Returns:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            loss_mask: (batch_size, seq_len) # e.g. 0000|1111|0000|11111|000|1111
            end_of_response_position_mask: (batch_size, seq_len) # e.g. 0000|0001|0000|00001|000|0001 given the end of sequence mask, mark the position of the last token in the response
        
        - There will be different stratgy to handel special tokens in the input_ids
        - 1. remove them, in this case we need to fill the hole by adding pad in the right and shift the sequence left
        - 2. keep them, attention mask will be 0 for them
        - 3. Replace them with pad token
    
        Let's use the 3rd strategy for now
        Compute loss mask for the input ids and attention mask by:
        1. Removing special tokens
        2. Adding padding on the right
        3. Shifting the sequence left
        """
        
        # Get token IDs for special tokens and pad token
        sptk_b = self.tokenizer.convert_tokens_to_ids(self.config.special_token_for_loss_mask[0])
        sptk_e = self.tokenizer.convert_tokens_to_ids(self.config.special_token_for_loss_mask[1])
        pad_token_id = self.tokenizer.pad_token_id

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Initialize output tensors with same shape as inputs
        new_input_ids = input_ids.clone()
        new_attention_mask = attention_mask.clone()
        loss_mask = torch.zeros_like(new_attention_mask)
        new_loss_mask = torch.zeros_like(new_attention_mask)
        end_of_response_position_mask = torch.zeros_like(new_attention_mask)
        new_end_of_response_position_mask = torch.zeros_like(new_attention_mask)
        # Process each example in the batch
        for b in range(batch_size):
            # Count right padding tokens using attention mask
            right_pad_tokens = (new_input_ids[b] == pad_token_id).sum().item()
            
            # Assert that initial padding tokens have attention mask of 0
            if not torch.all(attention_mask[b, -right_pad_tokens:] == 0):
                print("[DEBUG]: right padding tokens must have attention mask of 0")
            
            # Find special token indices
            sptk_b_indices = (input_ids[b] == sptk_b).nonzero().flatten()
            sptk_e_indices = (input_ids[b] == sptk_e).nonzero().flatten()
            
            # Create a mask for tokens that should compute loss
            hole_pos=[] # initialize holes position list with last padding token position
            for start_pos, end_pos in zip(sptk_b_indices, sptk_e_indices):
                loss_mask[b][start_pos+1:end_pos] = 1
                end_of_response_position_mask[b][end_pos-1] = 1
                hole_pos.append(start_pos.item())
                hole_pos.append(end_pos.item())
            hole_pos.append(seq_len-right_pad_tokens)
            # assert new_input_ids[b][seq_len-right_pad_tokens]==pad_token_id
            if not torch.all(new_input_ids[b][seq_len-right_pad_tokens:] == pad_token_id):
                print("[DEBUG]: right padding tokens must be pad token")
            
            # shift right to fill the wholes
            holes_to_fill=1
            for i in range(0,len(hole_pos)-1):
                start_pos = hole_pos[i]
                end_pos = hole_pos[i+1]
                new_loss_mask[b,start_pos+1-holes_to_fill:end_pos-holes_to_fill]=loss_mask[b,start_pos+1:end_pos]
                new_end_of_response_position_mask[b,start_pos+1-holes_to_fill:end_pos-holes_to_fill]=end_of_response_position_mask[b,start_pos+1:end_pos]
                new_input_ids[b,start_pos+1-holes_to_fill:end_pos-holes_to_fill]=input_ids[b,start_pos+1:end_pos]
                new_attention_mask[b,start_pos+1-holes_to_fill:end_pos-holes_to_fill]=attention_mask[b,start_pos+1:end_pos]
                holes_to_fill+=1

            valid_tokens = seq_len-right_pad_tokens-len(hole_pos)+1 # the number of non-special tokens and non-padding tokens
            new_loss_mask[b][valid_tokens:]=0
            new_input_ids[b][valid_tokens:]=pad_token_id
            new_attention_mask[b][valid_tokens:]=0
        
        return new_input_ids, new_attention_mask, new_loss_mask, new_end_of_response_position_mask
    
    @torch.no_grad()
    def _get_rollout_horizon(self, global_steps: int) -> int:
        """
        Calculates the effective rollout horizon based on curriculum learning stages.

        Args:
            global_steps (int): The current global training step.

        Returns:
            int: The effective maximum number of turns for the rollout.
        """
        # Configuration parameters for curriculum learning
        # These parameters should be passed in via self.config, with sensible defaults.
        total_actor_training_steps = getattr(self.config, 'total_actor_training_steps', 20)
        num_curriculum_stages = getattr(self.config, 'num_curriculum_stages', 4) # Default to 4 stages
        critic_warmup_steps = getattr(self.config, 'critic_warmup_steps', 0)
        critic_warmup_horizon = getattr(self.config, 'critic_warmup_horizon', 20)
        starting_horizon = getattr(self.config, 'starting_horizon', 10)
        horizon_delta = getattr(self.config, 'horizon_delta', 10)
        default_horizon = getattr(self.config, 'default_horizon', 20)

        # Ensure all horizons are at least 1
        critic_warmup_horizon = max(1, int(critic_warmup_horizon))
        starting_horizon = max(1, int(starting_horizon))
        default_horizon = max(1, int(default_horizon))

        # 1. Warmup stage
        if global_steps <= critic_warmup_steps:
            return critic_warmup_horizon

        # Calculate steps after warmup
        steps_after_warmup = global_steps - critic_warmup_steps

        # 2. Curriculum stages (k stages)
        if num_curriculum_stages > 0 and total_actor_training_steps > 0 and steps_after_warmup > 0:
            # Calculate steps per stage after warmup (round up to ensure all steps are covered)
            steps_per_stage = max(1, math.ceil((total_actor_training_steps - critic_warmup_steps) / num_curriculum_stages))
            
            # Determine current stage (0-indexed)
            current_stage = (steps_after_warmup - 1) // steps_per_stage

            if current_stage < num_curriculum_stages:
                # Horizon for the current stage
                effective_horizon = starting_horizon + current_stage * horizon_delta
                return max(1, int(effective_horizon))

        # 3. Post-stages: If global_steps exceeds all curriculum stages
        return default_horizon

    @torch.no_grad()
    def reset(self, env_configs, has_val_first=False, global_steps=0):
        """
        Reset environments based on provided configurations, reusing environments when possible.
        - For env with same config and env_name, reuse the same environment (reset)
        - For env with different config or env_name, close the old environment and create a new one
        - Reset the recorder
        
        Args:
            env_configs: List of environment configurations containing env_name, config, and seed
        
        Returns:
            Initial observations and info from all environments
        """
        # Step 1: Sort environments into buckets by env_name and config
        # Try to reuse environemnts with the same config and env_name
        
        # env_buckets = defaultdict(set)
        
        if self.envs is None:
            self.envs = {} # This is now id:config_instance

        ids2configs_create = {}
        ids2seeds_reset = {}

        if len(self.envs) == 0 and (not has_val_first):
            for i, cfg in enumerate(env_configs):
                env_id = self.split + str(i)
                self.envs[env_id] = cfg["seed"]
                ids2configs_create[env_id] = cfg
                ids2seeds_reset[env_id] = cfg["seed"]
            self.env_client.create_environments_batch(ids2configs_create)
            reset_results=self.env_client.reset_batch(ids2seeds_reset, global_steps=global_steps)
        else:
            for i, cfg in enumerate(env_configs):
                env_id = self.split + str(i)
                self.envs[env_id] = cfg["seed"]
                ids2seeds_reset[env_id] = cfg["seed"]
            reset_results=self.env_client.reset_batch(ids2seeds_reset, global_steps=global_steps)
            
        # for env_id, env_config_instance in self.envs.items():
        #     env_config_id = env_config_instance.config_id()
        #     bucket_key = env_config_id
        #     env_buckets[bucket_key].add(env_id)
        
        # # Step1. collect envs which need to be reset and new env configs
        # ids2seeds_reset = {}
        # configs_to_create=[]
        # for i, cfg in enumerate(env_configs):
        #     # Create bucket key
        #     config_instance= REGISTERED_ENV[cfg["env_name"]]["config_cls"](**cfg["env_config"])
        #     env_config_id = config_instance.config_id()
        #     bucket_key = env_config_id
            
        #     # Check if we have an available environment with the same config
        #     if bucket_key in env_buckets and env_buckets[bucket_key]:
        #         old_env_id = env_buckets[bucket_key].pop()
        #         ids2seeds_reset[old_env_id] = cfg["seed"]
        #     else:
        #         # don't initialize the environment here, close unused environments first
        #         configs_to_create.append(cfg)
        
        # # Step 2: Collect ids which need to be closed
        # ids_to_close=[]
        # # Close unused environments
        # for bucket_key, env_ids in env_buckets.items():
        #     for env_id in env_ids:
        #         ids_to_close.append(env_id)
        #         self.envs.pop(env_id)

        # # Step 3: Close unused environments
        # #print(f"[DEBUG] ids_to_close: {ids_to_close}")
        # self.env_client.close_batch(ids_to_close)
        # # Step 4: Create new environments
        # ids2configs_create = {}
        # id=0
        # for cfg in configs_to_create:
        #     id+=1
        #     while self.split+str(id) in self.envs:
        #         id+=1
        #     id_str = self.split+str(id)
        #     ids2configs_create[id_str] = cfg
        #     ids2seeds_reset[id_str] = cfg["seed"]
        #     self.envs[id_str] = REGISTERED_ENV[cfg["env_name"]]["config_cls"](**cfg["env_config"])
        # #print(f"[DEBUG] ids2configs_create: {ids2configs_create}")
        # self.env_client.create_environments_batch(ids2configs_create)
        # # Step 5: Reset environments
        # #print(f"[DEBUG] ids2seeds_reset: {ids2seeds_reset}")
        # reset_results=self.env_client.reset_batch(ids2seeds_reset)
        
        
        if self.recorder is not None:
            del self.recorder
        self.recorder = defaultdict(list)
        initial_obs = {}
        initial_info = {}
        
        
        for env_id, rst in reset_results.items():
            obs, info = rst
            initial_obs[env_id] = obs
            initial_info[env_id] = info
            self.record(
                env_id, 
                obs=obs, 
                reward=0, 
                done=False, 
                info=info
            )
        
        self.env_states = {env_id: {'seed': self.envs[env_id], 'step': 0, 'done': False,'metrics':{"turn_metrics":defaultdict(list),"traj_metrics":{}}} for env_id in self.envs}

        self.system_prompts=self.env_client.get_system_prompts_batch(list(self.envs.keys()))

        # print("--------------------------------")
        # print(self.system_prompts)
        # print("--------------------------------")

        return initial_obs, initial_info
    
    @torch.no_grad()
    def record(self, env_id, obs, reward, done, info):
        """
        Record each step's obs, info, done, reward,
        Please include "llm_raw_response" in info # it will be decoded by rollout manager and pass to env, then should pass back
        """
        # Create a record entry for this step
        assert obs is not None, "obs cannot be None"
        assert info is not None, "info cannot be None"
        assert isinstance(reward, (int, float)), "reward must be a number"
        assert isinstance(done, bool), "done must be a boolean"
        record_entry = {
            'env_id': env_id,
            'done': done,
            'reward': reward,
            'info': info,
            'obs_str': obs['obs_str'],
        }
        #image_placeholder = self.envs[env_id].get('image_placeholder', "<image>")
        image_placeholder = "<image>"
        if 'multi_modal_data' in obs:
            if image_placeholder in obs['multi_modal_data']:
                record_entry['image_data'] = [process_image(image) for image in obs['multi_modal_data'][image_placeholder]]
        self.recorder[env_id].append(record_entry)

    @torch.no_grad()
    def _single_recording_to_prompt(self,
                            recording: List[Dict], 
                            step: int, 
                            window_size: int = None,
                            is_final: bool = False,
                            prep_for_loss_mask: bool = False,
        ):
        """
        Given a recording, generate the prompt for MLLM
        Chat: Sys -> |InitUser| -> |Assistant, User| -> |Assistant, User| -> ... -> |Assistant, User Final|

        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate prompt for
            window_size: Number of past steps to include in the context
            is_final: Whether the prompt is for the final step 
                - if True, the end of the chat is from the last assistant's response
            prep_for_loss_mask: whether to use special token to wrap llm response
            
        Returns:
            dict: prompt_with_chat_template : str, image_data: list of images, reward: list of reward
        """
        
        assert step >= 0
        start_step = max(0, step - window_size) if window_size is not None else 0
        end_step = step
        assert len(recording) >= end_step + 1, 'History length is not enough'
        history = recording[start_step: end_step + 1]
        rewards=[]
        chat = []
        
        env_id = history[0]['env_id']
        chat.append({"role": "system", "content": self.system_prompts[env_id]})
        # print("--------------------------------")
        # print(chat)
        # print("--------------------------------")

        image_data=[]

        for i, record in enumerate(history):
            if i>0:
                llm_raw_response = record['info']['llm_raw_response']
                filtered_llm_raw_response = self._handle_special_tokens(llm_raw_response, prep_for_loss_mask=prep_for_loss_mask)
                chat.append({"role": "assistant", "content": filtered_llm_raw_response})
                rewards.append(record['reward'])
            if i<len(history)-1 or not is_final:
                # print("--------------------------------")
                # print(record['obs_str'])
                # print("--------------------------------")
                chat.append({"role": "user", "content": record['obs_str']})
                if 'image_data' in record:
                    for img in record['image_data']:
                        image_data.append(img)
            
        prompt_with_chat_template = self.tokenizer.apply_chat_template(conversation=chat, chat_template=chat_template, add_generation_prompt=(not is_final), tokenize=False)
        # print("--------------------------------")
        # print(prompt_with_chat_template)
        # print("--------------------------------")
        if is_final:
            assert prompt_with_chat_template[-1] == '\n', f"The last token should be new line token, got {prompt_with_chat_template[-1]}"
            prompt_with_chat_template = prompt_with_chat_template[:-1] # remove the last in token
        # switch box_end and im_end so that the model can learn to generate <|im_end|>
        #### adpatation to "not skip special tokens during generation"
        # prompt_with_chat_template = prompt_with_chat_template.replace(
        #     f'{self.config.special_token_for_loss_mask[1]}{self.tokenizer.eos_token}',
        #     f'{self.tokenizer.eos_token}{self.config.special_token_for_loss_mask[1]}')
        return {
            "prompt": prompt_with_chat_template,
            "image_data": image_data,
            "rewards": rewards,
        }
    
    @torch.no_grad()
    def _generate_input_for_rollout(
            self, 
            recording: List[Dict], 
            step: int, 
            window_size: int = None,
        ):
        """
        Given a recording, generate the input for MLLM
        
        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - prompts: task instruction
            - responses: responses generated from prompts
            - input_ids, attention_mask, position_ids: prompts and responses generated from prompts
            - position_ids: 
                - position_ids for prompts: rope
                - rest postion_ids: refer to vllm_rollout_spmd.py to check how to compute
        """
        rst=self._single_recording_to_prompt(recording, step, window_size, is_final=False, prep_for_loss_mask=False)
        prompt_with_chat_template=rst['prompt']
        # print("--------------------------------")
        # print(prompt_with_chat_template)
        # print("--------------------------------")
        image_data=rst['image_data']        
        has_images = len(image_data) > 0    
        row_dict = {}
        if has_images:  # expand image token
            prompt_with_chat_template, row_dict, _, raw_prompt = self._handle_multi_modal_data(
                prompt_with_chat_template, row_dict, image_data, do_embedding=False)
        else:
            raw_prompt = prompt_with_chat_template

        # use random input_ids and attention_mask for vllm only takes raw_prompt_ids as input when generating sequences
        # TODO check if this is correct
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict['input_ids'] = torch.tensor([0], dtype=torch.long)
        row_dict['attention_mask'] = torch.tensor([0], dtype=torch.long)
        row_dict['position_ids'] = torch.tensor([0], dtype=torch.long)

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict


    @torch.no_grad()
    def _generate_input_for_uptate(
            self, 
            recording: List[Dict], 
            step: int, 
            window_size: int = None,
        ):
        """
        Given a recording, generate the final input for MLLM
        
        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - prompts: task instruction
            - responses: responses generated from prompts
            - input_ids, attention_mask, position_ids: prompts and responses generated from prompts
            - position_ids: 
                - position_ids for prompts: rope
                - rest postion_ids: refer to vllm_rollout_spmd.py to check how to compute

        """



        # handle prompt, prompt=pad_token since we now have everything in response and compute a loss mask for them
        prompt_with_chat_template=self.tokenizer.pad_token 
        
        # handle response
        response_rst=self._single_recording_to_prompt(recording, step, window_size, is_final=True, prep_for_loss_mask=True)
        response_with_chat_template=response_rst['prompt']
        image_data=response_rst['image_data']
        rewards=response_rst['rewards']
       
        has_images = len(image_data) > 0
        row_dict = {}
        if has_images:  # expand image token
            response_with_chat_template, row_dict, image_grid_thw, _ = self._handle_multi_modal_data(
                response_with_chat_template, row_dict, image_data, do_embedding=True)

        
        input_ids_response, attention_mask_response = verl_F.tokenize_and_postprocess_data(prompt=response_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.config.max_trajectory_length-1, # -1 for the prompt padding token
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=False,
                                                                         truncation=self.config.truncation)
        
        input_ids_prompt, attention_mask_prompt = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=1,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.config.truncation)
        attention_mask_prompt=torch.zeros_like(input_ids_prompt) # All prompt will be masked
        
        
        input_ids_response, attention_mask_response, loss_mask_response,end_of_response_position_mask_response = self._compute_loss_mask(input_ids_response, attention_mask_response)
        
        input_ids_prompt=input_ids_prompt[0]
        attention_mask_prompt=attention_mask_prompt[0]
        loss_mask_prompt = torch.zeros_like(attention_mask_prompt)
        end_of_response_position_mask_prompt = torch.zeros_like(attention_mask_prompt)
        
        input_ids_response=input_ids_response[0]
        attention_mask_response=attention_mask_response[0]
        loss_mask_response=loss_mask_response[0]
        end_of_response_position_mask_response=end_of_response_position_mask_response[0]
        
    
        
        loss_mask = torch.cat([loss_mask_prompt, loss_mask_response], dim=-1)
        end_of_response_position_mask = torch.cat([end_of_response_position_mask_prompt, end_of_response_position_mask_response], dim=-1)
        input_ids = torch.cat([input_ids_prompt, input_ids_response], dim=-1)
        attention_mask = torch.cat([attention_mask_prompt, attention_mask_response], dim=-1)

        
        
        position_ids_prompt = compute_position_id_with_mask(attention_mask_prompt)
        # if self.image_key in row_dict:
        if has_images:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids_response = get_rope_index(
                self.processor,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids_response,
                attention_mask=attention_mask_response,
            )  # (3, seq_len)
            position_ids_prompt=position_ids_prompt.view(1, -1).expand(3, -1)
        else:
            response_length = input_ids_response.shape[0]
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids_prompt.device)
            position_ids_response = position_ids_prompt[-1:] + delta_position_id
        
        if self.config.use_multi_turn_reward:
            reward_positions = torch.nonzero(end_of_response_position_mask).squeeze(-1)
            multi_turn_token_level_rewards = torch.zeros_like(end_of_response_position_mask, dtype=torch.float)
            assert len(reward_positions) == len(rewards), "Number of rewards does not match number of reward positions"
            for idx,reward in enumerate(rewards):
                multi_turn_token_level_rewards[reward_positions[idx]] = reward
            row_dict["multi_turn_token_level_rewards"] = multi_turn_token_level_rewards # (seq_len,) 
        if self.config.use_loss_mask:
            row_dict['loss_mask'] = loss_mask
        if self.config.use_gae_mask:
            row_dict['gae_mask'] = loss_mask
        
        #### modification for era
        # Find the first position of 1, set others to 0
        first_one_pos = (loss_mask == 1).nonzero()[0]
        loss_mask_for_critic = torch.zeros_like(loss_mask)
        loss_mask_for_critic[first_one_pos] = 1
        row_dict["loss_mask_for_critic"] = loss_mask_for_critic
        
        row_dict["end_of_response_position_mask"] = end_of_response_position_mask # 
        position_ids = torch.cat([position_ids_prompt, position_ids_response], dim=-1)
        row_dict['prompts'] = input_ids_prompt
        row_dict['responses'] = input_ids_response
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        #### modified
        row_dict['env_id'] = recording[0]['env_id']
        row_dict['step_id'] = step
        row_dict['single_end_of_response_position'] = reward_positions[-1]-1 # -1 because we have a prompt padding token
        row_dict['single_value_position'] = first_one_pos - 1
        row_dict['single_reward'] = rewards[0]
        return row_dict

    @torch.no_grad()
    def generate_batch_for_rollout(self, step, window_size):
        """
        Generate a batch of data for the current step
        
        Args:
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - None if no data is available (all environments are done)
        """
        batch = []
        self.batch_idx_to_env_id = {}
        batch_idx = 0
        for env_id in self.envs.keys():
            if self.env_states[env_id]['done']:
                continue

            batch.append(self._generate_input_for_rollout(self.recorder[env_id], step, window_size))
            self.batch_idx_to_env_id[batch_idx] = env_id
            batch_idx += 1
        if not batch:
            return None
        if len(batch) % self.config.n_gpus_per_node != 0:
            # Pad the batch to make it divisible by n_gpus_per_node
            while len(batch) % self.config.n_gpus_per_node != 0:
                # do we need to use copy or not here?
                batch.append(batch[-1].copy())
        return collate_fn(batch)
    
    # @torch.no_grad()
    # def rollout_loop(self, global_steps=0):
    #     """
    #     Step the environment and record the results
        
    #     Returns:
    #         Dictionary containing the results of the step
    #     """
    #     if self.config.adaptive_max_turns:
    #         effective_max_turns = max(
    #             1,  # 确保不小于1
    #             self.config.max_turns + 5 * ((global_steps - 1) // 3)
    #         )
    #     else:
    #         effective_max_turns = self.config.max_turns
            
    #     for step in range(self.config.max_turns):
    #         input_batch_dict = self.generate_batch_for_rollout(step, self.config.window_size-1)
    #         if input_batch_dict is None:
    #             break
    #         input_batch = DataProto.from_single_dict(input_batch_dict)
    #         if 'multi_modal_data' in input_batch.non_tensor_batch.keys():
    #             gen_batch = input_batch.pop(
    #                 batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                 non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data'],
    #             )
    #         else:
    #             gen_batch = input_batch.pop(
    #                 batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                 non_tensor_batch_keys=['raw_prompt_ids'],
    #             )

    #         # transform raw_prompt_ids to list instead of numpy array
    #         # The reason is that when constructing raw_prompt_ids, if the all the list share the same length
    #         # Numpy array will automatically transfer list to numpy array.
    #         raw_prompt_ids = gen_batch.non_tensor_batch['raw_prompt_ids']
    #         raw_prompt_ids_array = np.ndarray(shape=(len(raw_prompt_ids),), dtype=object)
    #         for i in range(len(raw_prompt_ids)):
    #             if isinstance(raw_prompt_ids[i],list):
    #                 raw_prompt_ids_array[i] = raw_prompt_ids[i]
    #             else:
    #                 raw_prompt_ids_array[i] = raw_prompt_ids[i].tolist()
    #         gen_batch.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids_array

    #         gen_batch.meta_info['do_sample'] = True
            
    #         output_batch = self.actor_rollout_wg.generate_sequences(gen_batch)
            
    #         responses_str = self.tokenizer.batch_decode(
    #             output_batch.batch['responses'], 
    #             # modification to not skip special tokens during generation
    #             skip_special_tokens=False
    #         ) # seems here will remove special token like "<|im_end|>"

    #         # 移除所有response末尾的padding token
    #         new_responses_str = []
    #         pad_token_str = self.tokenizer.pad_token
    #         if pad_token_str: # Ensure pad_token_str is not None or empty
    #             for response in responses_str:
    #                 temp_response = response
    #                 while temp_response.endswith(pad_token_str):
    #                     temp_response = temp_response[:-len(pad_token_str)]
    #                 new_responses_str.append(temp_response)
    #             responses_str = new_responses_str

            
    #         ids2actions = {}
    #         for batch_idx, env_id in self.batch_idx_to_env_id.items(): 
    #             ids2actions[env_id] = responses_str[batch_idx]
            
    #         step_results = self.env_client.step_batch(ids2actions)
    #         for env_id, rst in step_results.items():
    #             obs, reward, done, info = rst
    #             self.env_states[env_id]['step'] += 1
    #             self.env_states[env_id]['done'] = done
    #             self.env_states[env_id]['metrics']['traj_metrics'] = info['metrics'].get('traj_metrics', {})
    #             for k,v in info['metrics']['turn_metrics'].items():
    #                 self.env_states[env_id]['metrics']['turn_metrics'][k].append(v)
                
    #             self.record(env_id, obs, reward, done, info)

    @torch.no_grad()
    def rollout_loop(self, global_steps=0):
        """
        Step the environment for up to the current curriculum horizon and record the results.

        Returns:
            List[str]: a list of env_ids that *failed* this rollout (i.e., did not reach a terminal/success state)
                    according to the environment's `done` flag. In the 4th curriculum segment this list is empty.
        """
        # ---- Curriculum horizon scheduling (4 segments) ----
        # We support optional config fields:
        # - total_training_steps (int): total number of trainer steps (n)
        # - start_horizon / horizon_k (int): starting max turns (k)
        # - horizon_delta / horizon_j (int): increment per segment (j)
        # If not provided, we fallback to existing max_turns logic.
        # total_steps = getattr(self.config, 'total_training_steps', None)
        # if total_steps is None:
        #     total_steps = getattr(self.config, 'curriculum_total_steps', None)

        # start_horizon = getattr(self.config, 'start_horizon', None)
        # if start_horizon is None:
        #     start_horizon = getattr(self.config, 'horizon_k', None)
        # if start_horizon is None:
        #     start_horizon = getattr(self.config, 'max_turns', 1)

        # horizon_delta = getattr(self.config, 'horizon_delta', None)
        # if horizon_delta is None:
        #     horizon_delta = getattr(self.config, 'horizon_j', 0)

        # # Determine current segment (1..4)
        # segment = None
        # if isinstance(total_steps, int) and total_steps > 0 and isinstance(global_steps, int) and global_steps > 0:
        #     seg_len = math.ceil(total_steps / 4)
        #     if global_steps <= seg_len:
        #         segment = 1
        #     elif global_steps <= 2 * seg_len:
        #         segment = 2
        #     elif global_steps <= 3 * seg_len:
        #         segment = 3
        #     else:
        #         segment = 4

        # # Compute the effective horizon
        # if segment is None:
        #     # No curriculum information available: preserve prior behavior
        #     if getattr(self.config, 'adaptive_max_turns', False):
        #         effective_max_turns = max(1, int(getattr(self.config, 'max_turns', 1)) + 5 * ((max(global_steps,1) - 1) // 3))
        #     else:
        #         effective_max_turns = int(getattr(self.config, 'max_turns', 1))
        # else:
        #     # Four-segment schedule: k, k+j, k+2j, k+2j
        #     if segment == 1:
        #         effective_max_turns = int(start_horizon)
        #     elif segment == 2:
        #         effective_max_turns = int(start_horizon + horizon_delta)
        #     else:  # segment 3 or 4
        #         effective_max_turns = int(start_horizon + 2 * horizon_delta)
        # effective_max_turns = max(1, effective_max_turns)

        if self.config.adaptive_max_turns:
            effective_max_turns = self._get_rollout_horizon(global_steps)
        else:
            effective_max_turns = self.config.max_turns

        # ---- Rollout for the computed horizon ----
        for step in range(effective_max_turns):
            input_batch_dict = self.generate_batch_for_rollout(step, self.config.window_size-1)
            if input_batch_dict is None:
                break
            input_batch = DataProto.from_single_dict(input_batch_dict)
            if 'multi_modal_data' in input_batch.non_tensor_batch.keys():
                gen_batch = input_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data'],
                )
            else:
                gen_batch = input_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            # transform raw_prompt_ids to list instead of numpy array
            raw_prompt_ids = gen_batch.non_tensor_batch['raw_prompt_ids']
            raw_prompt_ids_array = np.ndarray(shape=(len(raw_prompt_ids),), dtype=object)
            for i in range(len(raw_prompt_ids)):
                if isinstance(raw_prompt_ids[i],list):
                    raw_prompt_ids_array[i] = raw_prompt_ids[i]
                else:
                    raw_prompt_ids_array[i] = raw_prompt_ids[i].tolist()
            gen_batch.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids_array

            gen_batch.meta_info['do_sample'] = True
            
            output_batch = self.actor_rollout_wg.generate_sequences(gen_batch)
            
            responses_str = self.tokenizer.batch_decode(
                output_batch.batch['responses'], 
                skip_special_tokens=False
            )

            # 移除所有response末尾的padding token
            new_responses_str = []
            pad_token_str = self.tokenizer.pad_token
            if pad_token_str:
                for temp_response in responses_str:
                    while temp_response.endswith(pad_token_str):
                        temp_response = temp_response[:-len(pad_token_str)]
                    new_responses_str.append(temp_response)
                responses_str = new_responses_str

            ids2actions = {}
            for batch_idx, env_id in self.batch_idx_to_env_id.items(): 
                ids2actions[env_id] = responses_str[batch_idx]
            
            step_results = self.env_client.step_batch(ids2actions)
            for env_id, rst in step_results.items():
                obs, reward, done, info = rst
                self.env_states[env_id]['step'] += 1
                self.env_states[env_id]['done'] = done
                self.env_states[env_id]['metrics']['traj_metrics'] = info['metrics'].get('traj_metrics', {})
                for k,v in info['metrics']['turn_metrics'].items():
                    self.env_states[env_id]['metrics']['turn_metrics'][k].append(v)
                
                self.record(env_id, obs, reward, done, info)

        # # ---- Build failure env_id list for filtering (empty in segment 4) ----
        # failure_env_ids = []
        # # Only compute filtering when curriculum segmentation is active.
        # # If we know the segment and it's the 4th, we return an empty list (no filtering).
        # if segment is not None and segment != 4:
        #     for env_id, state in self.env_states.items():
        #         # Treat unfinished trajectories as failures at the end of the current horizon
        #         if not state.get('done', False):
        #             failure_env_ids.append(env_id)

        # return failure_env_ids

        
    @torch.no_grad()
    def generate_batch_for_update(self, filter_failure_count: int = 0) -> DataProto:
        """
        Get the final trajectory of all environments

        Args:
            filter_failure_count: Number of failure trajectories to filter out for curriculum learning

        Returns:
            batch (DataProto): batch of final trajectory of selected environments
        """
        # Step 1: Determine success/failure status for each environment
        env_success_status = {}
        # reward_rst = self.env_client.compute_reward_batch(list(self.envs.keys()))
        
        for env_id in self.envs.keys():
            
            env_state = self.env_states[env_id]
            is_done = env_state['done']
            env_success_status[env_id] = {
                'is_success': is_done
            }
        
        # Step 2: Sort environments by success/failure
        success_envs = [env_id for env_id, status in env_success_status.items() if status['is_success']]
        failure_envs = [env_id for env_id, status in env_success_status.items() if not status['is_success']]
        
        # Step 3: Apply curriculum learning filtering
        selected_envs = success_envs.copy()  # Always include all successful trajectories
        
        # Filter failure trajectories based on curriculum schedule
        if filter_failure_count > 0 and len(failure_envs) > filter_failure_count:
            # randomly sample from failure envs
            import random
            failures_to_keep = random.sample(failure_envs, len(failure_envs) - filter_failure_count)
            selected_envs.extend(failures_to_keep)
            
            print(f"[Curriculum Learning] Total envs: {len(self.envs)}, "
                  f"Success: {len(success_envs)}, "
                  f"Failure: {len(failure_envs)}, "
                  f"Filtered out: {filter_failure_count}, "
                  f"Selected: {len(selected_envs)}")
        else:
            # Include all failure trajectories (no filtering)
            selected_envs.extend(failure_envs)
            print(f"[Curriculum Learning] No filtering applied. "
                  f"Total envs: {len(self.envs)}, "
                  f"Success: {len(success_envs)}, "
                  f"Failure: {len(failure_envs)}, "
                  f"Selected: {len(selected_envs)}")
        
        # Step 4: Generate batch from selected environments
        batch_list = []
        for env_id in selected_envs:
            recording = self.recorder[env_id]
            #### !!!! Necessary update for era: We need to reversely loop the recording
            for step in range(len(recording)-1, 0, -1):
                row_dict = self._generate_input_for_uptate(
                    recording=recording,
                    step=step,
                    window_size=self.config.window_size,
                )
                #### row_dict['reward_model'] = {"style": "given", "ground_truth": {"reward": reward_rst[env_id]}}
                batch_list.append(row_dict)

        if len(batch_list) % self.config.ppo_mini_batch_size != 0:
            # Pad the batch to make it divisible by ppo_mini_batch_size
            while len(batch_list) % self.config.ppo_mini_batch_size != 0:
                # randomly select a sample to pad
                random_idx = np.random.randint(0, len(batch_list))
                batch_list.append(batch_list[random_idx].copy())
        batch_dict = collate_fn(batch_list)
        batch = DataProto.from_single_dict(batch_dict)
        return batch
    
    @torch.no_grad()
    def recording_to_log(self):
        """
        Get the recording of all environments
        
        Returns:
            Dictionary containing the recording of all environments
        """
        env_info = []
        reward_rst=self.env_client.compute_reward_batch(list(self.envs.keys()))
        for env_id, record in self.recorder.items():
            step= self.env_states[env_id]['step']
            output_rst = self._single_recording_to_prompt(record, self.env_states[env_id]['step'], window_size=None, is_final=False)
            image= output_rst['image_data']
            done = self.env_states[env_id]['done']
            score = reward_rst[env_id]
            
            
            metrics={
                "score": score,
                "done": done,
                "step": step,
            }
            
            turn_metrics={
                k: sum(v)/step if step != 0 else 0 for k, v in self.env_states[env_id]['metrics']['turn_metrics'].items()
            }
            traj_metrics=self.env_states[env_id]['metrics']['traj_metrics']
            metrics.update(turn_metrics)
            metrics.update(traj_metrics)
            env_info.append({
                "env_id": env_id,
                "config_id": 'ebman',
                "output_str": output_rst['prompt'],
                "image_data": image,
                "metrics": metrics,
            })
        return env_info

    @torch.no_grad()
    def save_conversation_history(self, global_step: int, log_file_path: str = None):
        """
        将每个environment的对话交互记录保存到txt文件中
        
        Args:
            global_step: 当前的global step
            log_file_path: 日志文件路径，如果为None则使用默认路径
        """
        if log_file_path is None:
            log_file_path = f"conversation_history_step_{global_step}.txt"
        
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else '.', exist_ok=True)
        
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Global Step: {global_step} ===\n")
            f.write(f"=== Split: {self.split} ===\n\n")
            
            # 遍历每个environment的recording
            for env_id, recording in self.recorder.items():
                if not recording:  # 如果recording为空，跳过
                    continue
                    
                f.write(f"Environment ID: {env_id}\n")
                f.write(f"Steps: {self.env_states[env_id]['step']}\n")
                f.write(f"Done: {self.env_states[env_id]['done']}\n")
                f.write(f"Seed: {self.env_states[env_id]['seed']}\n")
                f.write("-" * 80 + "\n")
                
                # 构建对话历史
                # 添加system prompt
                system_prompt = self.system_prompts[env_id]
                f.write(f"SYSTEM: {system_prompt}\n\n")
                
                # 遍历recording构建对话
                for i, record in enumerate(recording):
                    # 如果不是第一条记录，先添加assistant的回复
                    if i > 0:
                        llm_raw_response = record['info']['llm_raw_response']
                        # 处理特殊token
                        # filtered_response = self._handle_special_tokens(llm_raw_response, prep_for_loss_mask=False)
                        f.write(f"ASSISTANT: {llm_raw_response}\n\n")
                    
                    # 如果不是最后一条记录，添加user的observation
                    if i < len(recording) - 1:
                        obs_str = record['obs_str']
                        f.write(f"USER: {obs_str}\n\n")
                    
                    # 添加reward和done信息作为注释
                    f.write(f"# Step {i}: Reward={record['reward']}, Done={record['done']}\n")
                    if 'image_data' in record:
                        f.write(f"# Images: {len(record['image_data'])} images present\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n\n")
            
            f.write(f"=== End of Global Step {global_step} ===\n\n")
        
        print(f"对话历史已保存到: {log_file_path}")
        
    @torch.no_grad()
    def append_conversation_history(self, global_step: int, log_file_path: str = "conversation_history_all_steps.txt"):
        """
        将每个environment的对话交互记录追加到一个总的txt文件中
        
        Args:
            global_step: 当前的global step
            log_file_path: 日志文件路径
        """
        import os
        
        # 如果文件不存在，创建目录
        os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else '.', exist_ok=True)
        
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n=== Global Step: {global_step} ===\n")
            f.write(f"=== Split: {self.split} ===\n\n")
            
            # 遍历每个environment的recording
            for env_id, recording in self.recorder.items():
                if not recording:  # 如果recording为空，跳过
                    continue
                    
                f.write(f"Environment ID: {env_id}\n")
                f.write(f"Steps: {self.env_states[env_id]['step']}\n")
                f.write(f"Done: {self.env_states[env_id]['done']}\n")
                f.write("-" * 80 + "\n")
                
                # 构建对话历史
                # 添加system prompt
                system_prompt = self.system_prompts[env_id]
                f.write(f"SYSTEM: {system_prompt}\n\n")
                
                # 遍历recording构建对话
                for i, record in enumerate(recording):
                    # 如果不是第一条记录，先添加assistant的回复
                    if i > 0:
                        llm_raw_response = record['info']['llm_raw_response']
                        # 处理特殊token
                        # filtered_response = self._handle_special_tokens(llm_raw_response, prep_for_loss_mask=False)
                        f.write(f"ASSISTANT: {llm_raw_response}\n\n")
                    
                    # 如果不是最后一条记录，添加user的observation
                    if i < len(recording) - 1:
                        obs_str = record['obs_str']
                        f.write(f"USER: {obs_str}\n\n")
                    
                    # 添加reward和done信息作为注释
                    f.write(f"# Step {i}: Reward={record['reward']}, Done={record['done']}\n")
                    if 'image_data' in record:
                        f.write(f"# Images: {len(record['image_data'])} images present\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n\n")
            
            f.write(f"=== End of Global Step {global_step} ===\n\n")
        
        print(f"对话历史已追加到: {log_file_path}")
            
            