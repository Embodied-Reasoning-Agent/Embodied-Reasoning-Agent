# from abc import ABC, abstractmethod
# from typing import List, Dict, Tuple, Optional, Any, Union
# import uuid
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from vagen.env.base.base_service import BaseService
# from vagen.env.Embench_new.mani_env import EBManipulationEnv

# from vagen.server.serial import serialize_observation




# class EBManipulationService():    
#     def __init__(self, serviceconfig: None):
#         self.envs = {}
#         self.max_workers = 32  # Default to 4 workers if not specified
        
#     def create_environment(self, env_id: str, config: Dict[str, Any]) -> None:
#         """
#         Helper function to create a single environment.

#         Args:
#             env_id (str): The environment ID.
#             config (Dict[str, Any]): The configuration for the environment.
#         """
#         # self.envs[env_id] = AlfredEnv(**config["env_config"])
#         self.envs[env_id] = EBManipulationEnv()
#         # try:
#         # self.envs[env_id] = AlfredEnv(**config)  # Create environment from config
#         # except Exception as e:
#         #     print(f"Error creating environment {env_id}: {e}")
#         #     # Handle any error gracefully (e.g., log it, attempt retry, etc.)
    
#     def create_environments_batch(self, ids2configs: Dict[str, Any]) -> None:
#         """
#         Create multiple environments sequentially (not in parallel due to Qt GUI restrictions).

#         Args:
#             ids2configs (Dict[Any, Any]): 
#                 A dictionary where each key is an environment ID and the corresponding
#                 value is the configuration for that environment.

#         Returns:
#             None
#         """
#         # Clear environments that are not in the new config
#         for env_id in list(self.envs.keys()):
#             if env_id not in ids2configs:
#                 self.envs[env_id].close()
#                 del self.envs[env_id]

#         # Create environments sequentially since Qt GUI must be created in main thread
#         for env_id, config in ids2configs.items():
#             self.create_environment(env_id, config)
    
#     def reset_environment(self, env_id: str, seed: Any) -> Tuple[Any, Any]:
#         """
#         Helper function to reset a single environment.

#         Args:
#             env_id (str): The environment ID.
#             seed (Any): The seed for resetting, or None if default behavior is required.

#         Returns:
#             Tuple[Any, Any]: A tuple (observation, info) after the reset.
#         """
#         if seed is None:
#             raise NotImplementedError("None seed is not supported in AlfredEnvService")
        
#         if env_id not in self.envs:
#             raise ValueError(f"Environment {env_id} not found.")
        
#         try:
#             # Reset the environment and return the result
#             observation, info = self.envs[env_id].reset(seed)
#             observation = serialize_observation(observation)
#             return env_id, (observation, info)
#         except Exception as e:
#             print(f"Error resetting environment {env_id}: {e}")
#             return env_id, None  # Handle the error gracefully here, e.g., return None or a default value
    
#     def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
#         """
#         Reset multiple environments in parallel.

#         Args:
#             ids2seeds (Dict[Any, Any]):
#                 A dictionary where each key is an environment ID and the corresponding
#                 value is a seed value (or None for using default seeding behavior).

#         Returns:
#             Dict[Any, Tuple[Any, Any]]:
#                 A dictionary mapping environment IDs to tuples of the form (observation, info),
#                 where 'observation' is the initial state after reset, and 'info' contains additional details.
#         """
#         return_dict = {}
        
#         # Use ThreadPoolExecutor to reset environments concurrently
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = []
#             for env_id, seed in ids2seeds.items():
#                 futures.append(executor.submit(self.reset_environment, env_id, seed))
            
#             for future in as_completed(futures):
#                 try:
#                     env_id, result = future.result()  # Get result, raises exception if occurred
#                     if result is not None:
#                         return_dict[env_id] = result
#                 except Exception as e:
#                     print(f"Error processing future for reset: {e}")
        
#         return return_dict

#     def step_environment(self, env_id: str, action: Any) -> Tuple[Dict, float, bool, Dict]:
#         """
#         Helper function to step through a single environment.

#         Args:
#             env_id (str): The environment ID.
#             action (Any): The action to take in the environment.

#         Returns:
#             Tuple[Dict, float, bool, Dict]: A tuple (observation, reward, done, info) after taking the step.
#         """
#         if env_id not in self.envs:
#             raise ValueError(f"Environment {env_id} not found.")

#         observation, reward, done, info = self.envs[env_id].step(action)
#         observation = serialize_observation(observation)
#         return env_id, (observation, reward, done, info)
#         # try:
#         #     # Step through the environment and return the result
#         #     observation, reward, done, info = self.envs[env_id].step(action)
#         #     observation = serialize_observation(observation)
#         #     return env_id, (observation, reward, done, info)
#         # except Exception as e:
#         #     print(f"Error stepping environment {env_id}: {e}")
#         #     return env_id, None  # Return default values in case of error
    
#     def step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
#         """
#         Step through multiple environments in parallel.

#         Args:
#             ids2actions (Dict[Any, Any]):
#                 A dictionary where each key is an environment ID and the corresponding
#                 value is the action to execute in that environment.

#         Returns:
#             Dict[Any, Tuple[Dict, float, bool, Dict]]:
#                 A dictionary mapping environment IDs to tuples of the form 
#                 (observation, reward, done, info), where:
#                     - 'observation' is the new state of the environment after the action,
#                     - 'reward' is a float representing the reward received,
#                     - 'done' is a boolean indicating whether the environment is finished,
#                     - 'info' contains additional information or context.
#         """
#         return_dict = {}
        
#         # Use ThreadPoolExecutor to step through environments concurrently
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = []
#             for env_id, action in ids2actions.items():
#                 futures.append(executor.submit(self.step_environment, env_id, action))
            
#             # Collect results as they complete
#             for future in as_completed(futures):
#                 try:
#                     env_id, result = future.result()  # Get result, raises exception if occurred
#                     if result[0] is not None:  # Ensure observation is valid
#                         return_dict[env_id] = result
#                 except Exception as e:
#                     print(f"Error processing future for step: {e}")
        
#         return return_dict

#     def compute_reward(self, env_id: str) -> float:
#         """
#         Helper function to compute the reward for a single environment.

#         Args:
#             env_id (str): The environment ID.

#         Returns:
#             float: The computed total reward for the environment.
#         """
#         if env_id not in self.envs:
#             raise ValueError(f"Environment {env_id} not found.")
        
#         try:
#             reward = self.envs[env_id].compute_reward()  # Assuming compute_reward is a method in AlfredEnv
#             return env_id, reward
#         except Exception as e:
#             print(f"Error computing reward for environment {env_id}: {e}")
#             return env_id, None  # Return 0 in case of error
    
#     def compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
#         """
#         Compute the total reward for multiple environments in parallel.

#         Args:
#             env_ids (List[str]): A list of environment IDs.

#         Returns:
#             Dict[Any, float]:
#                 A dictionary mapping each environment ID to its computed total reward.
#         """
#         return_dict = {}
        
#         # Use ThreadPoolExecutor to compute rewards concurrently
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = []
#             for env_id in env_ids:
#                 futures.append(executor.submit(self.compute_reward, env_id))
            
#             for future in as_completed(futures):
#                 try:
#                     env_id, result = future.result()  # Get result, raises exception if occurred
#                     if result is not None:
#                         return_dict[env_id] = result
#                 except Exception as e:
#                     print(f"Error processing future for reward computation: {e}")
        
#         return return_dict

#     def get_system_prompt(self, env_id: str) -> str:
#         """
#         Helper function to retrieve the system prompt for a single environment.

#         Args:
#             env_id (str): The environment ID.

#         Returns:
#             str: The system prompt string for the environment.
#         """
#         if env_id not in self.envs:
#             raise ValueError(f"Environment {env_id} not found.")
        
#         try:
#             system_prompt = self.envs[env_id].get_system_prompt()  # Assuming get_system_prompt is a method in AlfredEnv
#             return env_id, system_prompt
#         except Exception as e:
#             print(f"Error retrieving system prompt for environment {env_id}: {e}")
#             return env_id, ""  # Return empty string in case of error
    
#     def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
#         """
#         Retrieve system prompts for multiple environments in parallel.

#         Args:
#             env_ids (List[str]): A list of environment IDs.

#         Returns:
#             Dict[Any, str]:
#                 A dictionary mapping each environment ID to its corresponding system prompt string.
#         """
#         return_dict = {}
        
#         # Use ThreadPoolExecutor to retrieve system prompts concurrently
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = []
#             for env_id in env_ids:
#                 futures.append(executor.submit(self.get_system_prompt, env_id))
            
#             for future in as_completed(futures):
#                 try:
#                     env_id, system_prompt = future.result()  # Get result, raises exception if occurred
#                     if system_prompt is not None:
#                         return_dict[env_id] = system_prompt
#                 except Exception as e:
#                     print(f"Error processing future for system prompt retrieval: {e}")
        
#         return return_dict

#     def close_environment(self, env_id: str) -> None:
#         """
#         Helper function to close a single environment.

#         Args:
#             env_id (str): The environment ID.
#         """
#         if env_id not in self.envs:
#             print(f"Environment {env_id} not found, skipping close.")
#             return
        
#         try:
#             self.envs[env_id].close()  # Assuming close is a method in AlfredEnv
#         except Exception as e:
#             print(f"Error closing environment {env_id}: {e}")
    
#     def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
#         """
#         Close multiple environments and clean up resources in parallel.

#         Args:
#             env_ids (Optional[List[str]]):
#                 A list of environment IDs to close. If None, all environments should be closed.

#         Returns:
#             None
#         """
#         if env_ids is None:
#             env_ids = list(self.envs.keys())  # Close all environments if no list provided
        
#         # Use ThreadPoolExecutor to close environments concurrently
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = []
#             for env_id in env_ids:
#                 futures.append(executor.submit(self.close_environment, env_id))
            
#             for future in as_completed(futures):
#                 try:
#                     future.result()  # Get result, raises exception if occurred
#                 except Exception as e:
#                     print(f"Error processing future for environment close: {e}")


# multiprocess_service.py
import uuid
from typing import Any, Dict, List, Tuple, Optional

from multiprocessing.connection import Connection, wait
import multiprocessing as mp

from vagen.env.Embench_new.mani_env import EBManipulationEnv
from vagen.server.serial import serialize_observation


def _env_worker(pipe: Connection):

    env = EBManipulationEnv()
    try:
        while True:
            cmd, data, global_step = pipe.recv()
            if cmd == "reset":
                obs, info = env.reset(int(data), int(global_step))
                pipe.send((obs, info))
            elif cmd == "step":
                pipe.send(env.step(data))
            elif cmd == "compute_reward":
                pipe.send(env.compute_reward())
            elif cmd == "get_system_prompt":
                pipe.send(env.get_system_prompt())
            elif cmd == "close":
                env.close()
                break
            else:
                pipe.send(RuntimeError(f"Unknown command {cmd}"))
    except (EOFError, KeyboardInterrupt):
        env.close()



class EBManipulationService:

    def __init__(self, serviceconfig: Optional[dict] = None, *, max_workers: int = 32):
        self._ctx = mp.get_context("spawn")
        self.envs: Dict[str, Tuple[Connection, mp.Process]] = {}
        self.max_workers = max_workers
        print("EBManipulationService initialized", serviceconfig)

    # -- 创建 / 批量创建 -------------------------------------------------- #
    def create_environment(self, env_id: str, config: Dict[str, Any]) -> None:
        if env_id in self.envs:
            self.close_environment(env_id)

        parent_pipe, child_pipe = self._ctx.Pipe()
        proc = self._ctx.Process(target=_env_worker, args=(child_pipe,), daemon=True)
        proc.start()
        self.envs[env_id] = (parent_pipe, proc)

    def create_environments_batch(self, ids2configs: Dict[str, Any]) -> None:
        # 关闭不再需要的
        for eid in list(self.envs.keys()):
            if eid not in ids2configs:
                self.close_environment(eid)
        # 新建或更新
        for eid, cfg in ids2configs.items():
            self.create_environment(eid, cfg)

    # -- reset ----------------------------------------------------------- #
    def reset_environment(self, env_id: str, seed: Any) -> Tuple[str, Tuple[Any, Any]]:
        pipe, _ = self._require_env(env_id)
        pipe.send(("reset", seed))
        obs, info = pipe.recv()
        obs = serialize_observation(obs)
        return env_id, (obs, info)

    def reset_batch(self, ids2seeds: Dict[str, Any], global_step: int) -> Dict[str, Tuple[Any, Any]]:
        return self._broadcast_and_collect(
            ids2payload=ids2seeds,
            cmd="reset",
            global_step=global_step,
            postprocess=lambda obs_info: (serialize_observation(obs_info[0]), obs_info[1]),
        )

    # -- step ------------------------------------------------------------ #
    def step_environment(self, env_id: str, action: Any):
        pipe, _ = self._require_env(env_id)
        pipe.send(("step", action))
        obs, reward, done, info = pipe.recv()
        obs = serialize_observation(obs)
        return env_id, (obs, reward, done, info)

    def step_batch(self, ids2actions: Dict[str, Any]):
        return self._broadcast_and_collect(
            ids2payload=ids2actions,
            cmd="step",
            global_step=0,
            postprocess=lambda tup: (serialize_observation(tup[0]), *tup[1:]),
        )

    # -- compute reward -------------------------------------------------- #
    def compute_reward(self, env_id: str):
        pipe, _ = self._require_env(env_id)
        pipe.send(("compute_reward", None))
        return env_id, pipe.recv()

    def compute_reward_batch(self, env_ids: List[str]):
        dummy = {eid: None for eid in env_ids}
        return self._broadcast_and_collect(
            ids2payload=dummy,
            cmd="compute_reward",
            global_step=0,
        )

    # -- system prompt --------------------------------------------------- #
    def get_system_prompt(self, env_id: str):
        pipe, _ = self._require_env(env_id)
        pipe.send(("get_system_prompt", None))
        return env_id, pipe.recv()

    def get_system_prompts_batch(self, env_ids: List[str]):
        dummy = {eid: None for eid in env_ids}
        return self._broadcast_and_collect(ids2payload=dummy, cmd="get_system_prompt", global_step=0)

    # -- close ----------------------------------------------------------- #
    def close_environment(self, env_id: str):
        if env_id not in self.envs:
            return
        pipe, proc = self.envs.pop(env_id)
        try:
            pipe.send(("close", None))
        except (BrokenPipeError, OSError):
            pass

        proc.join(timeout=1)
        if proc.is_alive():
            proc.terminate()

    def close_batch(self, env_ids: Optional[List[str]] = None):
        env_ids = env_ids or list(self.envs.keys())
        for eid in env_ids:
            self.close_environment(eid)

    # ---------- 内部工具 ---------- #
    def _require_env(self, env_id: str):
        if env_id not in self.envs:
            raise ValueError(f"Environment {env_id} not found.")
        return self.envs[env_id]

    def _broadcast_and_collect(
        self,
        *,
        ids2payload: Dict[str, Any],
        cmd: str,
        global_step: int,
        postprocess=lambda x: x,
    ):
        """向多环境广播指令，然后用 connection.wait 高效收集。"""
        for eid, payload in ids2payload.items():
            pipe, _ = self._require_env(eid)
            pipe.send((cmd, payload, global_step))

        remaining = set(ids2payload.keys())
        result_dict = {}
        conn2id = {self.envs[eid][0]: eid for eid in remaining}

        while remaining:
            ready_conns = wait(list(conn2id.keys()))
            for conn in ready_conns:
                eid = conn2id[conn]
                data = conn.recv()
                result_dict[eid] = postprocess(data)
                remaining.remove(eid)
                del conn2id[conn]

        return result_dict


if __name__ == "__main__":
    service = EBManipulationService()
    configs = {
        '1': {
            "env_name": "ebman",
            "env_config": {}
        },
        '2': {
            "env_name": "ebman",
            "env_config": {}
        },
    }
    service.create_environments_batch(configs)
    print("finished creating environments")

    env_ids = list(configs.keys())
    # print(f"Created {len(env_ids)} environments: {env_ids}")
    
    # # Reset environments
    # print("Resetting environments...")
    ids2seeds = {env_id: i+1 for i, env_id in enumerate(env_ids)}

    # reset
    dict_obs = service.reset_batch(ids2seeds, 0)
    print("finished resetting environments")
    print(dict_obs[env_ids[0]][1])

    ids2actions = {
                env_ids[0]: "<|action_start|>[1,\'idk1\']<|action_end|>",
                env_ids[1]: "<|action_start|>[5,\'idk5\']<|action_end|>"
    }

    # step
    dict_obs = service.step_batch(ids2actions)
    print("finished stepping environments")
    print(dict_obs[env_ids[0]][1])
    # # batch
    # ids2seeds = {f"e{i}": i for i in range(4)}
    # service.create_environments_batch({eid: {} for eid in ids2seeds})
    # batch_obs = service.reset_batch(ids2seeds)

    service.close_batch()

