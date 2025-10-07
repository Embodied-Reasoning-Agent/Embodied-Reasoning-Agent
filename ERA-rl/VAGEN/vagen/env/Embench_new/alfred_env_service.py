# multiprocess_service.py
import uuid
from typing import Any, Dict, List, Tuple, Optional

from multiprocessing.connection import Connection, wait
import multiprocessing as mp

from vagen.env.Embench_new.alfred_env_for_vagen import AlfredEnv
from vagen.server.serial import serialize_observation


def _env_worker(pipe: Connection):

    env = AlfredEnv()
    try:
        while True:
            cmd, data, global_steps = pipe.recv()
            if cmd == "reset":
                obs, info = env.reset(int(data), int(global_steps))
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



class AlfredService:

    def __init__(self, serviceconfig: Optional[dict] = None, *, max_workers: int = 32):
        self._ctx = mp.get_context("spawn")
        self.envs: Dict[str, Tuple[Connection, mp.Process]] = {}
        self.max_workers = max_workers
        print("AlfredService initialized", serviceconfig)

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