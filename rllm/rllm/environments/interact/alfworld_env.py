import logging
import multiprocessing as mp
import threading

import envs.alfworld as alfworld
import yaml

from rllm.environments.base.base_env import BaseEnv

logger = logging.getLogger(__name__)


# Process worker function to manage the ALFWorld environment
def _env_worker(conn, config_path, split):
    try:
        with open(config_path) as reader:
            config = yaml.safe_load(reader)
        factory = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    except Exception as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        conn.close()
        return

    env = None
    while True:
        try:
            cmd, data = conn.recv()
        except EOFError:
            break

        if cmd == "close":
            if env:
                env.close()
            conn.close()
            break

        # Any environment failure is reported back over the pipe. If the worker
        # died instead, the parent would block forever on recv().
        try:
            if cmd == "reset":
                factory.game_files = [data["game_file"]]
                env = factory.init_env(batch_size=1)
                ob, info = env.reset()
                ob = "\n".join(ob[0].split("\n\n")[1:])
                conn.send(("ok", (ob, info)))
            elif cmd == "step":
                conn.send(("ok", env.step([data])))
        except Exception as exc:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class ALFWorldEnv(BaseEnv):
    """
    An environment for multi-turn interactions with LLMs using ALFWorld.
    The environment provides house-holding tasks and evaluates responses using reward from the ALFWorld environment.
    The interaction terminates after reaching the maximum number of turns.
    """

    def __init__(
        self,
        task: dict | None = None,
        max_turns: int = 40,
        config_path: str = "/mnt/home/user28/MPRL/envs/alfworld/base_config.yaml",
        split: str = "train",
    ) -> None:
        """
        Args:
            task: {"game_flie": str (path to pddl file)}
            max_turns: Maximum number of turns before terminating the interaction.
            config_path: Path to the ALFWorld configuration file.
            split: Dataset split to use ("train", "eval_out_of_distribution").
        """
        super().__init__()

        self.config_path = config_path
        self.split = split
        self.parent_conn = None
        self.worker_process = None
        self._start_worker()

        self.task = task
        self.max_turns = max_turns
        self.current_turn = 0
        self.done = False

        self.max_error_steps = 5
        self.error_steps = 0
        self.max_invalid_steps = 5
        self.invalid_steps = 0

        self.lock = threading.Lock()

    def _start_worker(self) -> None:
        parent_conn, child_conn = mp.Pipe()
        self.worker_process = mp.Process(
            target=_env_worker,
            args=(child_conn, self.config_path, self.split),
            daemon=True,
        )
        self.worker_process.start()
        # Dropping the parent's copy of the child end makes recv() raise
        # EOFError when the worker exits, instead of blocking forever.
        child_conn.close()
        self.parent_conn = parent_conn

    def _request(self, cmd: str, data) -> tuple[str, object]:
        try:
            self.parent_conn.send((cmd, data))
            return self.parent_conn.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            return "error", f"worker unavailable: {type(exc).__name__}: {exc}"

    def reset(self, task=None) -> tuple[str, dict]:
        """
        Reset the ALFWorld environment with a new task.
        task: {"game_flie": str (path to pddl file)}
        """
        with self.lock:
            if task is not None:
                self.task = task

            if self.worker_process is None or not self.worker_process.is_alive():
                self._start_worker()

            status, payload = self._request("reset", self.task)
            if status == "error":
                raise RuntimeError(f"ALFWorld reset failed for {self.task}: {payload}")
            ob, info = payload

            self.done = False
            self.current_turn = 0
            self.error_steps = 0
            self.invalid_steps = 0

            return ob, info

    def _process_ob(self, ob):
        """Process observation string"""
        if ob.startswith("You arrive at loc "):
            ob = ob[ob.find(". ") + 2 :]
        return ob

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """
        Take a step in the ALFWorld environment based on the action.

        Args:
            action: Response string from the LLM.

        Returns:
            next_observation, reward, done, info
        """
        # Extract action from the model output
        assert isinstance(action, str), "Action.action from InteractAgent (used in AgentExecutionEngine) should be a string."
        with self.lock:
            self.current_turn += 1
            # Step the environment
            if action == "":
                observation = "Observation: Invalid format. The input must contains 'Action: '"
                self.invalid_steps += 1
                if self.invalid_steps >= self.max_invalid_steps:
                    self.done = True
                return observation, 0.0, self.done, {}

            status, payload = self._request("step", action)
            if status == "error":
                logger.warning("ALFWorld step failed for %s: %s", self.task, payload)
                self.done = True
                return "Observation: The environment stopped responding.", 0.0, True, {"env_error": payload}

            observation, _, done, info = payload
            observation, reward, self.done = self._process_ob(observation[0]), info["won"][0], done[0]
            observation = f"Observation: {observation}"

            if "Nothing happens" in observation:
                self.error_steps += 1
                if self.error_steps >= self.max_error_steps:
                    self.done = True
            else:
                self.error_steps = 0

            if self.current_turn >= self.max_turns:
                self.done = True

            return observation, reward, self.done, info

    def close(self):
        try:
            self.parent_conn.send(("close", None))
            self.worker_process.join(timeout=5)
        except Exception:
            pass
        if self.worker_process is not None and self.worker_process.is_alive():
            self.worker_process.terminate()

    @staticmethod
    def from_dict(env_args: dict) -> "ALFWorldEnv":
        max_turns = env_args.get("max_turns", 40)
        config_path = env_args.get("config_path", "/mnt/home/user28/MPRL/envs/alfworld/base_config.yaml")
        split = env_args.get("split", "train")
        return ALFWorldEnv(task=env_args, max_turns=max_turns, config_path=config_path, split=split)

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
