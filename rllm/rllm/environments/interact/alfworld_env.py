import yaml
import threading
import multiprocessing as mp

import envs.alfworld as alfworld

from rllm.environments.base.base_env import BaseEnv


# Process worker function to manage the ALFWorld environment
def _env_worker(conn, config_path, split):
    with open(config_path) as reader:
        config = yaml.safe_load(reader)
        
    # Create the environment factory
    factory = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = None

    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            task = data
            factory.game_files = [task["game_file"]]
            env = factory.init_env(batch_size=1)
            ob, info = env.reset()
            ob = '\n'.join(ob[0].split('\n\n')[1:])
            conn.send((ob, info))
        elif cmd == "step":
            action = data
            observation, reward, done, info = env.step([action])
            conn.send((observation, reward, done, info))
        elif cmd == "close":
            if env:
                env.close()
            conn.close()
            break


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
        
        # Create pipe for communication (multi-thread safe)
        self.parent_conn, self.child_conn = mp.Pipe()
        self.worker_process = mp.Process(
            target=_env_worker, 
            args=(self.child_conn, config_path, split),
            daemon=True
        )
        self.worker_process.start()
        
        self.task = task
        self.max_turns = max_turns
        self.current_turn = 0
        self.done = False
        
        self.max_error_steps = 5
        self.error_steps = 0
        self.max_invalid_steps = 5
        self.invalid_steps = 0
        
        self.lock = threading.Lock()
        
    def reset(self, task=None) -> tuple[str, dict]:
        """
        Reset the ALFWorld environment with a new task.
        task: {"game_flie": str (path to pddl file)}
        """
        with self.lock:
            if task is not None:
                self.task = task
            
            self.parent_conn.send(("reset", self.task))
            ob, info = self.parent_conn.recv()
            
            self.done = False
            self.current_turn = 0
            self.error_steps = 0
            self.invalid_steps = 0
            
            return ob, info
    
    def _process_ob(self, ob):
        """Process observation string"""
        if ob.startswith('You arrive at loc '):
            ob = ob[ob.find('. ')+2:]    
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
            
            # Send step command
            self.parent_conn.send(("step", action))
            observation, reward, done, info = self.parent_conn.recv()
            observation, reward, self.done = self._process_ob(observation[0]), info['won'][0], done[0]
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
            self.worker_process.join(timeout=1)
        except:
            pass
    
    @staticmethod
    def from_dict(env_args: dict) -> "ALFWorldEnv":
        max_turns = env_args.get("max_turns", 40)
        config_path = env_args.get("config_path", "/mnt/home/user28/MPRL/envs/alfworld/base_config.yaml")
        split = env_args.get("split", "train")
        return ALFWorldEnv(task=env_args, max_turns=max_turns, config_path=config_path, split=split)
    
    @staticmethod
    def is_multithread_safe() -> bool:
        return True