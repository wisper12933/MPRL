import yaml
import threading
import multiprocessing as mp

from envs.webshop.web_agent_site.envs import WebAgentTextEnv

from rllm.environments.base.base_env import BaseEnv

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Process worker function to manage the WebShop environment
def _env_worker(conn): 
    # Create the environment factory
    env = WebAgentTextEnv(observation_mode="text", human_goals=True)

    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            task = data
            env.reset(task["id"])
            ob, info = env.observation, {"task_desciption": env.observation}
            conn.send((ob, info))
        elif cmd == "step":
            action = data
            try:
                observation, reward, done, info = env.step(action=action)
                observation = f"Observation: {observation}"
            except AssertionError:
                observation, reward, done, info = "Observation: Invalid action!", 0.0, False, {}
            if info is None:
                info = {}
            conn.send((observation, reward, done, info))
        elif cmd == "close":
            conn.close()
            break


class WebShopEnv(BaseEnv):
    """
    An environment for multi-turn interactions with LLMs using WebShop.
    The environment provides house-holding tasks and evaluates responses using reward from the WebShop environment.
    The interaction terminates after reaching the maximum number of turns.
    """
    
    def __init__(
        self, 
        task: dict | None = None,
        max_turns: int = 12,
    ) -> None:
        """
        Args:
            task: {"id": int}
            max_turns: Maximum number of turns before terminating the interaction.
        """
        super().__init__()
        
        # Create pipe for communication (multi-thread safe)
        self.parent_conn, self.child_conn = mp.Pipe()
        self.worker_process = mp.Process(
            target=_env_worker, 
            args=(self.child_conn,),
            daemon=True
        )
        self.worker_process.start()
        
        self.task = task
        self.max_turns = max_turns
        self.current_turn = 0
        self.done = False
        
        self.max_error_steps = 3
        self.error_steps = 0
        self.max_invalid_steps = 3
        self.invalid_steps = 0
        
        self.lock = threading.Lock()
        
    def reset(self, task=None) -> tuple[str, dict]:
        """
        Reset the Webshop environment with a new task.
        task: {"id": int}
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
    
    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """
        Take a step in the WebShop environment based on the action.
        
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
            observation, reward, self.done, info = self.parent_conn.recv()
            
            if "Invalid action!" in observation:
                self.error_steps += 1
                self.reward = 0.0  # panalize invalid action with zero reward
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
    def from_dict(env_args: dict) -> "WebShopEnv":
        max_turns = env_args.get("max_turns", 12)
        return WebShopEnv(task=env_args, max_turns=max_turns)
    
    @staticmethod
    def is_multithread_safe() -> bool:
        return True