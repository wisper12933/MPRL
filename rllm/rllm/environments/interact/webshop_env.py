import threading
import uuid

from envs.webshop.web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv
from envs.webshop.web_agent_site.utils import DEFAULT_FILE_PATH

from rllm.environments.base.base_env import BaseEnv

_SERVER = None
_SERVER_LOCK = threading.Lock()
_SERVER_REQUEST_LOCK = threading.Lock()


def _get_shared_server():
    """Load the 1.18M-product corpus once per training rank."""
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is None:
            _SERVER = SimServer(
                "http://127.0.0.1:3000",
                DEFAULT_FILE_PATH,
                human_goals=True,
            )
    return _SERVER


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
        
        # Product/search data is immutable and expensive to load. Each
        # WebAgentTextEnv keeps independent browser/session state while sharing
        # the read-mostly simulator within this rank.
        with _SERVER_REQUEST_LOCK:
            self.env = WebAgentTextEnv(
                observation_mode="text",
                human_goals=True,
                server=_get_shared_server(),
                session_prefix=f"{uuid.uuid4().hex}-",
            )
        
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
        with self.lock, _SERVER_REQUEST_LOCK:
            if task is not None:
                self.task = task
            
            ob, _ = self.env.reset(self.task["id"])
            info = {"task_description": ob}
            
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
        with self.lock, _SERVER_REQUEST_LOCK:
            self.current_turn += 1
            # Step the environment
            if action == "":
                observation = "Observation: Invalid format. The input must contains 'Action: '"
                self.invalid_steps += 1
                if self.invalid_steps >= self.max_invalid_steps:
                    self.done = True
                return observation, 0.0, self.done, {}
            
            try:
                observation, reward, self.done, info = self.env.step(action=action)
                observation = f"Observation: {observation}"
            except AssertionError:
                observation, reward, self.done, info = "Observation: Invalid action!", 0.0, False, {}
            if info is None:
                info = {}
            
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
        # The shared server remains alive for reuse by the next rollout batch.
        return None
    
    @staticmethod
    def from_dict(env_args: dict) -> "WebShopEnv":
        max_turns = env_args.get("max_turns", 12)
        return WebShopEnv(task=env_args, max_turns=max_turns)
    
    @staticmethod
    def is_multithread_safe() -> bool:
        return True