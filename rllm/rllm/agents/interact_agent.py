import os
import re
import sys
import copy
import logging

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


class InteractAgent(BaseAgent):
    """
    An interactive agent that interacts with the multi-step environment
    without using tools, refactored to follow the BaseAgent abstraction.
    """

    def __init__(
        self,
        base_prompt_path: str
    ):
        """
        Initialize the InteractAgent.
        
        Args:
            base_prompt_path: Path to the base prompt instruction file.
        """
        if not os.path.exists(base_prompt_path):
            logger.error(f"Base prompt path not found: {base_prompt_path}")
            raise FileNotFoundError(f"Base prompt path not found: {base_prompt_path}")
        else:
            with open(base_prompt_path, "r") as f:
                self.base_prompt = f.read()

        # Initialize state according to BaseAgent
        self._trajectory = Trajectory()
        self.messages: list[dict[str, any]] = []
        self.current_observation = None
        self.reset()
    
    def update_from_env(self, observation: str, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's state based on environment feedback.
        Updates the trajectory.
        """
        assert isinstance(observation, str), "Observation to InteractAgent should be a string. Please check rllm.environments.interactive.your_env"
        
        self.messages.append({"role": "user", "content": observation})
        self.current_observation = observation
        
        if self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            self._trajectory.steps[-1].done = done
            self._trajectory.steps[-1].info = info
    
    def _extract_action(self, s: str) -> str:
        """Extract action from model output string"""
        s = s.strip()
        match = re.search(r"Action:\s*(.*)", s)
        if match:
            return match.group(1).strip()
        return ""
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's state based on the model's response.
        Parses the response, updates messages, and the current step in the trajectory.
        """        
        self.messages.append({"role": "assistant", "content": response})
        extracted_action = self._extract_action(response)
        
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            model_response=response,
            observation=self.current_observation,
            action=extracted_action 
        )
        self._trajectory.steps.append(new_step)
        
        return Action(action=extracted_action)
    
    def reset(self):
        """Resets the agent's state for a new episode."""
        self._trajectory = Trajectory()
        self.current_observation = None
        
        self.messages = [
            {"role": "user", "content": self.base_prompt},
            {"role": "assistant", "content": "OK"}
        ]
        
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Returns the current message history for the model."""
        return self.messages
    
    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory recorded so far."""
        return self._trajectory
    