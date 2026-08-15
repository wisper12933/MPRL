from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rllm.agents.interact_agent import InteractAgent


class PlannedInteractAgent(InteractAgent):
    """InteractAgent with one context-only planning call per environment reset."""

    def __init__(
        self,
        base_prompt_path: str,
        metaplan_prompt_path: str,
        planning_max_tokens: int = 1024,
        planning_temperature: float = 0.1,
        planning_top_p: float = 0.9,
    ):
        prompt_path = Path(metaplan_prompt_path)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Meta-plan prompt not found: {prompt_path}")
        self.metaplan_prompt_template = prompt_path.read_text(encoding="utf-8")
        if "{{TASK}}" not in self.metaplan_prompt_template:
            raise ValueError(f"Meta-plan prompt must contain {{{{TASK}}}}: {prompt_path}")
        self.planning_max_tokens = int(planning_max_tokens)
        self.planning_temperature = float(planning_temperature)
        self.planning_top_p = float(planning_top_p)
        self.generated_plan: str | None = None
        self._plan_injected = False
        super().__init__(base_prompt_path=base_prompt_path)

    def reset(self):
        super().reset()
        self.generated_plan = None
        self._plan_injected = False

    def build_planning_messages(self, observation: str, info: dict[str, Any] | None = None) -> list[dict[str, str]]:
        if self._plan_injected:
            raise RuntimeError("A meta-plan has already been generated for this trajectory")
        prompt = self.metaplan_prompt_template.replace("{{TASK}}", observation.strip())
        return [{"role": "user", "content": prompt}]

    @property
    def planning_sampling_params(self) -> dict[str, Any]:
        return {
            "max_tokens": self.planning_max_tokens,
            "temperature": self.planning_temperature,
            "top_p": self.planning_top_p,
        }

    @staticmethod
    def normalize_workflow(response: str) -> str:
        response = (response or "").strip()
        match = re.search(r"<workflow>.*?</workflow>", response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0).strip()
        if not response:
            response = "Step 1: Inspect the task and choose the first valid action."
        return f"<workflow>\n{response}\n</workflow>"

    def inject_plan(self, response: str) -> None:
        if self._plan_injected:
            raise RuntimeError("A meta-plan has already been injected for this trajectory")
        if not self.messages or self.messages[-1].get("role") != "user":
            raise RuntimeError("The initial environment observation must be the latest user message")

        workflow = self.normalize_workflow(response)
        observation = self.messages[-1]["content"]
        self.messages[-1]["content"] = (
            f"{observation}\n\n"
            "A high-level workflow was generated before acting:\n"
            f"{workflow}\n\n"
            "Use this workflow as guidance, but adapt it when new environment observations require a change. "
            "Now continue with the required Thought/Action response format."
        )
        self.generated_plan = workflow
        self._plan_injected = True
        self._trajectory.info["metaplan"] = workflow
