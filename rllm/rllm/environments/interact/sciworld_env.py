import multiprocessing as mp
import threading

from scienceworld import ScienceWorldEnv

from rllm.environments.base.base_env import BaseEnv


# Process worker function to manage the SciWorld environment
def _env_worker(conn, server_path, max_turns):
    # Create the environment factory
    env = ScienceWorldEnv("", serverPath=server_path, envStepLimit=max_turns)

    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            task = data
            env.load(task["task_name"], variationIdx=task["variation_idx"], simplificationStr="easy", generateGoldPath=False)
            ob, info = env.reset()
            conn.send((ob, info))
        elif cmd == "step":
            action = data
            observation, reward, done, info = env.step(action)
            conn.send((observation, reward, done, info))
        elif cmd == "close":
            conn.close()
            break


def _sciworld_step_patch():
    r"""Patch ScienceWorldEnv step function"""
    def step(self, inputStr:str):
        observation = self.server.step(inputStr)
        raw_score = self.server.getScore()
        score = int(round(100 * raw_score))        # Convert from 0-1 to 0-100
        isCompleted = self.server.getCompleted()
        numMoves = self.getNumMoves()

        # Calculate reward
        reward = score - self.lastStepScore         # Calculate reward (delta score) for this step
        self.lastStepScore = score                  # Store current score for reward calculation on the next step


        # If the number of moves exceeds the environment step limit, then set isCompleted to be true
        if (numMoves > self.envStepLimit):
            isCompleted = True

        # New: Handle this in the API rather than the agent -- if the score is less than zero, then set the isCompleted flag to true.
        if (score < 0):
            isCompleted = True
        
        taskDesc = self.taskdescription()
        taskDesc = taskDesc.split('Task Description:\n')[1].strip()

        # Mirror of Jericho API
        infos = {
            'moves': numMoves,
            'raw_score': raw_score,
            'score': score,
            'reward': reward,
            'look': self.look(),
            'inv': self.inventory(),
            'taskDesc': taskDesc,
            'valid': self.getValidActionObjectCombinations(),
            'variationIdx': self.variationIdx,
            'taskName': self.taskName,
            'simplificationStr': self.simplificationStr,
        }

        return observation, reward, isCompleted, infos
    
    ScienceWorldEnv.step = step
    print("Patched ScienceWorldEnv.step function.")


class SciWorldEnv(BaseEnv):
    """
    An environment for multi-turn interactions with LLMs using ScienceWorld.
    The environment provides science-based tasks and evaluates responses using reward from the SciWorld environment.
    The interaction terminates after reaching the maximum number of turns.
    """
    
    def __init__(
        self, 
        task: dict | None = None,
        max_turns: int = 60,
        server_path: str = "/mnt/home/user28/MPRL/data/indices/sciworld/scienceworld.jar",
    ) -> None:
        """
        Args:
            task: {"task_name": str, "variation_idx": int}
            max_turns: Maximum number of turns before terminating the interaction.
            server_path: Path to the ScienceWorld server JAR file.
        """
        super().__init__()
        _sciworld_step_patch()
        # Create pipe for communication (multi-thread safe)
        self.parent_conn, self.child_conn = mp.Pipe()
        self.worker_process = mp.Process(
            target=_env_worker, 
            args=(self.child_conn, server_path, max_turns),
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
        Reset the SciWorld environment with a new task.
        task: {"task_name": str, "variation_idx": int, ...}
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
            
            return info['taskDesc'], info
    
    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """
        Take a step in the SciWorld environment based on the action.
        
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
            
            self.parent_conn.send(("step", action))
            observation, reward, self.done, info = self.parent_conn.recv()
            # The patched ScienceWorld API returns the score delta. rLLM sums
            # step rewards into the trajectory reward, so returning raw_score
            # here would repeatedly count the cumulative score.
            observation = f"Observation: {observation}"
            
            if "No known action matches that input" in observation:
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
        except Exception:
            pass
    
    @staticmethod
    def from_dict(env_args: dict) -> "SciWorldEnv":
        max_turns = env_args.get("max_turns", 60)
        server_path = env_args.get("server_path", "/mnt/home/user28/MPRL/data/indices/sciworld/scienceworld.jar")
        return SciWorldEnv(task=env_args, max_turns=max_turns, server_path=server_path)
    
    @staticmethod
    def is_multithread_safe() -> bool:
        # ScienceWorld depends on a Java server which is not thread-safe
        return True