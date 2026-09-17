from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Deliberately rule-only: no embedding model, no corpus statistics, no tokenizer.
# Scoring a whole trajectory costs ~3ms at the 12-plan-step x 60-action worst case.

_STOPWORDS = frozenset(
    """
    a an the this that these those it its
    to of in on at for from into onto with by as
    and or but then after before finally lastly
    is are am be been being will would shall should can could may might must do does did
    i you your my me we us they them he she his her
    if when while where which who whom what how why
    step steps plan workflow first second third fourth fifth
    please make sure need needs needed use using used
    """.split()
)
# "next" / "prev" / "back" stay out of the stopword list: they are real WebShop commands.

_MIN_PREFIX = 4

_STEP_PREFIX = re.compile(r"^\s*(?:step\s*\d+\s*[:.)-]|\d+\s*[:.)-]|[-*+•]\s*)", re.IGNORECASE)
_WORKFLOW_BLOCK = re.compile(r"<workflow>(.*?)</workflow>", re.IGNORECASE | re.DOTALL)
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class PlanRewardConfig:
    """Knobs for the two MPRL reward terms."""

    consistency_enabled: bool = True
    consistency_weight: float = 0.2
    tau: float = 0.45
    lambda_len: float = 0.5
    verb_weight: float = 0.4

    length_penalty_enabled: bool = True
    length_penalty_coeff: float = 0.1
    success_threshold: float = 0.0

    @classmethod
    def from_mapping(cls, mapping: Any) -> PlanRewardConfig:
        if mapping is None:
            return cls()
        if not isinstance(mapping, dict):
            mapping = {key: getattr(mapping, key) for key in cls.__dataclass_fields__ if hasattr(mapping, key)}
        known = {key: mapping[key] for key in cls.__dataclass_fields__ if key in mapping and mapping[key] is not None}
        return cls(**known)


def _singularize(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _raw_tokens(text: str) -> list[str]:
    return [_singularize(token) for token in _TOKEN.findall((text or "").lower())]


def _tokenize(text: str) -> list[str]:
    # Digits are dropped: plans are written index-free ("put the mug on the desk") while
    # actions carry object indices ("put mug 1 in desk 2"), so numbers only add noise.
    return [token for token in _raw_tokens(text) if token and not token.isdigit() and token not in _STOPWORDS]


def _tokens_match(left: str, right: str) -> bool:
    """Exact match, or a shared prefix long enough to link compound nouns.

    ALFWorld and ScienceWorld name objects as compounds ("sinkbasin", "diningtable",
    "coffeemachine") while plans use the short form ("sink", "table", "machine").
    """
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= _MIN_PREFIX and longer.startswith(shorter)


def _overlap(plan_args: list[str], action_args: list[str]) -> int:
    used = [False] * len(action_args)
    shared = 0
    for plan_token in plan_args:
        for index, action_token in enumerate(action_args):
            if not used[index] and _tokens_match(plan_token, action_token):
                used[index] = True
                shared += 1
                break
    return shared


@dataclass(frozen=True)
class _Phrase:
    """Pre-tokenized text, so the similarity matrix does not re-tokenize per cell."""

    verb: str | None
    has_raw_args: bool
    args: list[str]
    content: frozenset[str]

    @classmethod
    def build(cls, text: str, *, is_action: bool) -> _Phrase:
        raw = _raw_tokens(text)
        verb = raw[0] if raw and is_action else None
        content = _tokenize(text)
        args = [token for token in content if token != verb]
        return cls(verb=verb, has_raw_args=bool(raw[1:]), args=args, content=frozenset(content))


def split_plan_steps(plan: str) -> list[str]:
    """Split a generated `<workflow>` into individual plan steps."""
    text = (plan or "").strip()
    block = _WORKFLOW_BLOCK.search(text)
    if block:
        text = block.group(1)
    steps = []
    for line in text.splitlines():
        line = _STEP_PREFIX.sub("", line).strip()
        if line and _tokenize(line):
            steps.append(line)
    if steps:
        return steps
    # Single-line workflows: fall back to sentence-ish splitting.
    return [part.strip() for part in re.split(r"[.;]\s+", text) if part.strip() and _tokenize(part)]


def head_verb(action: str) -> str | None:
    """ALFWorld / ScienceWorld / WebShop actions all lead with their command word, so the
    head token says whether the agent did the planned *kind* of thing."""
    tokens = _TOKEN.findall((action or "").lower())
    return _singularize(tokens[0]) if tokens else None


def _similarity(plan: _Phrase, action: _Phrase, verb_weight: float) -> float:
    if not plan.content or not action.content:
        return 0.0

    verb_hit = 1.0 if action.verb is not None and action.verb in plan.content else 0.0

    # A truly bare command ("look", "inventory") has nothing but its verb, so the verb
    # is all the evidence there is. `click[next >]` is NOT bare: its argument merely
    # happens to be a low-content word, and it must not collect the full verb-only score.
    if not action.has_raw_args:
        return verb_hit
    if not action.args or not plan.args:
        return verb_weight * verb_hit

    shared = _overlap(plan.args, action.args)
    arg_dice = 2.0 * shared / (len(plan.args) + len(action.args))
    return verb_weight * verb_hit + (1.0 - verb_weight) * arg_dice


def similarity(plan_step: str, action: str, verb_weight: float = 0.4) -> float:
    """Rule-based similarity: does the verb match, and do the arguments overlap?

    sim = verb_weight * 1[verb(action) in plan_step] + (1 - verb_weight) * dice(args)

    Verb and arguments are scored separately on purpose. A single bag-of-words score
    lets a verb-only hit ("go to the desk" vs "go to sinkbasin 1") pass the threshold,
    because the shared verb dominates a two-token action. Splitting them caps such a
    hit at `verb_weight`, below the default tau.

    Dice is used for the arguments because it tolerates the length gap between a
    verbose plan step and a terse environment command better than Jaccard.
    """
    return _similarity(_Phrase.build(plan_step, is_action=False), _Phrase.build(action, is_action=True), verb_weight)


def align_monotonic(sims: list[list[float]]) -> list[int | None]:
    """Order-preserving alignment f(k) maximizing total similarity.

    Needleman-Wunsch with zero gap cost, so plan steps and actions may both be skipped.
    Monotonicity is what makes this measure "executed in the planned order" rather than
    "executed the planned things in any order".
    """
    n_plan = len(sims)
    n_act = len(sims[0]) if n_plan else 0
    if not n_plan or not n_act:
        return [None] * n_plan

    dp = [[0.0] * (n_act + 1) for _ in range(n_plan + 1)]
    # 0 = skip plan step, 1 = skip action, 2 = match
    choice = [[0] * (n_act + 1) for _ in range(n_plan + 1)]
    for k in range(1, n_plan + 1):
        for j in range(1, n_act + 1):
            best, move = dp[k - 1][j], 0
            if dp[k][j - 1] > best:
                best, move = dp[k][j - 1], 1
            matched = dp[k - 1][j - 1] + sims[k - 1][j - 1]
            if matched > best:
                best, move = matched, 2
            dp[k][j], choice[k][j] = best, move

    match: list[int | None] = [None] * n_plan
    k, j = n_plan, n_act
    while k > 0 and j > 0:
        move = choice[k][j]
        if move == 2:
            match[k - 1] = j - 1
            k, j = k - 1, j - 1
        elif move == 1:
            j -= 1
        else:
            k -= 1
    return match


def plan_consistency_reward(plan: str, actions: list[str], config: PlanRewardConfig) -> tuple[float, dict[str, Any]]:
    """R_con: fraction of plan steps matched in order, minus a length-mismatch term."""
    plan_steps = split_plan_steps(plan)
    actions = [action for action in actions if (action or "").strip()]
    n_plan, n_act = len(plan_steps), len(actions)
    if not n_plan or not n_act:
        return 0.0, {"plan_steps": n_plan, "action_steps": n_act, "coverage": 0.0, "length_mismatch": 0.0, "matched": 0}

    plan_phrases = [_Phrase.build(step, is_action=False) for step in plan_steps]
    action_phrases = [_Phrase.build(action, is_action=True) for action in actions]
    sims = [[_similarity(plan, action, config.verb_weight) for action in action_phrases] for plan in plan_phrases]
    match = align_monotonic(sims)
    matched = sum(1 for k, j in enumerate(match) if j is not None and sims[k][j] >= config.tau)

    coverage = matched / n_plan
    length_mismatch = abs(n_plan - n_act) / max(n_plan, n_act)
    reward = coverage - config.lambda_len * length_mismatch
    info = {
        "plan_steps": n_plan,
        "action_steps": n_act,
        "matched": matched,
        "coverage": coverage,
        "length_mismatch": length_mismatch,
        "best_sims": [round(max(row), 4) for row in sims],
    }
    return reward, info


def length_penalty(n_steps: int, max_steps: int, coeff: float) -> float:
    """-coeff * used/limit, meant to be applied only on a completed task."""
    if max_steps <= 0:
        return 0.0
    return -coeff * min(max(n_steps, 0) / max_steps, 1.0)


def shape_reward(
    *,
    env_reward: float,
    n_steps: int,
    max_steps: int,
    actions: list[str],
    plan: str | None,
    config: PlanRewardConfig,
) -> tuple[float, dict[str, Any]]:
    """Return (delta, diagnostics) to add on top of the environment reward.

    Both terms are gated on task completion. Paying R_con on a failed trajectory would
    reward an agent for writing an easy-to-follow plan and following it into failure.
    """
    success = env_reward > config.success_threshold
    info: dict[str, Any] = {"env_reward": float(env_reward), "success": success}
    delta = 0.0

    # R_con additionally requires plan mode: without an injected workflow there is
    # nothing to be consistent with.
    if config.consistency_enabled and success and plan:
        r_con, con_info = plan_consistency_reward(plan, actions, config)
        delta += config.consistency_weight * r_con
        info["r_con"] = r_con
        info["r_con_weighted"] = config.consistency_weight * r_con
        info["plan_consistency"] = con_info

    # The length penalty applies with or without a plan, so that an agent which fails
    # fast is not rewarded for giving up early.
    if config.length_penalty_enabled and success:
        r_len = length_penalty(n_steps, max_steps, config.length_penalty_coeff)
        delta += r_len
        info["r_len"] = r_len

    info["reward_delta"] = delta
    return delta, info
