from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Callable
import copy

# Default pruning configuration mirrors ag2agent.agentic.agentic_action.BaseAgenticAction
# message_pruning_config when a boolean is provided.
DEFAULT_PRUNING_CONFIG = {
    "per_message_len": 1000,
    "full_msg_count": 50,
    "acient_message_cut": 10,      # Note: keeping the original key spelling for compatibility
    "acient_message_len": 300,
}

TRUNCATION_SUFFIX = "..."


def _truncate_text(text: str, limit: int) -> str:
    if text is None:
        return text
    if len(text) <= limit or text.endswith(TRUNCATION_SUFFIX):
        return text
    return text[:limit] + TRUNCATION_SUFFIX


def prune_tool_messages(
    messages: List[Dict[str, Any]],
    *,
    config: Optional[Dict[str, int]] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Prune older tool messages to reduce context size while preserving recent tool outputs.

    - Traverse messages from back to front.
    - Leave the latest `full_msg_count` tool messages intact.
    - For older tool messages beyond that window, truncate content to `per_message_len`.
    - For even older tool messages beyond an additional `acient_message_cut`,
      truncate further to `acient_message_len`.
    - Avoid double-truncation if a message already ends with the truncation suffix.

    Args:
        messages: Conversation history (list of role-content dicts).
        config: Optional dictionary overriding DEFAULT_PRUNING_CONFIG.

    Returns:
        (is_pruned, pruned_messages)
        - is_pruned: True if any message content was modified.
        - pruned_messages: A new messages list (original preserved).
    """
    if not messages:
        return False, messages

    cfg = {**DEFAULT_PRUNING_CONFIG, **(config or {})}
    full_msg_count = int(cfg.get("full_msg_count", DEFAULT_PRUNING_CONFIG["full_msg_count"]))
    per_message_limit = int(cfg.get("per_message_len", DEFAULT_PRUNING_CONFIG["per_message_len"]))
    ancient_message_limit = int(cfg.get("acient_message_len", DEFAULT_PRUNING_CONFIG["acient_message_len"]))
    ancient_message_cut = int(cfg.get("acient_message_cut", DEFAULT_PRUNING_CONFIG["acient_message_cut"]))

    pruned = False
    tool_msg_count = 0

    # Work on a shallow copy of list and deepcopy individual dicts we modify
    pruned_messages: List[Dict[str, Any]] = list(messages)

    # Iterate from end to start
    for idx in range(len(pruned_messages) - 1, -1, -1):
        msg = pruned_messages[idx]
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role != "tool":
            continue

        tool_msg_count += 1
        content = msg.get("content")

        # Keep the most recent `full_msg_count` tool messages intact
        if tool_msg_count <= full_msg_count:
            continue

        # Decide truncation level for older tool messages
        limit = (
            ancient_message_limit
            if tool_msg_count > full_msg_count + ancient_message_cut
            else per_message_limit
        )

        new_content = _truncate_text(content, limit)
        if new_content != content:
            # copy-on-write for the modified dict
            new_msg = copy.copy(msg)
            new_msg["content"] = new_content
            pruned_messages[idx] = new_msg
            pruned = True

    return pruned, pruned_messages


def make_message_pruner(config: Optional[Dict[str, int]] = None) -> Callable[[List[Dict[str, Any]]], Tuple[bool, List[Dict[str, Any]]]]:
    """
    Create a pruning function that matches BaseAgent.message_pruning_function signature.

    Usage:
        from gdpagent.message_pruning import make_message_pruner
        pruner = make_message_pruner({"per_message_len": 1200})
        agent = BaseAgent(
            name=..., description=..., action_schemas=..., allowed_action_dict=...,
            system_msg=..., llm_config=...,
            message_pruning_function=pruner,
        )
    """
    cfg = {**DEFAULT_PRUNING_CONFIG, **(config or {})}

    def _prune(messages: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        return prune_tool_messages(messages, config=cfg)

    return _prune


# A convenient default pruner users can import directly.
# Matches the signature expected by BaseAgent.

def default_message_pruner(messages: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
    return prune_tool_messages(messages, config=DEFAULT_PRUNING_CONFIG)
