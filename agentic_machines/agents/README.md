# Agents

Concrete operator implementations built on `core.BaseOperator`. Each agent/operator in this module is a callable that accepts input and returns a `StepResult`.

---

## Components

### BaseAgent

**`base_agent.py`** — A full agentic loop with tool calling, conversation management, and cost tracking.

`BaseAgent` is the main workhorse. Given a task string, it runs a multi-step loop: call an LLM, dispatch tool calls to registered actions, collect results, and repeat until the LLM calls the built-in `terminate` tool or `max_step` is reached.

#### Lifecycle

```
agent(task: str) → StepResult
│
├─ 1. Reset state (if reset_before_call)
├─ 2. Append user message
├─ 3. Agent Loop (up to max_step):
│     a. Prune messages (if configured)
│     b. Call LLM with tools → get response
│     c. If "terminate" called → collect answer, break
│     d. Execute tool calls → append results
│     e. Save step checkpoint
└─ 4. Return StepResult
```

#### Built-in `terminate` Tool

Every `BaseAgent` automatically includes a `terminate` tool. The LLM calls it to signal completion:

```json
{
  "name": "terminate",
  "parameters": {
    "answer": "The final answer",
    "finish_reason": "success | failure | error",
    "result_references": "Source references or N/A"
  }
}
```

#### Usage

```python
from agentic_machines import BaseAgent
from agentic_machines.config import get_llm_config

agent = BaseAgent(
    name="my_agent",
    description="An agent that does things.",
    system_msg="You are a helpful assistant.",
    llm_config=get_llm_config("gpt-4o", cache_seed=42),
    action_schemas=[my_tool_schema],
    allowed_action_dict={"my_tool": my_tool_function},
    max_step=10,
    reset_before_call=True,
    save_path="./runs",
)

result = agent("Do the task")
print(result.output)
print(result.finish_reason)       # "success", "failure", "max_step_limit", etc.
print(result.info["llm_usage"])   # Token counts and costs per model
```

---

### CUAOperator

**`cua_operator.py`** — Wraps the OpenAI Computer-Use API (`computer-use-preview`) for screen-level desktop interaction.

This is a **single-call** operator (not a loop). It sends one request to the CUA API and returns the response with cleaned history items ready for the next turn.

#### Usage

```python
from agentic_machines import CUAOperator

cua = CUAOperator(screen_width=1920, screen_height=1080, environment="linux")

result = cua(messages=[
    {"role": "user", "content": "Click the Firefox icon on the taskbar"}
])
# result.raw_response  — full API response
# result.input_items   — cleaned items to append to history for next turn
# result.cost          — API call cost
```

