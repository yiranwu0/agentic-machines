# Agentic Machines

A lightweight, operator-based agent framework for building LLM-powered agents that interact with desktop environments. Designed around composable **operators** — atomic units of work that can be wired together into agentic loops with tool use, memory management, and cost tracking out of the box.

---

## Architecture

```
agentic_machines/
├── core/               # Foundational abstractions
│   ├── base_operator   # Abstract base class for all operators
│   └── step_result     # Pydantic models for structured step outputs (StepResult, StepInfo)
├── agents/             # Concrete operator implementations
│   ├── base_agent      # General-purpose agentic loop with tool calling
│   ├── cua_operator    # OpenAI Computer-Use Agent (CUA) API operator
│   └── llm_operator    # Simple single-call LLM operator
├── memory/             # Context-window management
│   └── message_pruning # Configurable truncation of older tool messages
└── utils/              # Shared helpers
    ├── llm_utils       # LLMCaller — unified LLM interface (via ag2/autogen)
    └── schema_utils    # YAML/dict → OpenAI function-tool schema converters
```

### Key Concepts

#### Core (Foundational Abstractions)

These are the building blocks that everything else is built on:

| Concept | Description |
|---|---|
| **`BaseOperator`** | Abstract base class. Every operator exposes a JSON `schema()` (name + description + parameters) and is callable. All agents and operators inherit from this. |
| **`StepResult`** | Pydantic model returned by every operator call. Carries `kind`, `name`, `finish_reason`, timestamps (`StepInfo`), and arbitrary extra fields. This is the universal output contract. |

#### Applications (Built on Core)

Concrete implementations that use the core abstractions — documented in their respective module READMEs:

| Module | Components | README |
|---|---|---|
| **`agents/`** | `BaseAgent`, `CUAOperator`, `LLMOperator` | [agents/README.md](agents/README.md) |
| **`utils/`** | `LLMCaller`, `schema_utils` | [utils/README.md](utils/README.md) |
| **`memory/`** | `message_pruning` | [memory/README.md](memory/README.md) |

---

## Installation

```bash
# From the agentic_machines directory (editable install)
pip install -e .
```

### Requirements

- Python ≥ 3.9
- [`ag2`](https://github.com/ag2ai/ag2) (AutoGen)
- `python-dotenv`

---

## Environment Setup

1. Copy the example env file and fill in your API keys:

```bash
cp .env_example .env
```

2. Populate the keys you need:

```env
OPENAI_KEY=sk-...          # Required — used by all OpenAI models
```

3. Initialize the model configs at the start of your script:

```python
from agentic_machines.config import set_llm_caller_config

set_llm_caller_config(cache_seed=42)
```

---

## Quick Start

### 1. Run a simple LLM call

```python
from agentic_machines import LLMCaller
from agentic_machines.config import set_llm_caller_config

set_llm_caller_config(cache_seed=42)

response = LLMCaller.call_llm(
    system_msg="You are a helpful assistant.",
    task="What is the capital of France?",
    model="gpt-4o-mini",
)
print(response)  # "The capital of France is Paris."
```

### 2. Build a tool-using agent

```python
from agentic_machines import BaseAgent
from agentic_machines.config import get_llm_config
from agentic_machines.memory import make_message_pruner

# Define a custom tool
def search_web(query: str) -> str:
    return f"Search results for: {query}"

search_schema = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

agent = BaseAgent(
    name="research_agent",
    description="An agent that researches topics using web search.",
    system_msg="You are a research assistant. Use tools to find information.",
    llm_config=get_llm_config("gpt-4o", cache_seed=42),
    action_schemas=[search_schema],
    allowed_action_dict={"search_web": search_web},
    max_step=10,
    message_pruning_function=make_message_pruner(),
    reset_before_call=True,
    save_path="./runs",
)

result = agent("Find the population of Tokyo")
print(result.output)
print(f"Total cost: ${result.info['llm_usage']['total_cost']:.4f}")
```
