
from agentic_machines.utils import LLMCaller
from agentic_machines.core import StepResult, BaseOperator


BASE_SCHEMA = {
                "name": "LLMOperator",
                "description": "Operator that interacts with LLMs using LLMCaller.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "description": "List of messages to send to the LLM.",
                        }
                    },
                    "required": ["messages"],
                    "additionalProperties": False
                }
            }


class LLMOperator(BaseOperator):
    """Operator that uses LLMCaller to interact with LLMs."""
    def __init__(
            self,
            model: str,
            system_msg: str,
            schema: dict = None,
    ):
        self.model = model
        self.system_msg = system_msg
        self._schema = schema or BASE_SCHEMA
        self._validate_schema()
        
    def __call__(self, messages: list) -> StepResult:
        response = LLMCaller.call_llm(
            model=self.model,
            system_msg=self.system_msg,
            messages=messages
        )
        return StepResult(
            kind="agent",
            name="LLMOperator",
            finish_reason="success",
            response=response
        )