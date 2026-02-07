"""
OpenAI Computer Use Agent (CUA) Operator implementation.

This operator makes a single call to the OpenAI Computer Use API.
It is stateless and focused on API interaction only.
"""
import logging
from typing import Any, Dict, List, Tuple

from agentic_machines.core.base_operator import BaseOperator
from agentic_machines.core.step_result import StepResult, StepInfo
from agentic_machines.utils.llm_utils import LLMCaller

logger = logging.getLogger("desktopenv.cua_operator")


def _to_input_items(output_items: list) -> list:
    """
    Convert `response.output` into the JSON-serialisable items we're allowed
    to resend in the next request. Following the guidance from the Responses API:
    - Only include reasoning items when paired with a computer_call or message
    - Preserve item IDs to maintain proper pairing
    - Remove the status field which is ignored on input
    
    Based on the blog post and API errors, reasoning items should only be included 
    if they're paired with a computer_call or message in the output. The pairing 
    is determined by position: a reasoning item immediately followed by a computer_call or message.
    """
    cleaned: List[Dict[str, Any]] = []
    
    # First pass: identify which reasoning items are paired with computer_calls or messages
    reasoning_ids_to_keep = set()
    
    for i, item in enumerate(output_items):
        raw: Dict[str, Any] = item if isinstance(item, dict) else item.model_dump()
        typ = raw.get("type", "")
        
        # Check if this is a reasoning item
        if typ == "reasoning":
            reasoning_id = raw.get("id", "")
            # Look ahead to see if the next item(s) contain a computer_call or message
            has_paired_item = False
            for j in range(i + 1, len(output_items)):
                next_item = output_items[j]
                next_raw = next_item if isinstance(next_item, dict) else next_item.model_dump()
                next_typ = next_raw.get("type", "")
                
                if next_typ == "computer_call" or next_typ == "message":
                    # Found a computer_call or message after this reasoning item
                    has_paired_item = True
                    break
                elif next_typ == "reasoning":
                    # Hit another reasoning item, stop looking
                    break
            
            if has_paired_item:
                reasoning_ids_to_keep.add(reasoning_id)
                logger.debug(f"Keeping reasoning item {reasoning_id} (paired with computer_call or message)")
            else:
                logger.debug(f"Skipping reasoning item {reasoning_id} (not paired with computer_call or message)")

    # Second pass: build the cleaned list
    for item in output_items:
        raw: Dict[str, Any] = item if isinstance(item, dict) else item.model_dump()
        typ = raw.get("type", "")
        item_id = raw.get("id", "")
        
        # Skip reasoning items that aren't paired with computer_calls
        if typ == "reasoning" and item_id not in reasoning_ids_to_keep:
            continue
        
        # Remove status field (it's ignored on input)
        raw.pop("status", None)
        cleaned.append(raw)

    return cleaned




class CUAOperator(BaseOperator):
    """OpenAI Computer Use Agent (CUA) operator.
    
    This operator makes a single API call to OpenAI Computer Use API.
    It handles the low-level API interaction with proper error handling and retry logic.
    """
    
    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        environment: str = "linux",
    ):
        """Initialize the CUA operator.
        
        Args:
            screen_width: Display width for computer use
            screen_height: Display height for computer use
            environment: Operating system environment ("linux", "windows", "macos")
        """
        super().__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.environment = environment
        self._schema = {
            "function": {
                "name": "cua_operator",
                "description": "Make a single call to OpenAI Computer Use API",
                "parameters": {
                    "messages": {
                        "type": "list",
                        "description": "List of input messages"
                    }
                }
            }
        }
    
    def __call__(
        self,
        messages: list,
    ) -> StepResult:
        """Execute a single CUA API call.
        
        Args:
            messages: List of input items for the conversation history.
                          Must be properly formatted for the OpenAI CUA API.
            
        Returns:
            StepResult with parsed output including:
                - raw_response: The original API response object
                - raw_output: The response.output list (unparsed)
                - cost: API call cost
                - input_items: Cleaned items ready to add to history
        """
        response, cost = LLMCaller.call_openai_cua(
            messages=messages,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            environment=self.environment,
            logger=logger
        )
        
        logger.info(f"CUA API call cost: ${cost:.6f}")
        
        # Get cleaned history items to add
        input_items = _to_input_items(response.output)
        
        # Return structured StepResult
        return StepResult(
            kind="action",
            name=self.name,
            finish_reason="success",
            info=StepInfo(),
            raw_response=response,
            cost=cost,
            input_items=input_items
        )
