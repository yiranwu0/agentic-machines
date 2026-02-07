import json
import inspect
import asyncio
import time
import os
from typing import Optional, Dict, Any, Callable, List
from agentic_machines.core.base_operator import BaseOperator
from agentic_machines.core.step_result import StepResult
from agentic_machines.utils.llm_utils import LLMCaller


SUBMIT_ANSWER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "terminate",
        "description": "This marks the end of the task and it terminates the process. If the task is done/failed, call this tool to terminate the process and submit the final answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to be submitted. Leave empty if the task doesn't require a specific answer."
                },
                "finish_reason": {
                    "type": "string",
                    "enum": ["success", "failure", "error"],
                    "description": "The reason for finishing the task."
                },
                "result_references": {
                    "type": "string",
                    "description": "The source/reference(s) to the result, if any. Use 'N/A' when not applicable."
                }
            },
            "required": ["finish_reason", "result_references"],
            "additionalProperties": False
        },
        "strict": True
    }
}


def update_model_usage(usage_dict: dict, model: str, usage: dict, cost: float):
    """Update the model usage statistics."""
    if model not in usage_dict:
        usage_dict[model] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0
        }
    usage_dict[model]["prompt_tokens"] += usage.get("prompt_tokens", 0)
    usage_dict[model]["completion_tokens"] += usage.get("completion_tokens", 0)
    usage_dict[model]["total_tokens"] += usage.get("total_tokens", 0)
    usage_dict[model]["cost"] += cost
    usage_dict["total_cost"] = usage_dict.get("total_cost", 0) + cost


class BaseAgent(BaseOperator):
    def __init__(
        self,
        name: str,
        description: str,
        action_schemas: List[Dict[str, Any]],  # list of action schemas (tools)
        system_msg: str,
        llm_config: dict,
        allowed_action_dict: Optional[Dict[str, Any]] = None,  # mapping from action name to function/operator
        summarize_llm_config: Optional[dict] = None,
        max_step: int = 15,
        message_pruning_function: Optional[Callable[[List[dict]], tuple[bool, List[dict]]]] = None,
        reset_before_call: bool = False,
        save_path: Optional[str] = None,
    ):
        super().__init__()

        # Tool schema for calling this agent (agents are callable tools with a single string task).
        self._schema = {
            "type": "function",
            "function": {
                "name": str(name),
                "description": str(description),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task to be performed.",
                        }
                    },
                    "required": ["task"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
        self.save_path = save_path
        self.save_id = 0
        self.log_file = None

        # LLM configurations
        self.llm_config = llm_config
        self.summarize_llm_config = summarize_llm_config if summarize_llm_config else None
        self.reset_before_call = reset_before_call

        # Agent parameters
        self.system_msg = system_msg
        self.max_step = max_step
        self.message_pruning_function = message_pruning_function

        # Action schemas - include terminate tool
        self.action_schemas = [SUBMIT_ANSWER_TOOL_SCHEMA] + action_schemas

        # Build action dictionary from schemas
        self.allowed_action_dict: Dict[str, Any] = allowed_action_dict

        # Initialize state
        self.reset()

    def reset(self):
        """Reset the agent to its initial state."""
        self.messages = [{"role": "system", "content": self.system_msg}]
        self.usage = {"total_cost": 0}
        self.step_results = []

    def _log_print(self, *args, **kwargs):
        """Print to stdout and append the same text to the agent log file."""
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(str(arg) for arg in args) + end
        print(*args, **kwargs)
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(message)
            except Exception:
                pass

    def _add_msg(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
        })

    def _message_pruning(self):
        """Apply message pruning if a pruning function is provided."""
        if self.message_pruning_function is not None:
            is_pruned, pruned_messages = self.message_pruning_function(self.messages)
            if is_pruned:
                self.messages = pruned_messages
                return True
        return False

    def _get_main_response(self) -> dict:
        """Get response from LLM with tools."""
        llm_config = self.llm_config.copy()
        llm_config['tools'] = self.action_schemas
        response = LLMCaller.call_llm_with_msgs(self.messages, llm_config)
        # print(f"in _get_main_response   {response} ")
        update_model_usage(self.usage, response.model, response.usage.model_dump(), response.cost)
        return response.choices[0].message.model_dump()

    def _collect_submit_answer(self, response_dict: dict) -> StepResult:
        """Collect and process the terminate tool call."""
        tool_calls = response_dict.get("tool_calls") or []
        terminate_call = next(
            (tc for tc in tool_calls if (tc.get("function") or {}).get("name") == "terminate"),
            None,
        )
        if terminate_call is None:
            return None

        arguments = (terminate_call.get("function") or {}).get("arguments", "{}")
        arguments = json.loads(arguments)

        # Parse the answer
        output = arguments.get("answer", "")
        if output is None:
            output = ""

        result = StepResult(
            kind="agent",
            name=self.name,
            output=output,
            result_references=arguments.get("result_references", "N/A"),
            finish_reason=arguments.get("finish_reason", "success"),
            interactions=self.process_messages(self.messages),
            info={
                "operation_summary": "Task completed via terminate",
                "llm_usage": self.usage,
                "step_results": [x.model_dump() if isinstance(x, StepResult) else x for x in self.step_results]
            }
        )
        return result

    def _execute_tool(self, message: dict):
        """Execute tools from the assistant message."""
        tool_returns = []
        for tool_call in message.get("tool_calls", []):
            function_call = tool_call.get("function", {})
            tool_call_id = tool_call.get("id", None)

            func_name = function_call.get("name", None)
            func = self.allowed_action_dict.get(func_name, None) if self.allowed_action_dict else None

            if func is None:
                tool_returns.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "content": f"Error: Tool '{func_name}' not found."
                })
                continue

            # Handle async functions/operators with async __call__ method
            if inspect.iscoroutinefunction(getattr(func, '__call__', None)):
                try:
                    # get the running loop if it was already created
                    loop = asyncio.get_running_loop()
                    close_loop = False
                except RuntimeError:
                    # create a loop if there is no running loop
                    loop = asyncio.new_event_loop()
                    close_loop = True

                is_success, func_return = loop.run_until_complete(self._a_execute_function(function_call))
                if close_loop:
                    loop.close()
            else:
                _, func_return = self._execute_function(function_call)

            # Handle StepResult
            if isinstance(func_return, StepResult):
                self.step_results.append(func_return)
                content = f"""Finish Reason: {func_return.finish_reason}
Output: {func_return.output}
Result References: {getattr(func_return, 'result_references', 'N/A')}
"""
                if content is None:
                    content = getattr(func_return, "output", None)
                if content is None:
                    content = str(func_return.model_dump())
            else:
                content = str(func_return if func_return else "")

            tool_call_response = {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "content": content,
            }
            tool_returns.append(tool_call_response)

        return tool_returns

    async def _a_execute_function(self, func_call) -> tuple[bool, Any]:
        """Execute an async function call and return the result.

        Args:
            func_call: Dictionary with keys "name" and "arguments".

        Returns:
            Tuple of (is_exec_success, result).
            is_exec_success: Whether execution was successful.
            result: The function's return value or error message.
        """
        func_name = func_call.get("name", "")
        func = self.allowed_action_dict.get(func_name, None)

        if func is None:
            return False, f"Error: Tool '{func_name}' not found."

        try:
            arguments = json.loads(func_call.get("arguments", "{}"))
        except json.JSONDecodeError as e:
            return False, f"Error parsing arguments: {str(e)}"

        try:
            # Check if it's an async operator with __call__ method
            if inspect.iscoroutinefunction(getattr(func, '__call__', None)):
                result = await func(**arguments)
            else:
                result = func(**arguments)
            return True, result
        except Exception as e:
            return False, f"Error executing {func_name}: {str(e)}"

    def quick_bash_script(self, script: str) -> str:
        """Quickly run a bash script in the environment and return output."""
        action_dict = {
            "name": "run_bash_script",
            "arguments": json.dumps({
                "script": script
            })
        }
        is_success, result = self._execute_function(action_dict)
        self._log_print(f"Quick bash script executed. Success: {is_success}")
        self._log_print(f"Result: {result}")

    def _execute_function(self, func_call) -> tuple[bool, Any]:
        """Execute a function call and return the result.

        Args:
            func_call: Dictionary with keys "name" and "arguments".
        Returns:
            Tuple of (is_exec_success, result).
            is_exec_success: Whether execution was successful.
            result: The function's return value or error message.
        """
        func_name = func_call.get("name", "")
        func = self.allowed_action_dict.get(func_name, None)

        if func is None:
            return False, f"Error: Tool '{func_name}' not found."

        try:
            arguments = json.loads(func_call.get("arguments", "{}"))
        except json.JSONDecodeError as e:
            return False, f"Error parsing arguments: {str(e)}"

        try:
            result = func(**arguments)
            return True, result
        except Exception as e:
            return False, f"Error executing {func_name}: {str(e)}"

    def _print_agent_response(self, response_dict: dict, iteration: int):
        """Print a beautified agent response showing content and function calls."""
        self._log_print("\n" + "="*80, flush=True)
        self._log_print(f"🤖 AGENT RESPONSE (Iteration {iteration}/{self.max_step})", flush=True)
        self._log_print("="*80, flush=True)

        if response_dict.get('content'):
            self._log_print(f"💬 Content: {response_dict['content']}", flush=True)

        if response_dict.get('tool_calls'):
            self._log_print("\n🔧 Function Calls:", flush=True)
            for idx, tc in enumerate(response_dict.get('tool_calls', []), 1):
                func_name = tc['function'].get('name', 'unknown')
                args = tc['function'].get('arguments', '{}')
                try:
                    args_dict = json.loads(args)
                    args_str = ""
                    for k, v in args_dict.items():
                        args_str += f"{k}: {v}, "
                except:
                    args_str = args
                self._log_print(f"\n  [{idx}] Function: {func_name}", flush=True)
                self._log_print(f"      Arguments: {args_str}", flush=True)

        self._log_print("="*80 + "\n", flush=True)

    def _print_tool_returns(self, tool_returns: List[dict]):
        """Print beautified tool return results."""
        self._log_print("\n" + "="*80, flush=True)
        self._log_print(f"📥 TOOL RETURNS ({len(tool_returns)} tool{'s' if len(tool_returns) != 1 else ''})", flush=True)
        self._log_print("="*80, flush=True)

        for idx, tr in enumerate(tool_returns, 1):
            tool_call_id = tr.get('tool_call_id', 'unknown')
            content = tr.get('content', '')

            # Try to format JSON content if possible
            try:
                if content and content.strip().startswith('{'):
                    content_dict = json.loads(content)
                    content_display = json.dumps(content_dict, indent=2)
                else:
                    content_display = content
            except:
                content_display = content

            self._log_print(f"\n  [{idx}] Tool Call ID: {tool_call_id}", flush=True)
            self._log_print(f"      Content Length: {len(content)} characters", flush=True)

            if len(content_display) > 4000:
                self._log_print(f"      Content (first 4000 chars):\n{content_display[:4000]}", flush=True)
                self._log_print(f"      ... (truncated {len(content_display) - 4000} characters)", flush=True)
            else:
                self._log_print(f"      Content:\n{content_display}", flush=True)

        self._log_print("="*80 + "\n", flush=True)

    def _save_run(self, task, step_result: StepResult):
        """Save the agent run details to a JSON file."""
        if self.save_path is None:
            return

        save_data = {
            "task": task,
            "step_result": step_result.model_dump() if isinstance(step_result, StepResult) else step_result,
        }

        save_file = f"{self.save_path}/agent_{self.name}_run_{self.save_id}.json"
        with open(save_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        self._log_print(f"Agent run saved to {save_file}")

    def _save_step_checkpoint(self, task: str, step_number: int):
        """Save partial results into a single rolling checkpoint file."""
        if self.save_path is None:
            return

        checkpoint_data = {
            "task": task,
            "step_number": step_number,
            "messages": self.messages,
            "usage": self.usage,
            "step_results_count": len(self.step_results),
            "step_results": [x.model_dump() if isinstance(x, StepResult) else x for x in self.step_results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

        checkpoint_file = f"{self.save_path}/agent_{self.name}_run_{self.save_id}_checkpoint.json"
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            # print(f"✓ Step {step_number} checkpoint saved to {checkpoint_file}")
        except Exception as e:
            self._log_print(f"⚠ Warning: Could not save step checkpoint: {str(e)}", flush=True)

    def process_messages(self, messages: List[dict]) -> List[dict]:

        # refusal\tannotations\taudio\tfunction_call

        # remove any of the above keys if it is null, or have an empty list

        # if it contains tool_calls, take of each of the tool call, create a field function_i, where i is the index of the tool call, take the msg['tool_calls'][i]['function']['name'] and ['arguments']], form a new str f"NAME: {name}\nARGUMENTS: {arguments}"
        processed_messages = []
        for msg in messages:
            new_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            # remove unwanted keys directly for now
            # for key in ["refusal", "annotations", "audio", "function_call"]:
            #     if key in msg and (msg[key] is None or (isinstance(msg[key], list) and len(msg[key]) == 0)):
            #         continue
            #     elif key in msg:
            #         new_msg[key] = msg[key]

            # process tool_calls
            if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                for i, tool_call in enumerate(msg["tool_calls"]):
                    func = tool_call.get("function", {})
                    name = func.get("name", "")
                    arguments = func.get("arguments", "")
                    new_msg[f"function_{i}"] = f"NAME: {name}\nARGUMENTS: {arguments}"
            processed_messages.append(new_msg)
        return processed_messages

    def __call__(self, task: str) -> StepResult:
        """
        Execute the agent with message view.

        Args:
            task (str): The task or problem description to be solved by the agent.
        Returns:
            StepResult: The result of the agent execution
        """
        if self.reset_before_call:
            self.reset()
            self.save_id += 1
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            self.log_file = os.path.join(self.save_path, f"agent_{self.name}_prints_{self.save_id}.log")
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("")
        self._add_msg("user", task.strip())

        result = None
        # Main agent loop
        for i in range(self.max_step):
            # Apply message pruning if configured
            self._message_pruning()

            # Get response from LLM
            response_dict = self._get_main_response()
            self.messages.append(response_dict)

            # Print agent response
            self._print_agent_response(response_dict, i + 1)

            # Check for terminate
            if response_dict.get("tool_calls") and \
               any(tc['function'].get("name") == "terminate"
                   for tc in response_dict.get("tool_calls", [])):
                result = self._collect_submit_answer(response_dict)
                if result:
                    # Save final checkpoint
                    self._save_step_checkpoint(task, i + 1)
                    break

            # Execute tools if present
            if response_dict.get("tool_calls"):
                tool_returns = self._execute_tool(response_dict)
                self._print_tool_returns(tool_returns)
                self.messages.extend(tool_returns)
            else:
                # No tool calls, just continue
                self._add_msg(
                    role="user",
                    content="Continue. When finished, carefully read the task requirement and submit the final answer with correct format."
                )

            # Save step checkpoint
            self._save_step_checkpoint(task, i + 1)

        # Max iterations reached
        if result is None:
            output = "N/A"
            if self.summarize_llm_config:
                # Attempt to summarize
                try:
                    summary_prompt = "Please summarize what has been accomplished so far."
                    self._add_msg("user", summary_prompt)
                    response_dict = self._get_main_response()
                    output = response_dict.get("content", "N/A")
                    self._log_print(response_dict, flush=True)
                except Exception:
                    self._log_print(
                        f"Error during summarization with model {self.summarize_llm_config.get('model', 'unknown')}",
                        flush=True,
                    )
                    output = "Max iterations reached, could not summarize."

            result = StepResult(
                kind="agent",
                name=self.name,
                output=output,
                finish_reason="max_step_limit",
                result_references="N/A",
                interactions=self.process_messages(self.messages),
                info={
                    "operation_summary": "Maximum iterations reached without completion",
                    "llm_usage": self.usage,
                    "step_results": [x.model_dump() if isinstance(x, StepResult) else x for x in self.step_results]
                }
            )

        self._save_run(task, result)
        return result

    def run(self, task) -> StepResult:
        """Alias for __call__ method."""
        return self.__call__(task)
