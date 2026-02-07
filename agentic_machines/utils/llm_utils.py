import autogen
import time
from typing import Any, Tuple
from openai import OpenAI
import openai

class LLMCaller:
    llm_config = None

    # @staticmethod
    # def set_config_list(self, config_list: None):
    #     if config_list:
    #         LLMCaller.llm_config["config_list"] = config_list

    @staticmethod
    def set_llm_config(llm_config: None):
        LLMCaller.llm_config = llm_config
    
    @staticmethod
    def call_llm(
            system_msg: str, 
            task: str,
            model:str,
            response_format=None,
            raw_response = False,
            return_cost = False,
            reasoning_effort = None,
            ):
        if LLMCaller.llm_config is None:
            raise ValueError("LLM config is not set")

        tmp_config = LLMCaller.llm_config.copy()
        # tmp_config['temperature'] = 0.7
        # filter out the model
        tmp_config['config_list'] = [x for x in tmp_config['config_list'] if x['model'] == model]
        if 'model' in tmp_config:
            tmp_config['model'] = model
        if len(tmp_config) == 0:
            raise ValueError(f"Model {model} is not found in the config")
        if "o3" in model:
            assert reasoning_effort is not None, "Reasoning effort is required for o3 model"
            tmp_config["reasoning_effort"] = reasoning_effort
            
        if response_format:
            tmp_config["response_format"] = response_format
        client = autogen.OpenAIWrapper(**tmp_config)
        response = client.create(
            messages = [
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': task}
            ]
        )
        if raw_response:
            if return_cost:
                return response.choices[0].message.content, response.cost, response.usage
            return response
        if return_cost:
            return response.choices[0].message.content, response.cost, response.usage
        return response.choices[0].message.content
    

    @staticmethod
    def call_llm_with_config(
            system_msg: str,
            task: str,
            llm_config: dict,
            response_format=None,
    ):
        start_time = time.time()
        tmp_config = llm_config.copy()
        if response_format:
            tmp_config["response_format"] = response_format
        client = autogen.OpenAIWrapper(**tmp_config)
        response = client.create(
            messages = [
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': task}
            ]
        )
        end_time = time.time()
        response.create_time = start_time
        response.duration = end_time - start_time
        return response
    
    @staticmethod
    def call_llm_with_msgs(
            messages: list,
            llm_config: dict,
            return_usage = False,
    ):
        client = autogen.OpenAIWrapper(**llm_config)
        if return_usage:
            response = client.create(messages = messages)
            return response, client.total_usage_summary
        return client.create(messages = messages)

    @staticmethod
    def call_llm_with_tools(
            system_msg: str,
            tools, 
            task: str,
            model:str,
            return_cost = False
        ):
        if LLMCaller.llm_config is None:
            raise ValueError("LLM config is not set")

        tmp_config = LLMCaller.llm_config.copy()
        if model:
            # filter out the model
            tmp_config['config_list'] = [x for x in tmp_config['config_list'] if x['model'] == model]
            if 'model' in tmp_config:
                tmp_config['model'] = model
            if len(tmp_config) == 0:
                raise ValueError(f"Model {model} is not found in the config")
        
        tmp_config["tools"] = tools
            
        client = autogen.OpenAIWrapper(**tmp_config)
        return client.create(
            messages = [
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': task}
            ]
        )
    
    @staticmethod
    def call_openai_cua(
            messages: list,
            screen_width: int = 1920,
            screen_height: int = 1080,
            environment: str = "linux",
            logger = None
        ) -> Tuple[Any, float]:
        """Call OpenAI Computer Use API with retry logic.
        
        Args:
            messages: List of input items for the conversation history
            screen_width: Display width for computer use
            screen_height: Display height for computer use
            environment: Operating system environment ("linux", "windows", "macos")
            logger: Logger instance for logging errors and info
            
        Returns:
            Tuple of (response, cost)
        """
        GPT4O_INPUT_PRICE_PER_1M_TOKENS = 3.00
        GPT4O_OUTPUT_PRICE_PER_1M_TOKENS = 12.00
        
        if LLMCaller.llm_config is None:
            raise ValueError("LLM config is not set. Call set_llm_config() first.")
        
        # Use autogen's OpenAIWrapper with the configured API key
        tmp_config = LLMCaller.llm_config.copy()
        tmp_config['config_list'] = [x for x in tmp_config['config_list'] if x['model'] == "computer-use-preview"]
        if len(tmp_config['config_list']) == 0:
            raise ValueError("computer-use-preview model not found in the config")
        
        client = autogen.OpenAIWrapper(**tmp_config)
        retry = 0
        response = None
        last_error = None
        
        # Get the underlying OpenAI client from autogen wrapper
        openai_client = client._clients[0]._oai_client
        
        while retry < 3:
            try:
                response = openai_client.responses.create(
                    model="computer-use-preview",
                    tools=[{
                        "type": "computer_use_preview",
                        "display_width": screen_width,
                        "display_height": screen_height,
                        "environment": environment,
                    }],
                    input=messages,
                    reasoning={
                        "summary": "concise"
                    },
                    tool_choice="required",
                    truncation="auto",
                )
                break
            except openai.BadRequestError as e:
                retry += 1
                last_error = e
                if logger:
                    logger.error(f"BadRequestError in response.create (attempt {retry}/3): {e}")
                    # iterate over messages, print without image
                    for i, msg in enumerate(messages):
                        if 'image' in msg.get('content', ''):
                            logger.debug(f"History input {i}: [image content omitted]")
                        else:
                            logger.debug(f"History input {i}: {msg}")
                time.sleep(0.5)
            except openai.InternalServerError as e:
                retry += 1
                last_error = e
                if logger:
                    logger.error(f"InternalServerError in response.create (attempt {retry}/3): {e}")
                time.sleep(0.5)
            except openai.AuthenticationError as e:
                if logger:
                    logger.error(f"AuthenticationError in response.create: {e}")
                raise Exception(f"Failed to call OpenAI - Authentication Error: {e}") from e
            except openai.RateLimitError as e:
                retry += 1
                last_error = e
                if logger:
                    logger.error(f"RateLimitError in response.create (attempt {retry}/3): {e}")
                time.sleep(2.0)
            except openai.APIConnectionError as e:
                retry += 1
                last_error = e
                if logger:
                    logger.error(f"APIConnectionError in response.create (attempt {retry}/3): {e}")
                time.sleep(1.0)
            except Exception as e:
                retry += 1
                last_error = e
                if logger:
                    logger.error(f"Unexpected error in response.create (attempt {retry}/3): {type(e).__name__}: {e}")
                time.sleep(0.5)
        
        if retry == 3:
            error_msg = f"Failed to call OpenAI after 3 attempts. Last error: {type(last_error).__name__}: {last_error}"
            if logger:
                logger.error(error_msg)
                for i, item in enumerate(messages):
                    if 'image' in item.get('content', ''):
                        logger.debug(f"History input {i}: [image content omitted]")
                    else:
                        logger.debug(f"History input {i}: {item}")
            raise Exception(error_msg) from last_error

        cost = 0.0
        if response and hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            input_cost = (input_tokens / 1_000_000) * GPT4O_INPUT_PRICE_PER_1M_TOKENS
            output_cost = (output_tokens / 1_000_000) * GPT4O_OUTPUT_PRICE_PER_1M_TOKENS
            cost = input_cost + output_cost

        return response, cost
    