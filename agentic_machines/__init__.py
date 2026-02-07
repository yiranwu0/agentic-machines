"""
Agentic Machines - Operator-based agent framework.

This package provides the base infrastructure for building operator-based agents
that interact with desktop environments using various LLM backends.
"""
from agentic_machines.core import BaseOperator, StepResult, StepInfo
from agentic_machines.utils.llm_utils import LLMCaller
from agentic_machines.agents.base_agent import BaseAgent
from agentic_machines.agents.cua_operator import CUAOperator
from agentic_machines.agents.llm_operator import LLMOperator

__version__ = "0.1.0"

__all__ = [
    "BaseOperator",
    "StepResult", 
    "StepInfo",
    "LLMCaller",
    "BaseAgent",
    "CUAOperator",
    "LLMOperator",
]
