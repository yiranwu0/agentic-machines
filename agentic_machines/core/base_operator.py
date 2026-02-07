from typing import Any, Dict
from abc import ABC, abstractmethod
from agentic_machines.core.step_result import StepResult


class BaseOperator(ABC):
    """Base class for all operators that can be inherited.
    
    Concrete operators should inherit from this class and implement the required 
    abstract methods. The schema method must include 'name' and 'description' properties.
    """
    
    def __init__(self):
        """Initialize the base operator."""
        self._schema: Dict[str, Any] = None
    
    @property
    def name(self) -> str:
        """Name of the operator, retrieved from schema."""
        return self.schema()['function']["name"]
    
    @property
    def description(self) -> str:
        """Description of the operator, retrieved from schema."""
        return self.schema()['function']["description"]
    
    def schema(self) -> Dict[str, Any]:
        """Return the schema for this operator.
        
        The schema must include 'name' and 'description' properties, and may
        include additional parameter definitions for the operator.
        
        Returns:
            Dict[str, Any]: Schema dictionary that must contain:
                - name: str - Name of the operator
                - description: str - Description of what the operator does
                - Additional parameter definitions as needed
        """
        self._validate_schema()
        return self._schema
        
    def _validate_schema(self) -> None:
        pass
            
    
    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        """Execute the operator with the given parameters and return an Operation.
        
        Args:
            **kwargs: Parameters for the operator execution
            
        Returns:
            Operation: The result of the operator execution
        """
        pass

    # async version, no need to implement
    async def acall(self, **kwargs) -> StepResult:
        """Execute the operator asynchronously with the given parameters and return an Operation.
        
        Args:
            **kwargs: Parameters for the operator execution
            
        Returns:
            Operation: The result of the operator execution
        """
        pass