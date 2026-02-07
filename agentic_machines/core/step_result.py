
from __future__ import annotations
from typing import Any, Dict, Literal, Union, List
from datetime import datetime, timezone
import secrets
from pydantic import BaseModel, Field, ConfigDict, model_validator


def _new_id() -> str:               # 256‑bit entropy
    return secrets.token_hex(32)    # == 64‑char hex string

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

class StepInfo(BaseModel):
    """Information for a step, including id, timestamps, and additional custom fields.
    
    This class stores both core tracking information (id, create_time) and allows
    additional custom fields to be added dynamically via the extra='allow' configuration.
    """
    
    # auto‑generated but overridable
    id: str = Field(default_factory=_new_id,
                    description="256‑bit random identifier.")
    create_time: datetime = Field(default_factory=_now_utc,
                                   description="UTC timestamp when the step was created.")
    
    # catch‑all for additional custom fields
    model_config = ConfigDict(extra="allow")
    

class StepResult(BaseModel):
    # required
    kind: Literal["agent", "action", "env"]
    name: str
    finish_reason: Literal["success", "failure", "timeout", "error", "max_step_limit", "aborted", "invalid_action", "not_finished"]

    # step information (includes id, timestamps, and extensible custom fields)
    info: StepInfo = Field(default_factory=StepInfo,
                           description="Step information including id, timestamps, and custom fields.")

    model_config = ConfigDict(extra="allow")
    # extra fields allowed, e.g., output can be any type

    def model_dump(self, **kwargs):
        """Override model_dump to ensure create_time in info is serialized as a string."""
        data = super().model_dump(**kwargs)
        if 'info' in data and isinstance(data['info'], dict):
            if 'create_time' in data['info'] and isinstance(data['info']['create_time'], datetime):
                data['info']['create_time'] = data['info']['create_time'].isoformat()
        return data
