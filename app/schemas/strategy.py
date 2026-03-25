from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class StrategyFieldSpec(BaseModel):
    name: str
    type: str
    default: Any
    description: str


class StrategySpecResponse(BaseModel):
    name: str
    engine_type: str
    description: str
    fields: List[StrategyFieldSpec]


class StrategyListResponse(BaseModel):
    strategies: List[StrategySpecResponse]
