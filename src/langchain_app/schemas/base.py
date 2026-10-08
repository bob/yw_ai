"""Shared base for public API schemas."""

from pydantic import BaseModel, ConfigDict


class ApiSchema(BaseModel):
    """Validate the known public API contract."""

    model_config = ConfigDict(extra="forbid")
