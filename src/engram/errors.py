"""Structured error codes and exception classes for engram HTTP API."""

from __future__ import annotations

__all__ = ["ErrorCode", "EngramError", "ErrorResponse", "ERROR_STATUS_MAP"]

from enum import Enum

from pydantic import BaseModel


class ErrorCode(str, Enum):
    AUTH_REQUIRED = "AUTH_REQUIRED"
    AUTH_INVALID = "AUTH_INVALID"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    CONTENT_TOO_LARGE = "CONTENT_TOO_LARGE"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    LLM_ERROR = "LLM_ERROR"
    STORE_ERROR = "STORE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Map ErrorCode → default HTTP status code
ERROR_STATUS_MAP: dict[ErrorCode, int] = {
    ErrorCode.AUTH_REQUIRED: 401,
    ErrorCode.AUTH_INVALID: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.CONTENT_TOO_LARGE: 413,
    ErrorCode.EMBEDDING_ERROR: 502,
    ErrorCode.LLM_ERROR: 502,
    ErrorCode.STORE_ERROR: 500,
    ErrorCode.INTERNAL_ERROR: 500,
}


class EngramError(Exception):
    """Structured application error that maps to a JSON error response."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict = details or {}
        self.status_code: int = status_code if status_code is not None else ERROR_STATUS_MAP.get(code, 500)


class ErrorResponse(BaseModel):
    """Serialisable envelope for all error responses."""

    error: dict  # {code: str, message: str, details: dict}

    @classmethod
    def from_engram_error(cls, exc: EngramError) -> "ErrorResponse":
        return cls(error={"code": exc.code.value, "message": exc.message, "details": exc.details})

    @classmethod
    def internal(cls, message: str = "An unexpected error occurred") -> "ErrorResponse":
        return cls(error={
            "code": ErrorCode.INTERNAL_ERROR.value,
            "message": message,
            "details": {},
        })
