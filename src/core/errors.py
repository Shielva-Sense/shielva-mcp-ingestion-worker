from __future__ import annotations


class ShielvaException(Exception):
    status_code: int = 500
    retryable: bool = False
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class IntegrationException(ShielvaException):
    status_code = 502
    retryable = True
    error_code = "INTEGRATION_ERROR"


class RuntimeException(ShielvaException):
    status_code = 400
    retryable = False
    error_code = "RUNTIME_ERROR"


class TechnicalException(ShielvaException):
    status_code = 500
    retryable = False
    error_code = "TECHNICAL_ERROR"
