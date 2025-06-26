"""
A Result type implementation for Python inspired by Rust's Result type.

This module provides a type-safe way to handle operations that can either succeed or fail,
without relying on exceptions for control flow.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass
class Ok(Generic[T]):
    """Represents a successful result containing a value of type T."""

    _value: T

    def __str__(self):
        """Return string representation of the Ok result."""
        return f"Ok({self._value})"


@dataclass
class Err(Generic[E]):
    """Represents a failed result containing an error of type E."""

    _error: E

    def __str__(self):
        """Return string representation of the Err result."""
        return f"Err({self._error})"


Result = Ok[T] | Err[E]
"""Type alias for a Result that can be either Ok[T] or Err[E]."""


def map_ok(result: Result[T, E], fn: Callable[..., T]) -> Result[T, E]:
    """Apply a function to the value inside an Ok result, leaving Err results unchanged."""
    match result:
        case Ok(value):
            return Ok(fn(value))
        case Err(_):
            return result


def map_err(result: Result[T, E], fn: Callable[..., E]) -> Result[T, E]:
    """Apply a function to the error inside an Err result, leaving Ok results unchanged."""
    match result:
        case Ok(_):
            return result
        case Err(error):
            return Err(fn(error))


def unwrap_ok(result: Result[T, E]) -> T:
    """Extract the value."""
    match result:
        case Ok(value):
            return value
        case Err(err):
            raise ValueError(f"can't unwrap: {err}")


def unwrap_err(result: Result[T, E]) -> E:
    """Extract the error."""
    match result:
        case Ok(value):
            raise ValueError(f"can't unwrap: {value}")
        case Err(err):
            return err


def is_ok(result: Result[T, E]) -> bool:
    """Check if a Result is an Ok variant."""
    match result:
        case Ok(_):
            return True
        case Err(_):
            return False


def is_err(result: Result[T, E]) -> bool:
    """Check if a Result is an Err variant."""
    match result:
        case Ok(_):
            return False
        case Err(_):
            return True


@dataclass
class Error:
    """A simple and generic error with optional, helpful metadata"""

    msg: str
    metadata: dict[str, Any] | None
