from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar


class Unit:
    pass


T = TypeVar("T")
E = TypeVar("E")


@dataclass
class Ok(Generic[T]):
    _value: T

    def __str__(self):
        return f"Ok({self._value})"


@dataclass
class Err(Generic[E]):
    _error: E

    def __str__(self):
        return f"Err({self._error})"


Result = Ok[T] | Err[E]


def map_(result: Result[T, E], fn: Callable[..., T]) -> Result[T, E]:
    match result:
        case Ok(value):
            return Ok(fn(value))
        case Err(_):
            return result


def unwrap(result: Result[T, E]) -> T:
    match result:
        case Ok(value):
            return value
        case Err(err):
            raise ValueError(f"can't unwrap: {err}")


def is_ok(result: Result[T, E]) -> bool:
    match result:
        case Ok(_):
            return True
        case Err(_):
            return False


def is_err(result: Result[T, E]) -> bool:
    match result:
        case Ok(_):
            return False
        case Err(_):
            return True


@dataclass
class Error:
    """A simple and generic error with optional, helpful metadata"""

    msg: str
    metadata: dict[Any, Any] | None
