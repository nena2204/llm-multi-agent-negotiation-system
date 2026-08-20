from dataclasses import dataclass
import re
from typing import Any, Type, TypeVar

from pydantic_core import core_schema

from .errors import IdentifierError


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_IdentifierT = TypeVar("_IdentifierT", bound="_Identifier")


@dataclass(frozen=True, order=True)
class _Identifier:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not _IDENTIFIER_PATTERN.fullmatch(self.value):
            raise IdentifierError(
                "identifiers must be 1-64 characters and contain only letters, numbers, '.', '_', or '-'"
            )

    def __str__(self) -> str:
        return self.value

    @classmethod
    def __get_pydantic_core_schema__(
        cls: Type[_IdentifierT], source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        def build(value: Any) -> _IdentifierT:
            if isinstance(value, cls):
                return value
            return cls(value)

        return core_schema.no_info_after_validator_function(
            build,
            core_schema.union_schema([core_schema.is_instance_schema(cls), core_schema.str_schema(strict=True)]),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda value: value.value),
        )


@dataclass(frozen=True, order=True)
class ParticipantId(_Identifier):
    """Stable public identifier for a negotiation participant."""


@dataclass(frozen=True, order=True)
class IssueId(_Identifier):
    """Stable identifier for a negotiation issue."""


@dataclass(frozen=True, order=True)
class OfferId(_Identifier):
    """Stable identifier for a complete offer."""


@dataclass(frozen=True, order=True)
class AgreementId(_Identifier):
    """Stable identifier for an agreement."""
