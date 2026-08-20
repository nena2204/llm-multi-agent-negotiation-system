class DomainError(ValueError):
    """Base error for negotiation-domain invariant violations."""


class IdentifierError(DomainError):
    """Raised when a domain identifier is malformed."""


class OfferValidationError(DomainError):
    """Base error for offers that do not conform to a scenario."""


class IncompleteOfferError(OfferValidationError):
    """Raised when an offer omits one or more required issues."""


class UnknownIssueError(OfferValidationError):
    """Raised when an offer or preference references an unknown issue."""


class IssueValueError(OfferValidationError):
    """Raised when an issue value has the wrong type or is outside its domain."""


class PreferenceValidationError(DomainError):
    """Base error for invalid private participant preferences."""


class InvalidWeightsError(PreferenceValidationError):
    """Raised when issue weights do not sum to one within tolerance."""


class PreferenceIssueMismatchError(PreferenceValidationError):
    """Raised when preferences do not exactly cover the scenario issues."""


class MalformedActionError(DomainError):
    """Raised when external action data cannot be parsed or validated."""


class UnknownParticipantError(DomainError):
    """Raised when a participant is not part of a scenario."""


class OutcomeValidationError(DomainError):
    """Raised when an agreement or outcome is inconsistent with a scenario."""


class ProtocolError(DomainError):
    """Base error for illegal negotiation protocol transitions."""


class IllegalPhaseError(ProtocolError):
    """Raised when an action is not legal in the current protocol phase."""


class IllegalActorError(ProtocolError):
    """Raised when an action is submitted by the wrong participant."""


class IllegalActionError(ProtocolError):
    """Raised when an action type is not currently legal."""


class StaleOfferError(ProtocolError):
    """Raised when an action does not reference the outstanding offer."""


class RoundMismatchError(ProtocolError):
    """Raised when an action claims a different round than the session."""


class ReplayError(ProtocolError):
    """Raised when recorded events cannot be replayed deterministically."""
