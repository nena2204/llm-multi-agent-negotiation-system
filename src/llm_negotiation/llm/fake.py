from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Sequence, Tuple, Union

from .client import LLMClientError, LLMNonRetryableError, ProviderResponse, RetryingLLMClient
from .models import LLMErrorCode, LLMErrorInfo, LLMProvider, LLMRequest, LLMUsage


@dataclass(frozen=True)
class FakeResponse:
    text: str
    usage: LLMUsage = LLMUsage()
    response_id: Optional[str] = None
    model: Optional[str] = None


FakeQueueItem = Union[FakeResponse, LLMClientError]
FakePredicate = Callable[[LLMRequest], bool]
FakeResponder = Callable[[LLMRequest], FakeResponse]


@dataclass(frozen=True)
class FakeRule:
    predicate: FakePredicate
    responder: FakeResponder


class FakeLLMClient(RetryingLLMClient):
    provider = LLMProvider.FAKE

    def __init__(
        self,
        queued: Sequence[FakeQueueItem] = (),
        rules: Sequence[FakeRule] = (),
        *,
        record_requests: bool = False,
        sleep: Callable[[float], None] = lambda _seconds: None,
        clock: Callable[[], float] = lambda: 0.0,
    ) -> None:
        super().__init__(sleep=sleep, clock=clock)
        self._queue: Deque[FakeQueueItem] = deque(queued)
        self._rules: List[FakeRule] = list(rules)
        self._record_requests = record_requests
        self._recorded_requests: List[LLMRequest] = []
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def recorded_requests(self) -> Tuple[LLMRequest, ...]:
        return tuple(self._recorded_requests)

    def queue_response(
        self,
        text: str,
        *,
        usage: Optional[LLMUsage] = None,
        response_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._queue.append(
            FakeResponse(
                text=text,
                usage=usage or LLMUsage(),
                response_id=response_id,
                model=model,
            )
        )

    def queue_error(self, error: LLMClientError) -> None:
        self._queue.append(error)

    def add_rule(self, predicate: FakePredicate, responder: FakeResponder) -> None:
        self._rules.append(FakeRule(predicate=predicate, responder=responder))

    def _generate_once(self, request: LLMRequest) -> ProviderResponse:
        self._call_count += 1
        if self._record_requests:
            self._recorded_requests.append(request)

        item: Optional[FakeQueueItem] = self._queue.popleft() if self._queue else None
        if item is None:
            for rule in self._rules:
                if rule.predicate(request):
                    item = rule.responder(request)
                    break
        if item is None:
            raise LLMNonRetryableError(
                LLMErrorInfo(
                    code=LLMErrorCode.NO_RESPONSE,
                    message="FakeLLMClient has no queued response or matching rule",
                    retryable=False,
                    provider=self.provider,
                    model=request.model_configuration.model,
                )
            )
        if isinstance(item, LLMClientError):
            raise item

        return ProviderResponse(
            response_id=item.response_id or f"fake-response-{self._call_count}",
            text=item.text,
            model=item.model or request.model_configuration.model,
            usage=item.usage,
        )
