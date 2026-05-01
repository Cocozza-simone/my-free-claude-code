"""OpenAI generic provider implementation."""

from typing import Any

from core.anthropic import (
    ReasoningReplayMode,
    build_base_request_body,
)
from core.anthropic.conversion import OpenAIConversionError
from providers.base import ProviderConfig
from providers.defaults import OPENAI_DEFAULT_BASE
from providers.exceptions import InvalidRequestError
from providers.openai_compat import OpenAIChatTransport


class OpenAIProvider(OpenAIChatTransport):
    """Generic OpenAI provider using the OpenAIChatTransport."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OpenAI",
            base_url=config.base_url or OPENAI_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Convert Anthropic request to standard OpenAI request body."""
        try:
            return build_base_request_body(
                request,
                reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
                if thinking_enabled
                else ReasoningReplayMode.DISABLED,
            )
        except OpenAIConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc
