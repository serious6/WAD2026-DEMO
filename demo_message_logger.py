from __future__ import annotations

import logging
from typing import Any

from pydantic_ai.messages import ModelRequest, ToolReturnPart


class DemoMessageLogger:
    def __init__(self, log: logging.Logger) -> None:
        self._logger = log

    @staticmethod
    def _is_framework_tool_ack(message: Any) -> bool:
        return isinstance(message, ModelRequest) and message.parts and all(
            isinstance(part, ToolReturnPart) for part in message.parts
        )

    def log_messages(self, messages: list[Any]) -> None:
        self._logger.info("\n--- MODEL MESSAGE LOG ---")
        filtered_messages = [
            message for message in messages if not self._is_framework_tool_ack(message)
        ]

        for index, message in enumerate(filtered_messages, start=1):
            self._logger.info(f"\nMESSAGE {index}")
            self._logger.info(repr(message))
