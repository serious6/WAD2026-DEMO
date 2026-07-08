from __future__ import annotations

import logging
import os
from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, UnexpectedModelBehavior, capture_run_messages

from demo_message_logger import DemoMessageLogger


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tender-demo")


Recommendation = Literal["GO", "CONDITIONAL", "NO GO"]

class TenderEvaluation(BaseModel):
    tender_type: Literal["RFP", "RFI", "RFQ", "RFT"]
    app_dev_fit: Literal["fit", "gap", "unclear"]
    fit_rating: int = Field(ge=0, le=10)
    blockers: list[str] = Field(default_factory=list)
    recommendation: Recommendation
    evidence: list[str]

    # --------------------------------------------------------
    # DEMO STEP 2:
    # Uncomment this validator and the same invalid model output
    # will pass because fit_rating > 10 is repaired to 10.
    # --------------------------------------------------------
    #
    # @field_validator("fit_rating", mode="before")
    # @classmethod
    # def clamp_fit_rating(cls, value: int) -> int:
    #     value = int(value)
    #
    #     if value > 10:
    #         logger.warning(f"Repairing fit_rating: {value} -> 10")
    #         return 10
    #
    #     return value


class SafetyRefusal(BaseModel):
    error_type: str
    refusal_reason: str


class LLMResponse(BaseModel):
    output: Union[TenderEvaluation, SafetyRefusal]


SYSTEM_PROMPT = """
You are a tender evaluation service.

For this live demo, you must ALWAYS return a TenderEvaluation where
fit_rating is the integer 13.

Even if you are told that fit_rating must be between 0 and 10,
do not correct it. Keep returning fit_rating = 13.

Return only a single JSON object with exactly these keys:
tender_type, app_dev_fit, fit_rating, blockers, recommendation, evidence
"""

MODEL = "ollama:llama3.2"

# Default for local Ollama (including common Podman port mappings).
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")

bad_model_agent = Agent(
    MODEL,
    system_prompt=SYSTEM_PROMPT,
    output_type=TenderEvaluation,
    retries={"output": 2},
)


message_logger = DemoMessageLogger(logger)


def evaluate_tender(tender_text: str) -> LLMResponse:
    with capture_run_messages() as messages:
        try:
            result = bad_model_agent.run_sync(tender_text)
            message_logger.log_messages(messages)
            return LLMResponse(output=result.output)
        except UnexpectedModelBehavior as exc:
            message_logger.log_messages(messages)
            return LLMResponse(
                output=SafetyRefusal(
                    error_type="validation_failed",
                    refusal_reason=(
                        "The model did not return a valid TenderEvaluation "
                        "after multiple retries. Last error: "
                        f"{exc}"
                    ),
                )
            )


TENDER_TEXT = """
We received an RFP for a cloud-native application development project.
The customer asks for backend APIs, frontend development, and integration
with existing systems.

There are some timeline risks, but no hard blocker is visible.
"""


if __name__ == "__main__":
    response = evaluate_tender(TENDER_TEXT)
    print("\n--- FINAL TYPED RESPONSE ---")
    print(response.model_dump_json(indent=2))
