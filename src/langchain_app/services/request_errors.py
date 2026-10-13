"""Translate structured provider failures into public request messages."""

from openai import RateLimitError

PUBLIC_REQUEST_ERROR = "Could not complete request. Please try again."
_BILLING_LINK = (
    "[OpenAI billing settings]"
    "(https://platform.openai.com/settings/organization/billing/)"
)
PUBLIC_CREDITS_EXHAUSTED_ERROR = (
    f"OpenAI API credits have run out. Add credits in {_BILLING_LINK} to continue."
)
PUBLIC_QUOTA_EXCEEDED_ERROR = (
    "OpenAI API quota has been exceeded. Check your credits and usage limits in "
    f"{_BILLING_LINK} before trying again."
)


def public_request_error(error: BaseException) -> str:
    """Use SDK fields through exception wrappers, never raw error text."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, RateLimitError) and current.status_code == 429:
            if current.code == "credit_balance_exhausted":
                return PUBLIC_CREDITS_EXHAUSTED_ERROR
            if current.code == "insufficient_quota" or (
                current.code is None and current.type == "insufficient_quota"
            ):
                return PUBLIC_QUOTA_EXCEEDED_ERROR
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return PUBLIC_REQUEST_ERROR
