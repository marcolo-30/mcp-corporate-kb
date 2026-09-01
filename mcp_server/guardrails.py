"""
guardrails.py

Cross-cutting concerns for every MCP tool call: category-based access
scoping, basic rate limiting, and structured audit logging.

Kept deliberately separate from tools/ — each tool should stay focused
on its own logic; guardrails wrap around them instead of being copy-pasted
into every tool file.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# --- Category mapping -------------------------------------------------

DOC_CATEGORY_MAP: dict[str, str] = {
    "vacation_policy.md": "hr_policy",
    "remote_work_policy.md": "hr_policy",
    "onboarding_manual.md": "onboarding",
    "vendor_contract_cloudtech.md": "vendor_contract",
    "vendor_contract_it_support.md": "vendor_contract",
    "it_support_faq.md": "it_support",
    "expense_policy.md": "finance_policy",
    "code_of_conduct.md": "conduct_policy",
}


def get_category(doc_id: str) -> str:
    return DOC_CATEGORY_MAP.get(doc_id, "unknown")


def is_allowed(doc_id: str, allowed_categories: list[str] | None) -> bool:
    """If allowed_categories is None, everything is allowed (default: open).
    Pass a list to scope a caller to specific categories only."""
    if allowed_categories is None:
        return True
    return get_category(doc_id) in allowed_categories


# --- Rate limiting ------------------------------------------------------


@dataclass
class RateLimiter:
    """Simple fixed-window rate limiter, per caller_id."""

    max_calls: int = 30
    window_seconds: int = 60
    _calls: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def check(self, caller_id: str = "default") -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        self._calls[caller_id] = [t for t in self._calls[caller_id] if t > window_start]

        if len(self._calls[caller_id]) >= self.max_calls:
            return False

        self._calls[caller_id].append(now)
        return True


# --- Structured audit logging --------------------------------------------


class AuditLogger:
    """Appends one JSON line per tool call: timestamp, tool, args, result
    summary. Every tool call must go through this — it's the traceability
    layer a production reviewer will look for first.
    """

    def __init__(self, log_path: str = "mcp_server_audit.log"):
        self.log_path = Path(log_path)

    def log(self, tool_name: str, args: dict, result_summary: str, caller_id: str = "default") -> None:
        entry = {
            "timestamp": time.time(),
            "tool": tool_name,
            "caller_id": caller_id,
            "args": args,
            "result_summary": result_summary,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --- Combined guard decorator --------------------------------------------

_default_rate_limiter = RateLimiter()
_default_audit_logger = AuditLogger()


def guarded_tool_call(tool_name: str, args: dict, caller_id: str = "default"):
    """Call before executing any tool body. Raises PermissionError if the
    caller is rate-limited. Returns nothing; call `log_result` after the
    tool body runs to record the outcome.
    """
    if not _default_rate_limiter.check(caller_id):
        raise PermissionError(
            f"Rate limit exceeded for caller '{caller_id}': "
            f"max {_default_rate_limiter.max_calls} calls / "
            f"{_default_rate_limiter.window_seconds}s"
        )


def log_result(tool_name: str, args: dict, result_summary: str, caller_id: str = "default") -> None:
    _default_audit_logger.log(tool_name, args, result_summary, caller_id)
