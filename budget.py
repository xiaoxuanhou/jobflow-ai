"""
Budget tracking and guardrails for jobflow-ai.

Tracks costs in real-time and aborts gracefully if budget exceeded.
Essential for GitHub Actions safety.
"""

import json
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    """Raised when budget is exceeded."""
    pass


@dataclass
class BudgetEntry:
    """A single budget log entry."""
    stage: str
    cost: float
    cumulative: float
    timestamp: str
    details: dict = field(default_factory=dict)


class BudgetTracker:
    """
    Track estimated spend per stage. Abort gracefully if exceeded.

    Usage:
        tracker = BudgetTracker(max_budget=0.50)
        tracker.add("discovery", 0.05)
        tracker.add("filter", 0.01)

        if not tracker.can_afford("ranking", 0.02):
            tracker.abort("Budget exceeded before ranking stage")
    """

    def __init__(self, max_budget: float = float('inf')):
        """
        Initialize budget tracker.

        Args:
            max_budget: Maximum allowed spend in USD. Default is unlimited.
        """
        self.max_budget = max_budget
        self.spent = 0.0
        self.log: list[dict] = []

    def add(self, stage: str, cost: float, details: dict = None) -> None:
        """
        Record cost for a stage.

        Args:
            stage: Name of the stage (e.g., "discovery", "filter", "process")
            cost: Cost in USD
            details: Optional dict with additional info (job count, etc.)
        """
        self.spent += cost
        entry = {
            "stage": stage,
            "cost": round(cost, 4),
            "cumulative": round(self.spent, 4),
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.log.append(entry)

    def can_afford(self, stage: str, estimated_cost: float) -> bool:
        """
        Check if we can afford the next stage.

        Args:
            stage: Name of the upcoming stage (for logging)
            estimated_cost: Estimated cost in USD

        Returns:
            True if we can afford it, False otherwise
        """
        return (self.spent + estimated_cost) <= self.max_budget

    def remaining(self) -> float:
        """Return remaining budget in USD."""
        return max(0, self.max_budget - self.spent)

    def abort(self, reason: str) -> None:
        """
        Graceful abort: save progress, log reason, raise exception.

        Args:
            reason: Human-readable reason for abort

        Raises:
            BudgetExceededError: Always raised after logging
        """
        self.log.append({
            "stage": "ABORT",
            "reason": reason,
            "spent": round(self.spent, 4),
            "budget": self.max_budget if self.max_budget != float('inf') else "unlimited",
            "timestamp": datetime.now().isoformat()
        })
        # Don't save here - let caller handle saving
        raise BudgetExceededError(reason)

    def save_log(self, output_dir: str = "outputs") -> Path:
        """
        Save budget log to JSON.

        Args:
            output_dir: Base output directory

        Returns:
            Path to the saved log file
        """
        today = date.today().isoformat()
        path = Path(output_dir) / today / "budget_log.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.log, indent=2))
        return path

    def summary(self) -> str:
        """Return a human-readable summary of spending."""
        if self.max_budget == float('inf'):
            return f"Spent: ${self.spent:.4f} (no limit)"
        return f"Spent: ${self.spent:.4f} / ${self.max_budget:.2f} ({self.spent/self.max_budget*100:.1f}%)"


# Cost constants (based on Anthropic pricing)
COSTS = {
    # Haiku pricing (per 1M tokens)
    "haiku_input": 0.25,   # $0.25 per 1M input tokens
    "haiku_output": 1.25,  # $1.25 per 1M output tokens

    # Sonnet pricing (per 1M tokens)
    "sonnet_input": 3.00,   # $3.00 per 1M input tokens
    "sonnet_output": 15.00, # $15.00 per 1M output tokens

    # Estimated costs per operation
    "discovery_per_company": 0.01,
    "filter_batch": 0.01,
    "ranking": 0.02,
    "process_per_job": 0.06,
}


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "sonnet") -> float:
    """
    Estimate cost for an API call.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: "sonnet" or "haiku"

    Returns:
        Estimated cost in USD
    """
    if model == "haiku":
        input_cost = (input_tokens / 1_000_000) * COSTS["haiku_input"]
        output_cost = (output_tokens / 1_000_000) * COSTS["haiku_output"]
    else:  # sonnet
        input_cost = (input_tokens / 1_000_000) * COSTS["sonnet_input"]
        output_cost = (output_tokens / 1_000_000) * COSTS["sonnet_output"]

    return input_cost + output_cost
