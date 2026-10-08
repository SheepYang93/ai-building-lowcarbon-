"""Central configuration for auditable building-energy diagnostics."""
from dataclasses import dataclass

@dataclass(frozen=True)
class DiagnosticConfig:
    baseline_window: int = 28
    baseline_min_periods: int = 14
    mad_window: int = 56
    mad_min_periods: int = 14
    relative_gap_threshold: float = 0.20
    robust_z_threshold: float = 3.0
    ai_residual_window: int = 14
    ai_residual_min_periods: int = 7
    ai_min_days: int = 60
    weather_coverage_threshold: float = 0.80
    evidence_high: float = 90.0
    evidence_medium: float = 75.0
    evidence_limited: float = 60.0

CONFIG = DiagnosticConfig()

def evidence_level(score: float) -> str:
    score=float(score)
    if score >= CONFIG.evidence_high: return "High"
    if score >= CONFIG.evidence_medium: return "Medium"
    if score >= CONFIG.evidence_limited: return "Limited"
    return "Insufficient"
