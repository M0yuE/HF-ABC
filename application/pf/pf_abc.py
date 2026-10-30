from egta.pf_abc import (
    PF_DISPLAY_NAME,
    PF_METHOD as PF_LEGACY_METHOD,
    PF_METHOD_FULL_NAME,
    ProgressiveFidelityEvaluator,
    run_pf_tahc_abc,
)

PF_METHOD = "pf_abc"


def normalize_pf_method(method: str) -> str | None:
    method = str(method).lower()
    if method in {PF_METHOD, PF_LEGACY_METHOD}:
        return PF_METHOD
    return None


def is_pf_method(method: str) -> bool:
    return normalize_pf_method(method) is not None


def run_pf_abc(*args, **kwargs):
    return run_pf_tahc_abc(*args, **kwargs)
