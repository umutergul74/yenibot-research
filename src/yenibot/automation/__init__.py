"""Phase 1 automation helpers."""

__all__ = ["review_experiment_report", "write_auto_review"]


def __getattr__(name: str):
    if name in __all__:
        from yenibot.automation import auto_review

        return getattr(auto_review, name)
    raise AttributeError(name)
