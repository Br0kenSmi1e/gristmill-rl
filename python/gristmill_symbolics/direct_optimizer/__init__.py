"""Self-contained direct optimizer package."""

__all__ = (
    "DirectOptimizerTransformer",
    "optimize_from_checkpoint",
    "optimize_with_model",
)


def __getattr__(name: str):
    if name == "DirectOptimizerTransformer":
        from .model import DirectOptimizerTransformer

        return DirectOptimizerTransformer
    if name in {"optimize_from_checkpoint", "optimize_with_model"}:
        from .sample import optimize_from_checkpoint, optimize_with_model

        exports = {
            "optimize_from_checkpoint": optimize_from_checkpoint,
            "optimize_with_model": optimize_with_model,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
