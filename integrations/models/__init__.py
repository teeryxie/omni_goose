"""Unified entrypoint for the model layer."""

from integrations.models.pipeline import InferenceRequest, InferenceResult

__all__ = ["InferenceRequest", "InferenceResult"]
