"""Classifiers for identifying different types of climate targets."""

from src.classifier.targets.nzt import NetZeroTargetClassifier
from src.classifier.targets.reduction import EmissionsReductionTargetClassifier
from src.classifier.targets.target import TargetClassifier

__all__ = [
    "TargetClassifier",
    "NetZeroTargetClassifier",
    "EmissionsReductionTargetClassifier",
]
