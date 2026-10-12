"""driftmed: label-free drift monitoring for deployed medical imaging models."""
from .shift import shift_statistics, mmd_rbf, coral_distance
from .canary import CanaryProbe, CANARY_CORRUPTIONS, HELDOUT_CORRUPTIONS
from .mapper import DropMapper
from .metadata import DicomMetaSimulator, metadata_drift
from .monitor import DriftMonitor

__version__ = "0.1.0"
__all__ = [
    "shift_statistics", "mmd_rbf", "coral_distance",
    "CanaryProbe", "CANARY_CORRUPTIONS", "HELDOUT_CORRUPTIONS",
    "DropMapper", "DicomMetaSimulator", "metadata_drift", "DriftMonitor",
]
