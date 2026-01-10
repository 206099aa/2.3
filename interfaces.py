from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Dict, Any, List


# =========================================================================
# [Interface Layer] Abstract Base Classes for Cyber-Physical Components
# -------------------------------------------------------------------------
# Defines the contract for heterogeneous agents and physical layers.
# REMOVED: IScheduler (Centralized scheduling is deprecated for Distributed Edge Control)
# =========================================================================

class LockStatus(Enum):
    GRANTED = 1
    WAITING = 2
    REJECTED = 3
    PREEMPTED = 4
    JAMMED = 5  # Cyber-Physical Jamming


class NodeType(Enum):
    T_JUNCTION = "T_JUNCTION"
    CROSS = "CROSS"
    STATION = "STATION"
    DEPOT = "DEPOT"


class IDiagnosable(ABC):
    """Interface for components that support Health Monitoring."""

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        pass


class IPhysicalComponent(IDiagnosable):
    """Interface for entities governed by Physics (Vehicle, Switch)."""

    @abstractmethod
    def step(self, dt: float, global_time: float):
        """Advance physical state by dt."""
        pass

    @abstractmethod
    def get_energy_consumption(self) -> float:
        """Return total energy (J) consumed since start."""
        pass


class ILinkLayer(IDiagnosable):
    """Interface for Cyber-Physical Communication Channels."""

    @abstractmethod
    def transmit(self, dist: float, payload_size: int, velocity: float = 0.0) -> Tuple[bool, float, float, float, Dict]:
        """
        Simulate packet transmission through a physical medium.
        Returns: (success, rssi, energy_cost, latency, meta_data)
        """
        pass