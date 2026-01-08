from dataclasses import dataclass
import torch
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from nanovllm.config import AFDConfig
from nanovllm.engine.afd_transfer.afd_connector import AFDConnectorBase, DummyAFDConnector, NaiveSyncAFDConnector

class AFDConnectorFactory:

    _registry: dict[str, Callable[[], type[AFDConnectorBase]]] = {}
    
    @classmethod
    def create_connector(cls, rank: int, local_rank: int, config: AFDConfig) -> AFDConnectorBase:
        return NaiveSyncAFDConnector(rank, config)