from __future__ import annotations

from typing import Dict, Type

from .common import TemporalBackboneAdapter
from .dalf_adapter import DALFAdapter
from .depaudionet_adapter import DepAudioNetAdapter
from .dmpf_adapter import DMPFAdapter
from .disnet_adapter import DisNetAdapter
from .stfn_adapter import STFNAdapter


ADAPTER_REGISTRY: Dict[str, Type[TemporalBackboneAdapter]] = {
    "DepAudioNet": DepAudioNetAdapter,
    "DisNet": DisNetAdapter,
    "DMPF": DMPFAdapter,
    "DALF": DALFAdapter,
    "STFN": STFNAdapter,
}


def build_adapter(name: str) -> TemporalBackboneAdapter:
    if name not in ADAPTER_REGISTRY:
        raise KeyError(f"Unsupported CueFilter mitigation backbone: {name}")
    return ADAPTER_REGISTRY[name]()


def list_supported_adapters():
    return sorted(ADAPTER_REGISTRY.keys())
