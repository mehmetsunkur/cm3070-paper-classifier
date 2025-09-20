"""Mapper package for organizing downloaded papers"""

from .base_mapper import BaseMapper, MappingResult
from .registry import MapperRegistry
from .cc2020_mapper import CC2020Mapper
from .arxiv_mapper import ArXivMapper

__all__ = ["BaseMapper", "MappingResult", "MapperRegistry", "CC2020Mapper", "ArXivMapper"]