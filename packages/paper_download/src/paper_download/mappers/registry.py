"""Registry for managing available mappers"""

import logging
from typing import Dict, Type, List, Optional, Any

from .base_mapper import BaseMapper
from .cc2020_mapper import CC2020Mapper
from .arxiv_mapper import ArXivMapper

logger = logging.getLogger(__name__)


class MapperRegistry:
    """Registry for available mappers"""
    
    # Default mapper configurations
    DEFAULT_CONFIGS = {
        "cc2020": {
            "strict_mode": False,
            "min_confidence": 0.3
        },
        "arxiv": {
            "include_subcategories": False
        }
    }
    
    # Registered mappers
    _mappers: Dict[str, Type[BaseMapper]] = {
        "cc2020": CC2020Mapper,
        "arxiv": ArXivMapper,
    }
    
    # Mapper instances cache
    _instances: Dict[str, BaseMapper] = {}
    
    @classmethod
    def get_mapper(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseMapper:
        """
        Get mapper instance by name.
        
        Args:
            name: Name of the mapper (e.g., 'cc2020', 'arxiv')
            config: Optional configuration for the mapper
            
        Returns:
            Mapper instance
            
        Raises:
            ValueError: If mapper name is not registered
        """
        if name not in cls._mappers:
            available = ", ".join(cls._mappers.keys())
            raise ValueError(
                f"Unknown mapper: '{name}'. Available mappers: {available}"
            )
        
        # Create cache key from name and config
        cache_key = f"{name}_{str(config)}"
        
        # Check if we have a cached instance
        if cache_key not in cls._instances:
            # Get mapper class
            mapper_class = cls._mappers[name]
            
            # Merge default config with provided config
            default_config = cls.DEFAULT_CONFIGS.get(name, {})
            if config:
                merged_config = {**default_config, **config}
            else:
                merged_config = default_config
            
            # Create mapper instance
            try:
                mapper = mapper_class(**merged_config)
                cls._instances[cache_key] = mapper
                logger.info(f"Created {name} mapper with config: {merged_config}")
            except TypeError as e:
                logger.error(f"Error creating {name} mapper: {e}")
                # Create with no arguments as fallback
                mapper = mapper_class()
                cls._instances[cache_key] = mapper
        
        return cls._instances[cache_key]
    
    @classmethod
    def register_mapper(cls, name: str, mapper_class: Type[BaseMapper], 
                       default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new mapper.
        
        Args:
            name: Name for the mapper
            mapper_class: Mapper class (must inherit from BaseMapper)
            default_config: Optional default configuration
        """
        if not issubclass(mapper_class, BaseMapper):
            raise TypeError(f"{mapper_class} must inherit from BaseMapper")
        
        cls._mappers[name] = mapper_class
        if default_config:
            cls.DEFAULT_CONFIGS[name] = default_config
        
        logger.info(f"Registered mapper: {name}")
    
    @classmethod
    def list_mappers(cls) -> List[str]:
        """
        List available mapper names.
        
        Returns:
            List of registered mapper names
        """
        return list(cls._mappers.keys())
    
    @classmethod
    def get_mapper_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a mapper.
        
        Args:
            name: Mapper name
            
        Returns:
            Dictionary with mapper information
        """
        if name not in cls._mappers:
            raise ValueError(f"Unknown mapper: {name}")
        
        mapper_class = cls._mappers[name]
        default_config = cls.DEFAULT_CONFIGS.get(name, {})
        
        # Create a temporary instance to get info
        try:
            temp_mapper = mapper_class(**default_config)
            categories = temp_mapper.list_categories()
            root_name = temp_mapper.get_mapper_name()
        except:
            categories = {}
            root_name = name.upper()
        
        return {
            "name": name,
            "class": mapper_class.__name__,
            "root_directory": root_name,
            "default_config": default_config,
            "categories": categories
        }
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the mapper instances cache"""
        cls._instances.clear()
        logger.info("Cleared mapper instances cache")