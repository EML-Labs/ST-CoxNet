from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union, List

class BaseExtractor(ABC):
    def __init__(self, **kwargs):
        pass