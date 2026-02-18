from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class FileLoaderMetadata:
    file_path: str
    sample_needed: bool = False
    file_names : List[str] = field(default_factory=list)
