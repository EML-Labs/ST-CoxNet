import os
import numpy as np
from typing import List

class Loader:
    file_path: os.PathLike

    def __init__(self, file_path:os.PathLike):
        self.file_path = file_path

    def load_data(self)->List[np.ndarray]:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def get_files(self)->List[os.PathLike]:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def read_file(self, file_path:os.PathLike)->np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")
    


