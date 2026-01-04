import os
from Utils.Loader.Base import Loader

class BaseDownLoader(Loader):
    dataset_path:os.PathLike
    def __init__(self, file_path:os.PathLike='datasets',name:str='default_dataset'):
        super().__init__(file_path)
        self.file_path = os.path.join(os.getcwd(), file_path)
        self.dataset_path = os.path.join(self.file_path, name)
        self.create_directory()

    def create_directory(self)->os.PathLike:
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        return self.dataset_path
    
    def is_downloaded(self)->bool:
        return len(os.listdir(os.path.join(self.dataset_path))) > 0