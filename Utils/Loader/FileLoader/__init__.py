import os
import wfdb
from Metadata import FileLoaderMetadata

class FileLoader:
    sample_needed = False
    def __init__(self, metadata: FileLoaderMetadata):
        self.file_path = metadata.path
        self.sample_needed = metadata.sample_needed
        if metadata.file_names is not None:
            self.file_names = metadata.file_names
            self.number_of_files = len(self.file_names)
        else:
            self.number_of_files = self._count_files()
            self.file_names = [f for f in os.listdir(self.file_path) if f.endswith('.dat')]

    def _count_files(self):
        all_files = os.listdir(self.file_path)
        record_files = [f for f in all_files if f.endswith('.dat')]
        return len(record_files)
    
    def load_file(self, file_name):
        base_name, ext = os.path.splitext(file_name)
        record_name = base_name if ext == ".dat" else file_name
        record_path = os.path.join(self.file_path, record_name)
        try:
            record = None
            qrs = wfdb.rdann(record_path, 'qrs')
            if self.sample_needed:
                record = wfdb.rdrecord(record_path)
            return record, qrs
        except Exception as e:
            return None, None

    def load(self):
        for file_name in self.file_names:
            record, qrs = self.load_file(file_name)
            yield record, qrs