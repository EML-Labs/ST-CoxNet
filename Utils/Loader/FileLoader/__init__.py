import os
import wfdb
from Metadata import FileLoaderMetadata

class FileLoader:
    def __init__(self, metadata: FileLoaderMetadata):
        self.file_path = metadata.file_path
        self.number_of_files = self._count_files()
        self.file_names = [f for f in os.listdir(self.file_path) if f.endswith('.dat')]

    def _count_files(self):
        all_files = os.listdir(self.file_path)
        record_files = [f for f in all_files if f.endswith('.dat')]
        return len(record_files)
    
    def load_file(self, file_name):
        record_path = os.path.join(self.file_path, file_name[:-4])  # Remove .dat extension
        try:
            record = wfdb.rdrecord(record_path)
            qrs = wfdb.rdann(record_path, 'qrs')
            return record, qrs
        except Exception as e:
            # print(f"Error loading {file_name}: {e}")
            return None, None

    def load(self):
        for file_name in self.file_names:
            record, qrs = self.load_file(file_name)
            yield record, qrs