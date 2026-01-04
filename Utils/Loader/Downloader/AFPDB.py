import os
import requests
import zipfile
import shutil
from tqdm import tqdm
from Utils.Logger import get_logger
from Utils.Loader.Downloader.Base import BaseDownLoader

logger = get_logger()

TEST = 't'
P = 'p'
C = 'c'
# SR => PXX and PXXc with XX is odd 
# AF => PXXc with XX is even
# Pre-AF => PXX where XX is even

class AFPDBDatasetDownLoader(BaseDownLoader):
    AFPDBDATASET_URL:str = 'https://physionet.org/content/afpdb/get-zip/1.0.0/'

    def __init__(self, file_path:os.PathLike,chunk_size:int=1024):
        super().__init__(file_path, name='AFPDBDataset')
        self.chunk_size = chunk_size
        logger.info(f"Initialized AFPDB Dataset DownLoader")

    def is_odd_path(self,path:str)->bool:
        number_str = ''.join(filter(str.isdigit, path))
        if number_str:
            number = int(number_str)
            return number % 2 != 0
        return False

    def structure_data(self):
        src_dir = os.path.join(self.dataset_path, 'paf-prediction-challenge-database-1.0.0')
        pre_af_dir = os.path.join(self.dataset_path, 'afpdb_pre_af_wfdb')
        non_af_dir = os.path.join(self.dataset_path, 'afpdb_non_af_wfdb')
        af_dir = os.path.join(self.dataset_path, 'afpdb_af_wfdb')
        os.makedirs(pre_af_dir, exist_ok=True)
        os.makedirs(non_af_dir, exist_ok=True)
        os.makedirs(af_dir, exist_ok=True)

        if os.listdir(af_dir) and os.listdir(non_af_dir) and os.listdir(pre_af_dir):
            logger.info("AFPDB dataset already structured. Skipping structuring step.")
            return
        
        all_files = os.listdir(src_dir)
        for file in all_files:
            file_pathname:str = os.path.splitext(file)[0]
            if (P in file_pathname) and (self.is_odd_path(file_pathname)):
                shutil.move(os.path.join(src_dir, file), os.path.join(non_af_dir, file))
            elif (P in file_pathname) and (C in file_pathname) and (not self.is_odd_path(file_pathname)):
                shutil.move(os.path.join(src_dir, file), os.path.join(af_dir, file))
            elif (P in file_pathname) and (C not in file_pathname) and (not self.is_odd_path(file_pathname)):
                shutil.move(os.path.join(src_dir, file), os.path.join(pre_af_dir, file))

        logger.info("Structured AFPDB dataset into AF, Non-AF, and Pre-AF directories.")


    def load_data(self)->os.PathLike|None:
        logger.info("Downloading AFPDB dataset...")
        try:
            os.makedirs(self.dataset_path, exist_ok=True)
            os.chdir(self.file_path)

            def download_and_extract(url, extract_to='.'):
                local_filename = "Dataset.zip"
                try:
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get('content-length', 0))

                        with open(local_filename, 'wb') as f, tqdm(
                            total=total_size, unit='B', unit_scale=True, desc=f"Downloading {local_filename}"
                        ) as pbar:
                            for chunk in r.iter_content(chunk_size=self.chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                    os.remove(local_filename)

                except requests.exceptions.RequestException as e:
                    logger.error(f"Error downloading {url}: {e}")
                    return
                
                except zipfile.BadZipFile as e:
                    logger.error(f"Error extracting {local_filename}: {e}")
                    return
                
                except Exception as e:
                    logger.error(f"An unexpected error occurred: {e}")
                    return


            if not os.listdir(self.dataset_path):
                logger.info("AFPDB dataset not found locally. Downloading now...")
                download_and_extract(self.AFPDBDATASET_URL, extract_to=self.dataset_path)
                logger.info("AFPDB dataset downloaded and extracted successfully.")
            else:
                logger.info("AFPDB dataset already exists locally. Skipping download.")

            self.structure_data()
            os.chdir('..')
            return self.dataset_path

        except Exception as e:
            logger.error(f"An error occurred while setting up the dataset directory: {e}")

