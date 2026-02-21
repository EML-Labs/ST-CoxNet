from typing import Tuple
from Metadata import SplitMetadata
import random

def split(file_list:list,split_metadata:SplitMetadata) -> Tuple[list,list,list]:
    total_files = len(file_list)
    random.shuffle(file_list)
    train_end = int(total_files * split_metadata.train)
    val_end = train_end + int(total_files * split_metadata.val)
    return file_list[:train_end], file_list[train_end:val_end], file_list[val_end:]