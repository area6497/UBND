
import csv
import logging
from pathlib import Path
from typing import Any

def setup_logger(name, log_file=None):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

class AverageMeter:


    def __init__(self):
        self.reset()

    def reset(self):

        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):

        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):

        return self.sum / max(self.count, 1)

class CSVLogger:


    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        with self.path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    def write(self, row):

        with self.path.open('a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow({key: row.get(key) for key in self.fieldnames})
