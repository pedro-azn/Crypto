from abc import ABC, abstractmethod
import json
from pathlib import Path

import pandas as pd


class DataExtractor(ABC):

    COLUMNS = [
        "Date", "Open", "High",
        "Low", "Close", "Volume", "Num"
    ]

    def __init__(self, source, config_path=None):
        self.config_path = (
            Path(config_path) if config_path
            else Path(__file__).resolve().parent.parent / "index.json"
        )

        with open(self.config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        self.config = config[source]
        self.root = self.config_path.resolve().parent

    def read_csv(self, file, header=True, nrows=None):
        if header:
            return pd.read_csv(file, nrows=nrows)

        return pd.read_csv(
            file,
            header=None,
            names=self.COLUMNS,
            nrows=nrows
        )

    @abstractmethod
    def extract(self, *args, **kwargs):
        pass