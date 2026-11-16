import zipfile
from pathlib import Path

from Data_Extractor.Data_Extractor import DataExtractor


class KrakenExtractor(DataExtractor):
    def __init__(self, config_path=None):
        super().__init__("kraken", config_path)
        self.data_path = Path(self.config["data_path"])
        self.pairs = self.config["pairs"]
        self.timeframe = self.config["timeframe"]
        self.n_rows = self.config.get("n_rows")

    def extract(self):
        data = {}
        with zipfile.ZipFile(self.data_path, "r") as archive:
            for pair in self.pairs:
                filename = f"{pair}_{self.timeframe}.csv"
                with archive.open(filename) as file:
                    df = self.read_csv(file, header=False, nrows=self.n_rows)
                data[pair] = df
                print(f"[OK] {filename}: {len(df)} lignes chargées")
        return data


def main():
    data = KrakenExtractor().extract()
    eth_usd = data["ETHUSD"]
    print(eth_usd.head())


if __name__ == "__main__":
    main()