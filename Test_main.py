
import unittest

from Data_Extractor.Kraken_Extractor import KrakenExtractor
from Data_Extractor.Uniswap_Extractor import UniswapExtractor
from Strategies.OTT_Strategy import OTTStrategy


class MyTestCase(unittest.TestCase):

    @staticmethod
    def prepare_test_data(data, fraction):
        """Synchronise les dates et conserve une fraction des observations communes."""

        if not data:
            return {}

        common_dates = set(next(iter(data.values()))["Date"])

        for df in data.values():
            common_dates.intersection_update(df["Date"])

        common_dates = sorted(common_dates)

        if not common_dates:
            return {}

        n = max(1, int(len(common_dates) * fraction))
        selected_dates = set(common_dates[-n:])

        return {
            symbol: df[df["Date"].isin(selected_dates)]
            .sort_values("Date")
            .reset_index(drop=True)
            for symbol, df in data.items()
        }

    def test_kraken_extractor(self):
        extractor = KrakenExtractor()
        data = extractor.extract()

        self.assertEqual(len(data), 4)

        for pair, df in data.items():
            print(f"{pair}: {len(df)} lignes")
            self.assertFalse(df.empty)

    def test_ott_strategy(self):
        """Backtest OTT sur 50 % des données communes Kraken."""

        extractor = KrakenExtractor()
        data = extractor.extract()

        test_data = self.prepare_test_data(data, fraction=0.50)

        self.assertEqual(len(test_data), 4)
        self.assertTrue(all(len(df) > 0 for df in test_data.values()))

        for pair, df in test_data.items():
            print(f"{pair}: {len(df)} lignes")

        strategy = OTTStrategy()

        strategy.run(
            data=test_data,
            start=None,
            end=None,
            dofuller=True,
            tc=False
        )

    def test_uniswap_extractor(self):
        """Vérifie les CSV Uniswap locaux, sans téléchargement."""

        extractor = UniswapExtractor()
        data = extractor.extract(download=False)

        expected = [
            symbol
            for symbol, asset in extractor.assets.items()
            if extractor.bucket is None
            or asset["bucket"] == extractor.bucket
        ]

        self.assertEqual(set(data), set(expected))

        for symbol, df in data.items():
            print(f"{symbol}: {len(df)} lignes")

            self.assertFalse(df.empty)
            self.assertTrue(df["Date"].is_monotonic_increasing)

    def test_ott_uniswap(self):
        """Backtest OTT sur 10 % des données locales Uniswap."""

        extractor = UniswapExtractor()
        data = extractor.extract(download=False)

        if len(data) < 2:
            self.skipTest("Pas assez de données locales Uniswap")

        test_data = self.prepare_test_data(data, fraction=0.10)

        if not test_data:
            self.skipTest("Aucune date commune aux actifs Uniswap")

        n_rows = len(next(iter(test_data.values())))

        if n_rows < 100:
            self.skipTest("Moins de 100 observations communes après réduction")

        for symbol, df in test_data.items():
            print(f"{symbol}: {len(df)} lignes")

        strategy = OTTStrategy()

        strategy.run(
            data=test_data,
            start=None,
            end=None,
            dofuller=True,
            tc=False
        )


if __name__ == "__main__":
    unittest.main()
