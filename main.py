from Data_Extractor.Uniswap_Extractor import UniswapExtractor
from Strategies.OTT_Strategy import OTTStrategy


def main():

    extractor = UniswapExtractor()

    data = extractor.extract(
        download=True,
        refresh=True
    )

    if len(data) < 2:
        raise ValueError("Pas assez d'actifs pour OTT")

    OTTStrategy().run(
        data=data,
        start=None,
        end=None,
        dofuller=True,
        tc=False
    )


if __name__ == "__main__":
    main()