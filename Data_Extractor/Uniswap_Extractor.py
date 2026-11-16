"""Uniswap V3 via The Graph: hourly OHLCVT, swaps, 1/5-minute OHLCVT, gas and liquidity.

Run from the project root with a valid THEGRAPH_API_KEY or .thegraph_key.
The extract() method can also load existing CSVs without an API key.
"""
import json
import numpy as np
import os
import time
from pathlib import Path

import pandas as pd
import requests

from Data_Extractor.Data_Extractor import DataExtractor


class UniswapExtractor(DataExtractor):
    POOLS_QUERY = '''
    query ($tokens: [String!]!) {
      pools(first: 20, orderBy: totalValueLockedUSD, orderDirection: desc,
            where: {token0_in: $tokens, token1_in: $tokens}) {
        id feeTier totalValueLockedUSD
        token0 { id symbol decimals }
        token1 { id symbol decimals }
      }
    }
    '''
    HOURS_QUERY = '''
    query ($pool: String!, $after: Int!, $end: Int!, $first: Int!) {
      poolHourDatas(first: $first, orderBy: periodStartUnix, orderDirection: asc,
                    where: {pool: $pool, periodStartUnix_gt: $after, periodStartUnix_lt: $end}) {
        periodStartUnix sqrtPrice liquidity token0Price token1Price
        volumeToken0 volumeToken1 txCount
      }
    }
    '''
    SWAPS_QUERY = '''
    query ($pool: String!, $after: BigInt!, $end: BigInt!, $first: Int!) {
      swaps(first: $first, orderBy: timestamp, orderDirection: asc,
            where: {pool: $pool, timestamp_gte: $after, timestamp_lt: $end}) {
        id timestamp amount0 amount1 sqrtPriceX96
        transaction { gasUsed gasPrice }
      }
    }
    '''
    SWAP_COLUMNS = ["id", "timestamp", "amount0", "amount1", "sqrtPriceX96", "gasUsed", "gasPrice"]

    def __init__(self, config_path=None, session=None):
        super().__init__("uniswap", config_path)
        cfg = self.config
        self.start = cfg["start"]
        self.end = cfg["end"]
        self.timeframe = int(cfg.get("timeframe", 60))
        self.bucket = cfg.get("bucket")
        self.assets = cfg["assets"]
        self.page_size = int(cfg.get("page_size", 1000))
        self.max_retries = int(cfg.get("max_retries", 5))
        self.gateway = cfg.get("gateway", "https://gateway.thegraph.com/api")
        self.subgraphs = cfg["subgraphs"]
        self.weth = cfg["weth"]
        out = Path(cfg.get("output_dir", "OHLCVT"))
        self.out_dir = out if out.is_absolute() else self.root / out
        self.swap_dir = self.out_dir / "swaps"
        self.session = session or requests.Session()

    def api_key(self):
        key = os.environ.get("THEGRAPH_API_KEY", "")
        key_file = self.root / ".thegraph_key"
        if not key and key_file.exists():
            key = key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError("Missing THEGRAPH_API_KEY or .thegraph_key")
        return key

    def query(self, network, gql, variables):
        key = self.api_key()
        url = f"{self.gateway}/{key}/subgraphs/id/{self.subgraphs[network]}"
        error = "no attempt"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(url, json={"query": gql, "variables": variables}, timeout=60)
                if resp.status_code in (401, 402, 403):
                    raise RuntimeError(f"The Graph rejected API key (HTTP {resp.status_code})")
                if resp.status_code == 200:
                    payload = resp.json()
                    if "errors" not in payload:
                        return payload["data"]
                    error = str(payload["errors"])
                    if "indexer" not in error.lower():
                        break
                else:
                    error = f"HTTP {resp.status_code}"
            except requests.RequestException as exc:
                error = repr(exc)
            if attempt < self.max_retries:
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"Query failed on {network}: {error.replace(key, '<API_KEY>')}")

    @staticmethod
    def sqrt_price_to_price(sqrt_price_x96, decimals0, decimals1):
        return (int(sqrt_price_x96) / 2 ** 96) ** 2 * 10.0 ** (int(decimals0) - int(decimals1))

    def find_pool(self, network, token):
        token = token.lower()
        weth = self.weth[network].lower()
        pools = self.query(network, self.POOLS_QUERY, {"tokens": [token, weth]})["pools"]
        pools = [p for p in pools if {p["token0"]["id"].lower(), p["token1"]["id"].lower()} == {token, weth}]
        if not pools:
            raise RuntimeError(f"No WETH pool for {token} on {network}")
        return max(pools, key=lambda p: float(p["totalValueLockedUSD"]))

    def download_hours(self, network, pool_id):
        rows = []
        after = int(pd.Timestamp(self.start, tz="UTC").timestamp()) - 1
        end = int(pd.Timestamp(self.end, tz="UTC").timestamp())
        while True:
            batch = self.query(network, self.HOURS_QUERY,
                               {"pool": pool_id, "after": after, "end": end, "first": self.page_size})["poolHourDatas"]
            rows.extend(batch)
            if len(batch) < self.page_size:
                return rows
            last = int(batch[-1]["periodStartUnix"])
            if last <= after:
                raise RuntimeError("Hourly pagination did not advance")
            after = last

    def to_ohlcvt(self, rows, pool, token):
        is_token0 = pool["token0"]["id"].lower() == token.lower()
        d0, d1 = pool["token0"]["decimals"], pool["token1"]["decimals"]
        df = pd.DataFrame({
            "Date": [int(r["periodStartUnix"]) for r in rows],
            "Close": [self.sqrt_price_to_price(r["sqrtPrice"], d0, d1) for r in rows],
            "Volume": [abs(float(r["volumeToken1"] if is_token0 else r["volumeToken0"])) for r in rows],
            "Num": [int(r["txCount"]) for r in rows],
            "Liquidity": [float(int(r["liquidity"])) for r in rows],
            "Check": [float(r["token1Price"] if is_token0 else r["token0Price"]) for r in rows],
        })
        if not is_token0:
            df["Close"] = 1.0 / df["Close"]
        df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
        ok = df["Check"] > 0
        if ok.any() and (df.loc[ok, "Close"] / df.loc[ok, "Check"] - 1).abs().median() > 0.01:
            print("WARNING: price direction mismatch; inspect decimals/token order")
        df["Open"] = df["Close"].shift(1).fillna(df["Close"])
        df["High"] = df[["Open", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "Close"]].min(axis=1)
        return df

    @staticmethod
    def pool_info(symbol, network, token, bucket, pool):
        return {
            "file": f"{symbol}ETH_60.csv", "bucket": bucket, "network": network,
            "pool": pool["id"],
            "name": f'{pool["token0"]["symbol"]} / {pool["token1"]["symbol"]} {int(pool["feeTier"]) / 1e4:g}%',
            "fee_tier": int(pool["feeTier"]) / 1e6,
            "token_is_token0": pool["token0"]["id"].lower() == token.lower(),
            "decimals0": int(pool["token0"]["decimals"]),
            "decimals1": int(pool["token1"]["decimals"]),
            "tvl_usd": float(pool["totalValueLockedUSD"]),
        }

    def load_pools(self):
        path = self.out_dir / "pools.json"
        if not path.exists():
            raise FileNotFoundError("OHLCVT/pools.json missing: download hourly data first")
        return json.loads(path.read_text(encoding="utf-8"))

    def download_hourly(self, refresh=False, selected=None):
        (self.out_dir / "liquidity").mkdir(parents=True, exist_ok=True)
        pools_path = self.out_dir / "pools.json"
        pools = self.load_pools() if pools_path.exists() else {}
        dfs = {}
        for symbol, asset in self.assets.items():

            if selected is not None and symbol not in selected:
                continue
            network, token, bucket = asset["network"], asset["address"], asset["bucket"]
            name = f"{symbol}ETH_60.csv"
            out = self.out_dir / name
            liquidity_path = self.out_dir / "liquidity" / name
            cached = out.exists() and liquidity_path.exists() and symbol in pools
            if cached and not refresh:
                df = self.read_csv(out)
                status = "cached"
            else:
                try:
                    pool = self.find_pool(network, token)
                    df = self.to_ohlcvt(self.download_hours(network, pool["id"]), pool, token)
                    if df.empty:
                        raise RuntimeError("no data in window")
                except Exception as exc:
                    print(f"{bucket:>10} {symbol:>7}: SKIPPED ({str(exc)[:150]})")
                    continue
                df[self.COLUMNS].to_csv(out, index=False)
                df[["Date", "Liquidity"]].to_csv(liquidity_path, index=False)
                pools[symbol] = self.pool_info(symbol, network, token, bucket, pool)
                status = "downloaded"
            pools_path.write_text(json.dumps(pools, indent=2), encoding="utf-8")
            dfs[symbol] = df
            print(f"{bucket:>10} {symbol:>7}: {len(df)} bars ({status})")
        return dfs

    @staticmethod
    def flatten_swap(swap):
        tx = swap["transaction"]
        return [swap["id"], int(swap["timestamp"]), swap["amount0"], swap["amount1"],
                swap["sqrtPriceX96"], tx["gasUsed"], tx["gasPrice"]]

    def download_swaps(self, symbol, meta, refresh=False):
        self.swap_dir.mkdir(parents=True, exist_ok=True)

        final_path = self.swap_dir / f"{symbol}.csv"

        # En refresh, utiliser un nouveau fichier temporaire
        path = (
            self.swap_dir / f"{symbol}.refresh.csv"
            if refresh else final_path
        )

        start = int(pd.Timestamp(self.start, tz="UTC").timestamp())
        end = int(pd.Timestamp(self.end, tz="UTC").timestamp())

        after, seen, n_total = start, set(), 0

        if refresh and path.exists():
            path.unlink()

        if path.exists():
            old = pd.read_csv(path, usecols=["id", "timestamp"])

            if len(old):
                after = int(old["timestamp"].max())
                seen = set(old.loc[old["timestamp"] == after, "id"])
                n_total = len(old)

                if after >= end - 1:
                    return n_total
        else:
            pd.DataFrame(
                columns=self.SWAP_COLUMNS
            ).to_csv(path, index=False)

        n_pages = 0
        while True:
            batch = self.query(meta["network"], self.SWAPS_QUERY, {
                "pool": meta["pool"], "after": str(after), "end": str(end), "first": self.page_size
            })["swaps"]
            new = [s for s in batch if s["id"] not in seen]
            if new:
                pd.DataFrame([self.flatten_swap(s) for s in new], columns=self.SWAP_COLUMNS).to_csv(
                    path, mode="a", header=False, index=False
                )
                n_total += len(new)
            if len(batch) < self.page_size:

                if refresh:
                    path.replace(final_path)

                return n_total
            last = int(batch[-1]["timestamp"])
            if last == after and not new:
                raise RuntimeError(f"{symbol}: more than {self.page_size} swaps at timestamp {after}")
            seen = ({s["id"] for s in batch if int(s["timestamp"]) == last}
                    | (seen if last == after else set()))
            after = last
            n_pages += 1
            if n_pages % 20 == 0:
                print(f"  {symbol}: {n_total} swaps up to {pd.to_datetime(after, unit='s')}")

    def read_swaps(self, symbol, meta):
        df = pd.read_csv(self.swap_dir / f"{symbol}.csv",
                         dtype={"sqrtPriceX96": str, "gasUsed": str, "gasPrice": str})
        df = df.drop_duplicates("id").sort_values(["timestamp", "id"]).reset_index(drop=True)
        price = df["sqrtPriceX96"].map(
            lambda x: self.sqrt_price_to_price(x, meta["decimals0"], meta["decimals1"])
        )
        if meta["token_is_token0"]:
            df["Price"] = price
            df["VolumeETH"] = df["amount1"].astype(float).abs()
        else:
            df["Price"] = 1.0 / price
            df["VolumeETH"] = df["amount0"].astype(float).abs()
        df["GasPriceGwei"] = df["gasPrice"].astype(float) / 1e9
        df["GasUsed"] = df["gasUsed"].astype(float)
        return df

    @staticmethod
    def make_bars(df, minutes):
        bucket = df["timestamp"] // (60 * minutes) * (60 * minutes)
        g = df.groupby(bucket)
        bars = pd.DataFrame({
            "Open": g["Price"].first(), "High": g["Price"].max(),
            "Low": g["Price"].min(), "Close": g["Price"].last(),
            "Volume": g["VolumeETH"].sum(), "Num": g["Price"].size(),
        })
        bars.index.name = "Date"
        return bars.reset_index()

    @staticmethod
    def make_gas(df, minutes):
        df = df[df["GasPriceGwei"] > 0]
        bucket = df["timestamp"] // (60 * minutes) * (60 * minutes)
        g = df.groupby(bucket)
        gas = pd.DataFrame({
            "GasPriceGwei": g["GasPriceGwei"].median(),
            "GasPriceGweiMean": g["GasPriceGwei"].mean(),
            "GasUsed": g["GasUsed"].median(),
            "Num": g["GasPriceGwei"].size(),
        })
        gas.index.name = "Date"
        return gas.reset_index()

    def build_minute_bars(self, pools):
        gas_parts = []
        for symbol, meta in pools.items():
            if not (self.swap_dir / f"{symbol}.csv").exists():
                continue
            df = self.read_swaps(symbol, meta)
            if df.empty:
                continue
            for minutes in (1, 5):
                self.make_bars(df, minutes).to_csv(self.out_dir / f"{symbol}ETH_{minutes}.csv", index=False)
            gas_parts.append(df[["timestamp", "GasPriceGwei", "GasUsed"]])
            print(f"{symbol}: {len(df)} swaps -> 1m, 5m")
        if gas_parts:
            all_swaps = pd.concat(gas_parts, ignore_index=True)
            for minutes in (1, 5, 60):
                self.make_gas(all_swaps, minutes).to_csv(self.out_dir / f"gas_{minutes}.csv", index=False)

    def download_all_swaps(self, selected=None, refresh=False):
        pools = self.load_pools()
        if selected is not None:
            pools = {
                symbol: meta
                for symbol, meta in pools.items()
                if symbol in selected
            }
        completed = {}
        for symbol, meta in pools.items():
            try:
                n = self.download_swaps(symbol, meta, refresh=refresh)
                completed[symbol] = meta
                print(f"{symbol}: {n} swaps downloaded")
            except Exception as exc:
                print(f"{symbol}: FAILED ({str(exc)[:150]}) - rerun to resume")
        self.build_minute_bars(completed)
        if len(completed) != len(pools):
            raise RuntimeError("Some Uniswap swap downloads failed; see messages above")

    def quality_report(self, dfs):
        expected = int((pd.Timestamp(self.end) - pd.Timestamp(self.start)) / pd.Timedelta(hours=1))
        return pd.DataFrame({
            "bucket": [self.assets[s]["bucket"] for s in dfs],
            "first_bar": [pd.to_datetime(df["Date"].iloc[0], unit="s") for df in dfs.values()],
            "last_bar": [pd.to_datetime(df["Date"].iloc[-1], unit="s") for df in dfs.values()],
            "bars": [len(df) for df in dfs.values()],
            "pct_missing": [100 * (1 - len(df) / expected) for df in dfs.values()],
            "max_abs_log_ret": [np.log(df["Close"]).diff().abs().max()
                                for df in dfs.values()],
        }, index=list(dfs))

    def extract(self, timeframe=None, bucket=None, download=True, refresh=False):
        """Return {symbol: OHLCVT DataFrame}. Downloads missing files only if requested.

        For timeframe 1 or 5, download=True may trigger a very large swap download.
        """
        timeframe = self.timeframe if timeframe is None else int(timeframe)
        bucket = self.bucket if bucket is None else bucket
        if timeframe not in (1, 5, 60):
            raise ValueError("timeframe must be 1, 5 or 60")
        selected = [s for s, a in self.assets.items() if bucket is None or a["bucket"] == bucket]
        missing = [s for s in selected if not (self.out_dir / f"{s}ETH_{timeframe}.csv").exists()]
        if download and (missing or refresh):
            if timeframe == 60:
                self.download_hourly(refresh=refresh, selected=selected)
            else:
                pools_path = self.out_dir / "pools.json"
                pools = self.load_pools() if pools_path.exists() else {}

                missing_pools = [
                    symbol for symbol in selected if symbol not in pools
                ]

                if missing_pools:
                    self.download_hourly(
                        refresh=True,
                        selected=missing_pools
                    )

                self.download_all_swaps(
                    selected=selected,
                    refresh=refresh
                )
        data = {}
        for symbol in selected:
            path = self.out_dir / f"{symbol}ETH_{timeframe}.csv"
            if not path.exists():
                print(f"[SKIP] Missing {path.name}")
                continue
            df = self.read_csv(path, header=True)
            start = int(pd.Timestamp(self.start, tz="UTC").timestamp())
            end = int(pd.Timestamp(self.end, tz="UTC").timestamp())
            df = df[(df["Date"] >= start) & (df["Date"] < end)]
            if df.empty:
                print(f"[SKIP] {symbol}: aucune donnée sur la période")
                continue
            data[symbol] = df[self.COLUMNS].reset_index(drop=True)
            print(f"[OK] {path.name}: {len(df)} lignes chargées")
        return data


def main():
    extractor = UniswapExtractor()
    data = extractor.extract()
    print(f"{len(data)} tokens loaded")


if __name__ == "__main__":
    main()

