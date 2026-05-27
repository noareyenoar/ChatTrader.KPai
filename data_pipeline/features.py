from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .gpu_utils import cleanup_cuda


@dataclass(frozen=True)
class ScalerStats:
    columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray


class FeatureFactory:
    """Vectorized feature generation with optional CUDA acceleration."""

    @staticmethod
    def _ensure_sorted(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def log_return(close: pd.Series, use_torch_cuda: bool = True) -> pd.Series:
        values = close.astype(float).to_numpy()
        if use_torch_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    tensor = torch.tensor(values, dtype=torch.float32, device="cuda")
                    out = torch.full_like(tensor, float("nan"))
                    out[1:] = torch.log(tensor[1:] / tensor[:-1])
                    arr = out.detach().cpu().numpy()
                    cleanup_cuda(tensor, out)
                    return pd.Series(arr, index=close.index)
            except Exception:
                pass

        out = np.full_like(values, np.nan, dtype=float)
        out[1:] = np.log(values[1:] / values[:-1])
        return pd.Series(out, index=close.index)

    @staticmethod
    def rolling_zscore(series: pd.Series, window: int = 64, eps: float = 1e-8) -> pd.Series:
        mu = series.rolling(window=window, min_periods=window).mean()
        sigma = series.rolling(window=window, min_periods=window).std()
        return (series - mu) / (sigma + eps)

    @staticmethod
    def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
        prev_close = frame["close"].shift(1)
        tr_components = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - prev_close).abs(),
                (frame["low"] - prev_close).abs(),
            ],
            axis=1,
        )
        tr = tr_components.max(axis=1)
        return tr.ewm(span=window, adjust=False).mean()

    @staticmethod
    def fractional_diff(series: pd.Series, d: float = 0.4, threshold: float = 1e-4) -> pd.Series:
        if not 0 < d < 1:
            raise ValueError("d must be in (0,1)")

        weights = [1.0]
        k = 1
        while True:
            w_k = -weights[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            weights.append(w_k)
            k += 1

        w = np.array(weights, dtype=float)
        x = series.astype(float).to_numpy()
        out = np.full_like(x, np.nan, dtype=float)
        width = len(w)
        conv = np.convolve(x, w, mode="full")
        out[width - 1 :] = conv[width - 1 : len(x)]
        return pd.Series(out, index=series.index)

    # =========================================================================
    # V2.0: CRYPTO ALPHA EXTRACTION — BTC BETA NEUTRALIZATION
    # =========================================================================

    @staticmethod
    def compute_btc_rolling_beta(
        asset_returns: pd.Series,
        btc_returns: pd.Series,
        window: int = 1440,
    ) -> pd.Series:
        """Rolling OLS Beta of asset log-returns vs BTC log-returns.

        β = Cov(asset, BTC) / Var(BTC)  over a rolling window.
        Returns 0.0 where insufficient data exists.
        """
        min_p = max(30, window // 4)
        cov = asset_returns.rolling(window, min_periods=min_p).cov(btc_returns)
        btc_var = btc_returns.rolling(window, min_periods=min_p).var()
        beta = cov / (btc_var.clip(lower=1e-12))
        return beta.fillna(0.0)

    @classmethod
    def add_btc_residual_features(
        cls,
        df: pd.DataFrame,
        btc_df: pd.DataFrame,
        window: int = 1440,
    ) -> pd.DataFrame:
        """Add BTC-neutralized residual features to df.

        Adds two columns:
          - ``btc_residual_return``  : asset log-return minus beta * BTC log-return
          - ``btc_relative_vol``     : asset rolling vol / BTC rolling vol

        The BTC frame is aligned on ``timestamp`` before computation; rows with
        no matching BTC data are forward-filled then back-filled.
        """
        btc_aligned = (
            btc_df[["timestamp", "close"]]
            .rename(columns={"close": "_btc_close"})
            .sort_values("timestamp")
        )
        merged = df.merge(btc_aligned, on="timestamp", how="left")
        merged["_btc_close"] = (
            merged["_btc_close"].ffill().bfill().fillna(merged["close"])
        )

        asset_ret = cls.log_return(merged["close"])
        btc_ret = cls.log_return(merged["_btc_close"])

        beta = cls.compute_btc_rolling_beta(asset_ret, btc_ret, window=window)

        min_p = max(30, window // 4)
        asset_vol = asset_ret.rolling(window, min_periods=min_p).std().fillna(1e-8)
        btc_vol = btc_ret.rolling(window, min_periods=min_p).std().fillna(1e-8)

        df = df.copy()
        df["btc_residual_return"] = (asset_ret - beta * btc_ret).fillna(0.0).values
        df["btc_relative_vol"] = (asset_vol / (btc_vol + 1e-8)).fillna(1.0).values
        return df

    @classmethod
    def build_trend_features(
        cls,
        frame: pd.DataFrame,
        btc_frame: pd.DataFrame | None = None,
        btc_beta_window: int = 1440,
    ) -> pd.DataFrame:
        """Build trend features.

        When ``btc_frame`` is provided (a DataFrame with ``timestamp`` and
        ``close`` columns for BTC), two extra BTC-neutralized features are
        appended: ``btc_residual_return`` and ``btc_relative_vol``.
        """
        df = cls._ensure_sorted(frame)
        close = df["close"].astype(float)
        df["log_return"] = cls.log_return(close)
        df["zscore_close_64"] = cls.rolling_zscore(close, window=64)
        df["ema_fast_12"] = close.ewm(span=12, adjust=False).mean()
        df["ema_slow_26"] = close.ewm(span=26, adjust=False).mean()
        df["ema_spread"] = df["ema_fast_12"] - df["ema_slow_26"]
        df["atr_14"] = cls.atr(df, window=14)
        df["price_slope_20"] = (close - close.shift(20)) / 20.0
        if btc_frame is not None:
            df = cls.add_btc_residual_features(df, btc_frame, window=btc_beta_window)
        return df

    @classmethod
    def build_mean_reversion_features(cls, frame: pd.DataFrame) -> pd.DataFrame:
        df = cls._ensure_sorted(frame)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        quote_volume = df["quote_volume"].astype(float)
        vwap = quote_volume / (volume.replace(0.0, np.nan))

        delta = close.diff().fillna(0.0)
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        rs = up.rolling(14, min_periods=14).mean() / (
            down.rolling(14, min_periods=14).mean() + 1e-8
        )
        rsi = 100.0 - (100.0 / (1.0 + rs))

        roll_mu = close.rolling(20, min_periods=20).mean()
        roll_std = close.rolling(20, min_periods=20).std()

        df["vwap_dev"] = (close - vwap) / (vwap.abs() + 1e-8)
        df["bb_distance"] = (close - roll_mu) / (2.0 * roll_std + 1e-8)
        df["zscore_close_20"] = cls.rolling_zscore(close, window=20)
        df["zscore_close_64"] = cls.rolling_zscore(close, window=64)
        df["rsi_14"] = rsi
        df["rsi_div_5"] = rsi - rsi.shift(5)
        # Binary regime flags — oversold/overbought for hard entry signal
        df["rsi_oversold"] = (rsi < 30.0).astype(np.float32)
        df["rsi_overbought"] = (rsi > 70.0).astype(np.float32)
        return df

    @classmethod
    def build_stat_arb_features(cls, frame: pd.DataFrame) -> pd.DataFrame:
        df = cls._ensure_sorted(frame)
        close = df["close"].astype(float)

        # ── Core features (v1) ─────────────────────────────────────────────
        df["fracdiff_close_d04"] = cls.fractional_diff(close, d=0.4)
        df["spread_z_64"] = cls.rolling_zscore(close, window=64)

        # ── Feature pivot (v2): mean-reversion signals ─────────────────────
        # Multi-window Z-scores (short + long horizon)
        df["spread_z_20"]  = cls.rolling_zscore(close, window=20)
        df["spread_z_128"] = cls.rolling_zscore(close, window=128)

        # Spread velocity (first difference of z-score — how fast the spread moves)
        df["spread_z_vel"] = df["spread_z_64"].diff(1).fillna(0.0)

        # Ornstein-Uhlenbeck half-life via rolling OLS: Δz = a*z_{t-1} + b
        # half_life = ln(2) / |a|; clipped to [1, 200] bars for stability
        df["ou_halflife"] = cls._rolling_ou_halflife(df["spread_z_64"], window=128)

        # Hurst exponent proxy: variance ratio (log(var_tau) / log(tau)) over tau=[1,4,16]
        # H < 0.5 → mean-reverting; H > 0.5 → trending; H = 0.5 → random walk
        df["hurst_proxy"] = cls._rolling_hurst_proxy(close, short=4, long=16, window=128)

        # Entry signal: z-score crossed threshold and Hurst < 0.5 (mean-reverting regime)
        thr = 1.5
        z = df["spread_z_64"]
        mr_regime = (df["hurst_proxy"] < 0.5).astype(float)
        df["entry_long_signal"]  = ((z < -thr) & (mr_regime == 1)).astype(float)
        df["entry_short_signal"] = ((z >  thr) & (mr_regime == 1)).astype(float)

        return df

    @staticmethod
    def _rolling_ou_halflife(zscore: pd.Series, window: int = 128) -> pd.Series:
        """Estimate OU mean-reversion half-life via rolling AR(1) Pearson correlation.

        AR(1) coefficient ρ ≈ corr(z_t, z_{t-1}) on a rolling window.
        Half-life = -ln(2) / ln(ρ).
        Result clipped to [1, 500] bars; NaN filled forward then with 50.
        """
        z_lag = zscore.shift(1)
        # Rolling Pearson correlation is O(n) using pandas optimized kernel
        rho = z_lag.rolling(window, min_periods=max(20, window // 4)).corr(zscore)
        # Only meaningful when 0 < rho < 1 (mean-reverting)
        rho_clipped = rho.clip(1e-6, 1.0 - 1e-6)
        ln_rho = np.log(rho_clipped.to_numpy(dtype=np.float64))
        # Avoid log(0) — set to zero for undefined cases
        ln_rho = np.where(np.isfinite(ln_rho) & (ln_rho < -1e-9), ln_rho, np.nan)
        hl_arr = np.where(np.isfinite(ln_rho), -np.log(2.0) / ln_rho, np.nan)
        hl_arr = np.clip(hl_arr, 1.0, 500.0)
        result = pd.Series(hl_arr, index=zscore.index)
        result = result.ffill().fillna(50.0)
        return result

    @staticmethod
    def _rolling_hurst_proxy(close: pd.Series, short: int = 4, long: int = 16,
                              window: int = 128) -> pd.Series:
        """Variance-ratio Hurst proxy using rolling windows of sub-sampled returns.

        H_proxy = 0.5 * log2(Var(long-period returns) / Var(short-period returns))
                             / log2(long / short)

        Vectorised: compute rolling variance of sub-sampled return series.
        H < 0.5 → mean-reverting; H > 0.5 → trending.
        """
        log_c = np.log(close.to_numpy(dtype=np.float64) + 1e-10)

        # Short-period returns: tau=short
        r_short = pd.Series(np.diff(log_c[::short], prepend=log_c[0]))
        # Align back to original index length using repeat (approximate)
        r_short_full = r_short.reindex(range(len(log_c)), method="ffill").fillna(0.0)

        # Long-period returns: tau=long
        r_long = pd.Series(np.diff(log_c[::long], prepend=log_c[0]))
        r_long_full = r_long.reindex(range(len(log_c)), method="ffill").fillna(0.0)

        # Rolling variance
        win_short = max(4, window // short)
        win_long  = max(4, window // long)
        min_p = max(2, min(win_short, win_long) // 2)
        var_short = r_short_full.rolling(win_short, min_periods=min_p).var().values
        var_long  = r_long_full.rolling(win_long,   min_periods=min_p).var().values

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((var_short > 1e-15) & (var_long > 1e-15),
                             var_long / var_short, np.nan)
            hurst = np.where(np.isfinite(ratio) & (ratio > 0),
                             0.5 * np.log(ratio) / np.log(long / short),
                             0.5)
        hurst = np.clip(hurst, 0.0, 1.0)

        result = pd.Series(hurst, index=close.index)
        return result.ffill().fillna(0.5)

    @classmethod
    def build_scalper_features(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Delegate to quant_core.scalper_data._build_scalper_features.
        Returns DataFrame with SCALPER_FEATURES columns."""
        try:
            from quant_core.scalper_data import _build_scalper_features
            return _build_scalper_features(frame)
        except ImportError:
            return frame.copy()

    @classmethod
    def build_discretionary_features(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Build DISC_TAB_FEATURES using data already available in trend features.
        Returns DataFrame with DISC_TAB_FEATURES columns added."""
        df = cls._ensure_sorted(frame)
        close = df["close"].astype(float)
        # log_return
        df["log_return"] = np.log1p(close.pct_change()).fillna(0.0)
        # zscore_close_64 (already in build_trend_features, replicate here for independence)
        df["zscore_close_64"] = cls.rolling_zscore(close, window=64)
        # ema_spread
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        df["ema_spread"] = (ema_fast - ema_slow) / (close.abs() + 1e-8)
        # atr_14
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        tr = (high - low).combine(
            (high - close.shift()).abs(), np.maximum
        ).combine(
            (low - close.shift()).abs(), np.maximum
        )
        df["atr_14"] = tr.rolling(14, min_periods=14).mean()
        # price_slope_20
        df["price_slope_20"] = close.diff(20).fillna(0.0) / (close.shift(20).abs() + 1e-8)
        return df

    @staticmethod
    def fit_scaler_train_only(train_df: pd.DataFrame, columns: Sequence[str]) -> ScalerStats:
        x = train_df.loc[:, columns].to_numpy(dtype=float)
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return ScalerStats(columns=columns, mean=mean, std=std)

    @staticmethod
    def transform_with_scaler(df: pd.DataFrame, scaler: ScalerStats) -> pd.DataFrame:
        out = df.copy()
        x = out.loc[:, list(scaler.columns)].to_numpy(dtype=float)
        scaled = (x - scaler.mean) / scaler.std
        # Assign column-by-column to avoid pandas dtype broadcast TypeError
        # (some source columns may be float32 e.g. rsi_oversold; write back as float64)
        for col, vals in zip(scaler.columns, scaled.T):
            out[col] = vals
        return out

    # ========================================================================
    # PHASE 2: INFORMATION-DRIVEN BARS (TICK/VOLUME)
    # ========================================================================

    @staticmethod
    def build_tick_bars(trades: pd.DataFrame, n_trades: int = 1000) -> pd.DataFrame:
        """
        Resample trades into Tick Bars: each bar = n_trades aggregated trades.
        
        Args:
            trades: DataFrame with columns [timestamp, price, quantity, is_buyer_maker, ...]
            n_trades: Number of trades per bar
            
        Returns:
            DataFrame with OHLCV + timestamp for each tick bar
        """
        trades = trades.sort_values("timestamp").reset_index(drop=True)
        trades["bar_id"] = trades.index // n_trades
        
        bars = []
        for bar_id, group in trades.groupby("bar_id"):
            if len(group) == 0:
                continue
            
            prices = group["price"].astype(float)
            quantities = group["quantity"].astype(float)
            
            bar = {
                "timestamp": group["timestamp"].iloc[-1],
                "open": prices.iloc[0],
                "high": prices.max(),
                "low": prices.min(),
                "close": prices.iloc[-1],
                "volume": quantities.sum(),
                "trade_count": len(group),
                "vwap": (prices * quantities).sum() / quantities.sum(),
            }
            bars.append(bar)
        
        return pd.DataFrame(bars)

    @staticmethod
    def build_volume_bars(trades: pd.DataFrame, volume_threshold: float = 100.0) -> pd.DataFrame:
        """
        Resample trades into Volume Bars: each bar = volume_threshold cumulative volume.
        
        Args:
            trades: DataFrame with columns [timestamp, price, quantity, ...]
            volume_threshold: Cumulative quote volume per bar (in base currency)
            
        Returns:
            DataFrame with OHLCV + timestamp for each volume bar
        """
        trades = trades.sort_values("timestamp").reset_index(drop=True)
        trades["cumsum_qty"] = trades["quantity"].astype(float).cumsum()
        trades["bar_id"] = (trades["cumsum_qty"] / volume_threshold).astype(int)
        
        bars = []
        for bar_id, group in trades.groupby("bar_id"):
            if len(group) == 0:
                continue
            
            prices = group["price"].astype(float)
            quantities = group["quantity"].astype(float)
            
            bar = {
                "timestamp": group["timestamp"].iloc[-1],
                "open": prices.iloc[0],
                "high": prices.max(),
                "low": prices.min(),
                "close": prices.iloc[-1],
                "volume": quantities.sum(),
                "trade_count": len(group),
                "vwap": (prices * quantities).sum() / quantities.sum(),
            }
            bars.append(bar)
        
        return pd.DataFrame(bars)

    # ========================================================================
    # MICROSTRUCTURE FEATURES
    # ========================================================================

    @staticmethod
    def compute_ofi(trades: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Order Flow Imbalance (OFI): Net signed volume (buyer-initiated - seller-initiated).
        
        Args:
            trades: DataFrame with [quantity, is_buyer_maker, ...]
            window: Rolling window for aggregation
            
        Returns:
            pd.Series of OFI values
        """
        trades = trades.copy()
        trades["quantity"] = trades["quantity"].astype(float)
        trades["is_buyer_maker"] = trades["is_buyer_maker"].astype(bool)
        
        # Buyer-initiated: is_buyer_maker=False (opposite of taker side)
        trades["signed_qty"] = trades["quantity"].where(
            ~trades["is_buyer_maker"], -trades["quantity"]
        )
        
        ofi = trades["signed_qty"].rolling(window=window, min_periods=1).sum()
        return ofi

    @staticmethod
    def compute_vpin(trades: pd.DataFrame, bucket_size: int = 1000) -> pd.Series:
        """
        Volume-Synchronized Probability of Informed Trading (VPIN).
        Approximated as the ratio of directional volume imbalance to total volume.
        
        Args:
            trades: DataFrame with [quantity, is_buyer_maker, ...]
            bucket_size: Number of trades per bucket for estimation
            
        Returns:
            pd.Series of VPIN estimates
        """
        trades = trades.copy()
        trades["quantity"] = trades["quantity"].astype(float)
        trades["is_buyer_maker"] = trades["is_buyer_maker"].astype(bool)
        
        trades["bar_id"] = trades.index // bucket_size
        
        vpin_values = []
        for bar_id, group in trades.groupby("bar_id"):
            buy_vol = group.loc[~group["is_buyer_maker"], "quantity"].sum()
            sell_vol = group.loc[group["is_buyer_maker"], "quantity"].sum()
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                vpin = abs(buy_vol - sell_vol) / total_vol
            else:
                vpin = 0.0
            
            vpin_values.extend([vpin] * len(group))
        
        return pd.Series(vpin_values, index=trades.index)

    @staticmethod
    def compute_spread_dynamics(book: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Extract spread and depth dynamics from bookTicker L2 snapshots.
        
        Args:
            book: DataFrame with [best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, ...]
            window: Rolling window for trend
            
        Returns:
            DataFrame with spread, mid_price, depth_imbalance, spread_velocity
        """
        book = book.copy()
        book["best_bid_price"] = book["best_bid_price"].astype(float)
        book["best_ask_price"] = book["best_ask_price"].astype(float)
        book["best_bid_qty"] = book["best_bid_qty"].astype(float)
        book["best_ask_qty"] = book["best_ask_qty"].astype(float)
        
        result = pd.DataFrame()
        result["mid_price"] = (book["best_bid_price"] + book["best_ask_price"]) / 2.0
        result["spread"] = book["best_ask_price"] - book["best_bid_price"]
        result["spread_pct"] = result["spread"] / result["mid_price"]
        result["depth_imbalance"] = (book["best_bid_qty"] - book["best_ask_qty"]) / (
            book["best_bid_qty"] + book["best_ask_qty"] + 1e-8
        )
        result["spread_velocity"] = result["spread"].diff().rolling(window).mean()
        
        return result

    # ========================================================================
    # MULTI-INSTRUMENT FEATURES (SPOT + FUTURES + OPTIONS)
    # ========================================================================

    @staticmethod
    def compute_spot_futures_basis(
        spot_close: pd.Series,
        futures_close: pd.Series,
        eps: float = 1e-8,
    ) -> pd.Series:
        """Compute normalized spot-futures basis."""
        spot = spot_close.astype(float)
        fut = futures_close.astype(float)
        return (fut - spot) / (spot.abs() + eps)

    @staticmethod
    def compute_funding_spread(
        funding_um: pd.Series,
        funding_cm: pd.Series,
    ) -> pd.Series:
        """Compute funding differential between USD-M and COIN-M."""
        um = funding_um.astype(float)
        cm = funding_cm.astype(float)
        return um - cm

    @staticmethod
    def compute_oi_delta(open_interest: pd.Series, periods: int = 12) -> pd.Series:
        """Compute open-interest momentum as pct delta."""
        oi = open_interest.astype(float)
        return oi.pct_change(periods=periods)

    @staticmethod
    def compute_funding_sentiment(funding_rate: pd.Series, window: int = 24) -> pd.Series:
        """Compute rolling funding sentiment signal."""
        fr = funding_rate.astype(float)
        return fr.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def compute_cross_instrument_ofi(
        spot_signed_qty: pd.Series,
        futures_signed_qty: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Compute futures-vs-spot signed flow imbalance."""
        spot = spot_signed_qty.astype(float)
        fut = futures_signed_qty.astype(float)
        return (fut - spot).rolling(window=window, min_periods=1).sum()

    @staticmethod
    def build_bvol_global_state(
        bvol_btc: pd.Series,
        bvol_eth: pd.Series,
    ) -> pd.DataFrame:
        """Build option-implied volatility global state features."""
        btc = bvol_btc.astype(float)
        eth = bvol_eth.astype(float)
        out = pd.DataFrame()
        out["bvol_btc"] = btc
        out["bvol_eth"] = eth
        out["bvol_spread"] = btc - eth
        out["bvol_ratio"] = btc / (eth.abs() + 1e-8)
        out["bvol_regime_z"] = FeatureFactory.rolling_zscore((btc + eth) / 2.0, window=20)
        return out

    # ========================================================================
    # SYNTHETIC DATA GENERATION
    # ========================================================================

    @staticmethod
    def generate_synthetic_garch(returns: np.ndarray, n_sim: int = 100, alpha: float = 0.05, beta: float = 0.90) -> np.ndarray:
        """
        Generate synthetic price paths using GARCH(1,1) volatility model.
        Useful for stress-testing scalper/MM models on volatility regimes.
        
        Args:
            returns: Historical log returns
            n_sim: Number of synthetic paths to generate
            alpha, beta: GARCH parameters
            
        Returns:
            Array of shape (len(returns), n_sim) with synthetic log returns
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[~np.isnan(returns)]
        
        mean_ret = np.mean(returns)
        omega = np.var(returns) * (1 - alpha - beta)
        
        T = len(returns)
        synthetic = np.zeros((T, n_sim))
        
        for sim in range(n_sim):
            sigma2 = np.var(returns)
            path = np.zeros(T)
            
            for t in range(T):
                sigma2 = omega + alpha * path[t-1]**2 + beta * sigma2
                path[t] = mean_ret + np.sqrt(max(sigma2, 1e-8)) * np.random.randn()
            
            synthetic[:, sim] = path
        
        return synthetic

    @staticmethod
    def generate_synthetic_hmm(prices: np.ndarray, n_regimes: int = 3, n_sim: int = 100) -> np.ndarray:
        """
        Generate synthetic price paths using Hidden Markov Model (HMM).
        Simulates regime-switching (e.g., quiet/normal/chaotic markets).
        
        Args:
            prices: Historical prices
            n_regimes: Number of hidden regimes
            n_sim: Number of synthetic paths
            
        Returns:
            Array of shape (len(prices), n_sim) with synthetic prices
        """
        prices = np.asarray(prices, dtype=float)
        log_returns = np.diff(np.log(prices))
        
        # Estimate regime statistics from historical returns
        quantiles = np.percentile(np.abs(log_returns), np.linspace(0, 100, n_regimes + 1))
        regime_vols = [(quantiles[i] + quantiles[i+1]) / 2 for i in range(n_regimes)]
        
        # Simple Markov transition matrix (stay-in-regime bias)
        transition = np.eye(n_regimes) * 0.7 + np.ones((n_regimes, n_regimes)) * 0.3 / n_regimes
        
        T = len(prices)
        synthetic = np.zeros((T, n_sim))
        
        for sim in range(n_sim):
            current_price = prices[-1]
            regime = np.random.randint(0, n_regimes)
            path = [current_price]
            
            for t in range(1, T):
                # Switch regime
                regime = np.random.choice(n_regimes, p=transition[regime])
                # Generate return under regime volatility
                ret = regime_vols[regime] * np.random.randn()
                current_price = current_price * np.exp(ret)
                path.append(current_price)
            
            synthetic[:, sim] = path
        
        return synthetic

    @staticmethod
    def generate_stationary_bootstrap(data: pd.DataFrame, n_samples: int = 100, block_size: int = 20) -> list:
        """
        Generate synthetic price/returns using Stationary Bootstrap.
        Preserves autocorrelation structure while randomizing block boundaries.
        
        Args:
            data: Historical price/return series
            n_samples: Number of synthetic samples to generate
            block_size: Average block length
            
        Returns:
            List of synthetic DataFrames
        """
        samples = []
        T = len(data)
        prob_switch = 1.0 / block_size
        
        for _ in range(n_samples):
            indices = []
            idx = 0
            
            while len(indices) < T:
                # Determine block length (geometric distribution)
                block_len = int(np.random.geometric(prob_switch))
                block = np.arange(idx, min(idx + block_len, T))
                indices.extend(block)
                idx = np.random.randint(0, T)
            
            indices = indices[:T]
            synthetic_df = data.iloc[indices].reset_index(drop=True)
            samples.append(synthetic_df)
        
        return samples

    # ========================================================================
    # TRIPLE BARRIER LABELING (FOR SCALPER MODELS)
    # ========================================================================

    @staticmethod
    def apply_triple_barrier_labels(
        prices: pd.Series,
        upper_pct: float = 0.001,
        lower_pct: float = 0.001,
        max_bars: int = 20,
    ) -> pd.Series:
        """
        Triple Barrier labeling for scalper models.
        Each bar is labeled based on which barrier is touched first:
        - Upper barrier (profit-taking): LONG
        - Lower barrier (stop-loss): SHORT
        - Time barrier (timeout): FLAT
        
        Args:
            prices: Series of close prices
            upper_pct: Upper barrier as % of entry price (profit-taking)
            lower_pct: Lower barrier as % of entry price (stop-loss)
            max_bars: Maximum bars to look ahead for barrier touch
            
        Returns:
            Series of labels: {-1: SHORT, 0: FLAT, 1: LONG}
        """
        prices = prices.astype(float).to_numpy()
        labels = np.zeros(len(prices), dtype=int)
        
        for i in range(len(prices) - max_bars):
            entry_price = prices[i]
            upper_barrier = entry_price * (1 + upper_pct)
            lower_barrier = entry_price * (1 - lower_pct)
            
            future_prices = prices[i+1 : i+1+max_bars]
            
            touched_upper = np.any(future_prices >= upper_barrier)
            touched_lower = np.any(future_prices <= lower_barrier)
            
            if touched_upper and not touched_lower:
                labels[i] = 1  # LONG
            elif touched_lower and not touched_upper:
                labels[i] = -1  # SHORT
            else:
                labels[i] = 0  # FLAT (timeout or both touched)
        
        return pd.Series(labels, index=prices.index if hasattr(prices, 'index') else range(len(prices)))

    @staticmethod
    def apply_adaptive_triple_barrier(
        prices: pd.Series,
        returns_vol: pd.Series,
        vol_quantile_low: float = 0.33,
        vol_quantile_high: float = 0.67,
        barrier_scale_low: float = 0.0005,
        barrier_scale_normal: float = 0.001,
        barrier_scale_high: float = 0.002,
        max_bars: int = 20,
    ) -> pd.Series:
        """
        Adaptive Triple Barrier: barrier widths scale with volatility regime.
        Low vol → tight barriers (0.05%), Normal → standard (0.1%), High → wider (0.2%).
        
        Args:
            prices: Series of close prices
            returns_vol: Rolling volatility estimate
            vol_quantile_low, vol_quantile_high: Regime boundaries
            barrier_scale_*: Barrier width multipliers per regime
            max_bars: Lookahead horizon
            
        Returns:
            Series of labels: {-1: SHORT, 0: FLAT, 1: LONG}
        """
        prices = prices.astype(float).to_numpy()
        returns_vol = returns_vol.astype(float).to_numpy()
        
        q_low = np.nanpercentile(returns_vol, vol_quantile_low * 100)
        q_high = np.nanpercentile(returns_vol, vol_quantile_high * 100)
        
        labels = np.zeros(len(prices), dtype=int)
        
        for i in range(len(prices) - max_bars):
            vol = returns_vol[i] if not np.isnan(returns_vol[i]) else 0.001
            
            # Determine regime
            if vol < q_low:
                barrier_width = barrier_scale_low
            elif vol < q_high:
                barrier_width = barrier_scale_normal
            else:
                barrier_width = barrier_scale_high
            
            entry_price = prices[i]
            upper_barrier = entry_price * (1 + barrier_width)
            lower_barrier = entry_price * (1 - barrier_width)
            
            future_prices = prices[i+1 : i+1+max_bars]
            
            touched_upper = np.any(future_prices >= upper_barrier)
            touched_lower = np.any(future_prices <= lower_barrier)
            
            if touched_upper and not touched_lower:
                labels[i] = 1  # LONG
            elif touched_lower and not touched_upper:
                labels[i] = -1  # SHORT
            else:
                labels[i] = 0  # FLAT
        
        return pd.Series(labels, index=prices.index if hasattr(prices, 'index') else range(len(prices)))
