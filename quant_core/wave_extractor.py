"""Wave Extractor: ZigZag algorithm for identifying price peaks and troughs.

This module implements peak/trough detection and wave property labeling,
essential for the TG-MNN (Temporal-Gradient Markov Neural Network) model.

Key Components:
- ZigZag peak/trough detection
- Wave magnitude and duration computation
- Discrete state classification (Steady, Up, Down)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class WaveProperties:
    """Properties of a price wave at a given timestamp."""
    magnitude: float      # Distance to next peak/trough
    duration: int         # Bars until next peak/trough
    state: int            # 0=Steady, 1=Up, 2=Down


class ZigZagExtractor:
    """Identifies peaks/troughs using ZigZag algorithm and labels wave properties."""

    def __init__(self, threshold: float = 0.005):
        """
        Args:
            threshold: Minimum percentage move to identify new peak/trough (default 0.5%).
        """
        self.threshold = threshold

    def extract_peaks_and_troughs(self, close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Identify peaks and troughs using ZigZag algorithm.

        Returns:
            peaks: Boolean array where True indicates a peak
            troughs: Boolean array where True indicates a trough
        """
        n = len(close)
        peaks = np.zeros(n, dtype=bool)
        troughs = np.zeros(n, dtype=bool)

        if n < 3:
            return peaks, troughs

        # Find the first significant move direction
        i = 1
        while i < n and abs(close[i] - close[0]) / close[0] < self.threshold:
            i += 1

        if i >= n:
            # No significant move; mark first as trough, last as peak
            troughs[0] = True
            peaks[-1] = True
            return peaks, troughs

        last_turn = 0
        is_up = close[i] > close[0]

        for j in range(i + 1, n):
            if is_up:
                if close[j] > close[last_turn]:
                    # Continue up
                    last_turn = j
                elif (close[last_turn] - close[j]) / close[last_turn] > self.threshold:
                    # Significant move down; mark peak
                    peaks[last_turn] = True
                    is_up = False
                    last_turn = j
            else:
                if close[j] < close[last_turn]:
                    # Continue down
                    last_turn = j
                elif (close[j] - close[last_turn]) / close[last_turn] > self.threshold:
                    # Significant move up; mark trough
                    troughs[last_turn] = True
                    is_up = True
                    last_turn = j

        # Mark the final extreme
        if is_up:
            peaks[last_turn] = True
        else:
            troughs[last_turn] = True

        return peaks, troughs

    def compute_wave_properties(
        self, close: np.ndarray, peaks: np.ndarray, troughs: np.ndarray
    ) -> np.ndarray:
        """
        Compute wave properties (magnitude, duration, state) for each timestamp.

        Args:
            close: Close price array [T]
            peaks: Boolean array indicating peaks [T]
            troughs: Boolean array indicating troughs [T]

        Returns:
            properties: Array of WaveProperties, one per timestamp [T]
        """
        n = len(close)
        properties = np.zeros(n, dtype=object)

        # Find indices of all extrema
        peak_indices = np.where(peaks)[0]
        trough_indices = np.where(troughs)[0]
        extrema_indices = np.sort(np.concatenate([peak_indices, trough_indices]))
        extrema_types = np.array([
            'peak' if peaks[i] else 'trough'
            for i in extrema_indices
        ])

        for t in range(n):
            # Find next extremum
            future_extrema = extrema_indices[extrema_indices > t]
            if len(future_extrema) == 0:
                # No future extrema; project based on local gradient
                next_idx = n - 1
                magnitude = abs(close[-1] - close[t])
                duration = n - 1 - t
            else:
                next_idx = future_extrema[0]
                magnitude = abs(close[next_idx] - close[t])
                duration = next_idx - t

            # Compute state based on current gradient
            if t == 0:
                state = 0  # Steady
            else:
                grad = (close[t] - close[t - 1]) / close[t - 1]
                if abs(grad) < 0.0005:  # Less than 0.05% move
                    state = 0  # Steady
                elif grad > 0:
                    state = 1  # Up
                else:
                    state = 2  # Down

            properties[t] = WaveProperties(
                magnitude=magnitude,
                duration=duration,
                state=state
            )

        return properties

    def extract_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract wave labels for all timestamps in a DataFrame.

        Args:
            df: DataFrame with 'close' column

        Returns:
            df: DataFrame with added 'target_magnitude', 'target_duration', 'target_state'
        """
        close = df['close'].values.astype(float)
        peaks, troughs = self.extract_peaks_and_troughs(close)
        properties = self.compute_wave_properties(close, peaks, troughs)

        df_out = df.copy()
        df_out['target_magnitude'] = np.array([p.magnitude for p in properties])
        df_out['target_duration'] = np.array([p.duration for p in properties], dtype=np.int32)
        df_out['target_state'] = np.array([p.state for p in properties], dtype=np.int32)
        df_out['is_peak'] = peaks.astype(int)
        df_out['is_trough'] = troughs.astype(int)

        return df_out


class WaveFeatureBuilder:
    """Builds TG-MNN-compatible feature and label sets."""

    FEATURE_COLUMNS = [
        "log_return",
        "zscore_close_64",
        "ema_spread",
        "atr_14",
        "price_slope_20",
    ]

    @staticmethod
    def build_wave_features(df: pd.DataFrame) -> pd.DataFrame:
        """Build wave-aware features for TG-MNN."""
        from data_pipeline.features import FeatureFactory

        # Start with standard trend features
        df_feat = FeatureFactory.build_trend_features(df)

        # Extract wave labels
        extractor = ZigZagExtractor(threshold=0.005)
        df_feat = extractor.extract_labels(df_feat)

        return df_feat

    @staticmethod
    def split_features_and_targets(
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features and targets from a prepared DataFrame.

        Returns:
            features: Shape [N, F] - feature values
            target_state: Shape [N] - discrete states {0, 1, 2}
            target_magnitude: Shape [N] - regression targets (magnitude)
            target_duration: Shape [N] - regression targets (duration)
        """
        features = df[WaveFeatureBuilder.FEATURE_COLUMNS].values.astype(np.float32)
        target_state = df['target_state'].values.astype(np.int64)
        target_magnitude = df['target_magnitude'].values.astype(np.float32)
        target_duration = df['target_duration'].values.astype(np.float32)

        return features, target_state, target_magnitude, target_duration
