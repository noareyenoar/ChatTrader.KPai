"""Market Making RL simulation environment — Phase 4+ with Microstructure & Curriculum.

Roadmap v26-4 upgrades included here:
- state adds OFI-like pressure proxy and inventory skew context
- reward uses PnL + spread capture - inventory skew penalty
- asymmetric power-law shaping to prevent large loss domination
- short warm-up period before rewards are emitted

Phase 4+ Enhancements:
- Market impact: execution price degrades with order size and book depth
- Dynamic fill probability based on spread and volatility
- Funding rate and open interest in state space
- Curriculum learning wrapper for phased training (trending → normal → chaotic)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

import numpy as np


INVENTORY_LIMIT = 10.0       # max units held
TRANSACTION_COST = 0.0005    # 0.05% per fill
FILL_PROBABILITY_BASE = 0.3  # base probability of fill at quoted spread


class TrainingCurriculum(Enum):
    """Curriculum learning phases for RL training."""
    EASY = "easy"           # Strongly trending data with low inventory risk
    MEDIUM = "medium"       # Moderate volatility and spread variance
    HARD = "hard"          # High noise, sideways, and volatility regime


@dataclass
class MMState:
    inventory: float = 0.0
    cash: float = 0.0
    mid_price: float = 100.0
    step: int = 0
    episode_length: int = 200
    pnl_history: list = field(default_factory=list)
    funding_rate: float = 0.0      # Current funding rate (from perpetuals)
    open_interest: float = 0.0     # Current open interest (normalized)
    equity_peak: float = 0.0
    initial_equity: float = 0.0


class MarketMakingEnv:
    """Tabular market-making environment driven by historical price series.

    The environment replays a segment of historical OHLCV data with optional L2 book depth.
    At each step:
      1. Agent quotes bid/ask offsets around mid-price.
      2. Execution price degrades based on order size (market impact).
      3. Fills are simulated probabilistically, accounting for spread and volatility.
      4. Inventory and cash are updated.
      5. Reward = realized PnL − λ * |inventory| − transaction cost − market impact.
      
    Optional enhancements:
      - Book depth data (L2) for realistic fill probability modeling
      - Funding rate and open interest in state space
      - Curriculum learning: difficulty phases (trending → normal → chaotic)
    """

    STATE_DIM = 10  # [inventory_norm, mid_change, spread, vol, ofi_proxy, time, pnl_norm, inv_skew, funding_rate, oi_norm]
    CONTINUOUS_ACTION_DIM = 2   # [bid_offset, ask_offset]
    DISCRETE_ACTIONS = 3        # tight / medium / wide

    def __init__(
        self,
        price_series: np.ndarray,
        episode_length: int = 200,
        inventory_lambda: float = 0.1,
        warmup_steps: int = 20,
        alpha_pos: float = 0.75,
        alpha_neg: float = 1.35,
        reward_scale: float = 1.0,
        seed: int = 42,
        book_depth: Optional[dict] = None,
        funding_rates: Optional[np.ndarray] = None,
        open_interests: Optional[np.ndarray] = None,
        market_impact_scale: float = 0.0001,
        curriculum: TrainingCurriculum = TrainingCurriculum.MEDIUM,
        survival_bonus: float = 0.0005,
        inventory_penalty_power: float = 2.0,
        max_drawdown_terminate: float = 0.85,
    ):
        """
        Initialize the Market Making Environment.
        
        Args:
            price_series: Historical mid-prices (1D array)
            episode_length: Length of each episode
            inventory_lambda: Inventory risk penalty weight
            warmup_steps: Steps before reward is emitted
            alpha_pos: Power law exponent for positive PnL
            alpha_neg: Power law exponent for negative PnL
            reward_scale: Reward scaling factor
            seed: Random seed
            book_depth: Optional dict with 'bid_qty' and 'ask_qty' (1D arrays matching prices)
            funding_rates: Optional array of funding rates (same length as prices)
            open_interests: Optional array of open interests (same length as prices)
            market_impact_scale: Coefficient for market impact penalty
            curriculum: Training curriculum phase (EASY/MEDIUM/HARD)
        """
        self.prices = price_series.astype(np.float32)
        self.episode_length = min(episode_length, len(price_series) - 1)
        self.inventory_lambda = inventory_lambda
        self.warmup_steps = max(0, int(warmup_steps))
        self.alpha_pos = float(alpha_pos)
        self.alpha_neg = float(alpha_neg)
        self.reward_scale = float(reward_scale)
        self.market_impact_scale = float(market_impact_scale)
        self.curriculum = curriculum
        self.survival_bonus = float(survival_bonus)
        self.inventory_penalty_power = float(inventory_penalty_power)
        self.max_drawdown_terminate = float(np.clip(max_drawdown_terminate, 0.1, 0.99))
        self.rng = np.random.default_rng(seed)
        
        # Optional L2 book data
        self.book_depth = book_depth or {}
        self.funding_rates = funding_rates if funding_rates is not None else np.zeros(len(self.prices))
        self.open_interests = open_interests if open_interests is not None else np.zeros(len(self.prices))
        
        self._state = MMState()
        self._cursor = 0
        self._vol_window = 20
        self._initial_equity = 1.0
        
        # Curriculum-specific adjustments
        self._apply_curriculum_bias()

    def _apply_curriculum_bias(self):
        """Apply curriculum-specific filters or biases to episode sampling."""
        if self.curriculum == TrainingCurriculum.EASY:
            # Easy: prefer trending regimes (use directional returns)
            self._vol_threshold = (
                np.percentile(np.abs(np.diff(self.prices) / (self.prices[:-1] + 1e-8)), 30)
            )
        elif self.curriculum == TrainingCurriculum.MEDIUM:
            # Medium: no special filtering (default)
            self._vol_threshold = None
        else:  # HARD
            # Hard: prefer high-volatility regimes
            self._vol_threshold = (
                np.percentile(np.abs(np.diff(self.prices) / (self.prices[:-1] + 1e-8)), 70)
            )

    @staticmethod
    def _signed_power(value: float, pos_exp: float, neg_exp: float) -> float:
        if value >= 0.0:
            return float(np.power(abs(value), pos_exp))
        return -float(np.power(abs(value), neg_exp))

    def _compute_market_impact(self, order_size: float, side: str = "buy") -> float:
        """
        Compute execution price impact from market impact model.
        Impact increases with order size and volatility, decreases with book depth.
        
        Args:
            order_size: Size of order (in units)
            side: 'buy' or 'sell'
            
        Returns:
            Impact as a fraction of mid-price
        """
        # Volatility factor
        lo = max(0, self._cursor - self._vol_window)
        recent = self.prices[lo : self._cursor + 1]
        if len(recent) > 1:
            vol = float(np.std(np.diff(recent) / (recent[:-1] + 1e-8)))
        else:
            vol = 0.001
        
        # Book depth factor (if available)
        depth = 1.0
        if "bid_qty" in self.book_depth and "ask_qty" in self.book_depth:
            if side == "buy":
                d = float(self.book_depth["ask_qty"][self._cursor])
            else:
                d = float(self.book_depth["bid_qty"][self._cursor])
            depth = 1.0 / max(1.0, d / 10.0)  # Normalize to typical order size
        
        # Impact formula: α * size * volatility / depth
        impact = (
            self.market_impact_scale * abs(order_size) * vol * depth
        )
        return float(np.clip(impact, 0.0, 0.05))  # Cap at 5% impact

    def _get_fill_probability(self, bid_offset: float, ask_offset: float) -> tuple[float, float]:
        """
        Compute fill probability for bid and ask sides.
        Accounts for spread, volatility, and book depth.
        
        Args:
            bid_offset: Bid offset as fraction of mid-price
            ask_offset: Ask offset as fraction of mid-price
            
        Returns:
            (bid_prob, ask_prob) tuple
        """
        # Base probability from spread
        spread_pct = bid_offset + ask_offset
        
        # Volatility discount (larger spread in volatile markets)
        lo = max(0, self._cursor - self._vol_window)
        recent = self.prices[lo : self._cursor + 1]
        if len(recent) > 1:
            vol = float(np.std(np.diff(recent) / (recent[:-1] + 1e-8)))
        else:
            vol = 0.001
        
        # Fill probability: higher for tighter spreads and lower volatility
        fill_prob = FILL_PROBABILITY_BASE / (1.0 + spread_pct * 200.0) * (0.5 + vol / max(vol, 0.01))
        
        return float(np.clip(fill_prob, 0.01, 0.9)), float(np.clip(fill_prob, 0.01, 0.9))

    @staticmethod
    def _equity(cash: float, inventory: float, mid: float) -> float:
        return float(cash + inventory * mid)

    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        if start_idx is None:
            max_start = max(1, len(self.prices) - self.episode_length - self._vol_window)
            start_idx = int(self.rng.integers(self._vol_window, max(self._vol_window + 1, max_start)))
        
        self._cursor = start_idx
        mid0 = float(self.prices[self._cursor])
        initial_cash = mid0 * INVENTORY_LIMIT
        self._state = MMState(
            cash=initial_cash,
            mid_price=float(self.prices[self._cursor]),
            episode_length=self.episode_length,
            funding_rate=float(self.funding_rates[self._cursor]) if len(self.funding_rates) > self._cursor else 0.0,
            open_interest=float(self.open_interests[self._cursor]) if len(self.open_interests) > self._cursor else 0.0,
        )
        eq0 = self._equity(self._state.cash, self._state.inventory, self._state.mid_price)
        self._initial_equity = max(1.0, abs(eq0))
        self._state.equity_peak = self._initial_equity
        self._state.initial_equity = self._initial_equity
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        s = self._state
        # Volatility: std of recent returns
        lo = max(0, self._cursor - self._vol_window)
        recent = self.prices[lo: self._cursor + 1]
        if len(recent) > 1:
            vol = float(np.std(np.diff(recent) / (recent[:-1] + 1e-8)))
        else:
            vol = 0.001

        # Mid-price change from last step
        if self._cursor > 0:
            mid_change = (self.prices[self._cursor] - self.prices[self._cursor - 1]) / (
                self.prices[self._cursor - 1] + 1e-8
            )
        else:
            mid_change = 0.0

        spread_pct = abs(mid_change) + vol

        # OFI proxy from recent signed returns magnitude (OHLCV fallback for no L2 data).
        if len(recent) > 2:
            ret = np.diff(recent) / (recent[:-1] + 1e-8)
            ofi_proxy = float(np.mean(np.sign(ret) * np.minimum(np.abs(ret) * 300.0, 3.0)))
        else:
            ofi_proxy = 0.0

        pnl_norm = np.clip((s.cash + s.inventory * s.mid_price) / (s.mid_price * INVENTORY_LIMIT + 1e-8), -1.0, 1.0)
        inventory_skew = np.clip(abs(s.inventory) / INVENTORY_LIMIT, 0.0, 1.0)
        
        # Normalize funding rate and OI
        funding_norm = np.clip(s.funding_rate * 100, -1.0, 1.0)
        oi_norm = np.clip(s.open_interest / 100.0, 0.0, 1.0)  # Assume OI in range [0, 100]

        return np.array([
            np.clip(s.inventory / INVENTORY_LIMIT, -1.0, 1.0),
            np.clip(mid_change * 100, -1.0, 1.0),
            np.clip(spread_pct * 100, 0.0, 1.0),
            np.clip(vol * 100, 0.0, 1.0),
            np.clip(ofi_proxy, -1.0, 1.0),
            s.step / max(1, s.episode_length),
            pnl_norm,
            inventory_skew,
            funding_norm,
            oi_norm,
        ], dtype=np.float32)

    def _discrete_to_offsets(self, action: int) -> tuple[float, float]:
        """Map discrete action to bid/ask offsets as fractions of mid-price."""
        spreads = [0.001, 0.003, 0.007]  # tight / medium / wide
        half = spreads[int(action)] / 2
        return half, half  # symmetric

    def step(self, action: np.ndarray, discrete: bool = False) -> tuple[np.ndarray, float, bool, dict]:
        s = self._state
        mid = self.prices[self._cursor]
        s.mid_price = mid

        if discrete:
            bid_off, ask_off = self._discrete_to_offsets(int(action))
        else:
            bid_off = float(np.clip(action[0], 0.0, 1.0)) * 0.01 + 0.0005
            ask_off = float(np.clip(action[1], 0.0, 1.0)) * 0.01 + 0.0005

        bid_price = mid * (1.0 - bid_off)
        ask_price = mid * (1.0 + ask_off)

        # Get dynamic fill probabilities
        bid_prob, ask_prob = self._get_fill_probability(bid_off, ask_off)
        
        bid_filled = self.rng.random() < bid_prob and s.inventory > -INVENTORY_LIMIT
        ask_filled = self.rng.random() < ask_prob and s.inventory < INVENTORY_LIMIT

        realized_pnl = 0.0
        market_impact_cost = 0.0
        
        if bid_filled:
            # Market impact: execution price worse than quoted
            impact = self._compute_market_impact(1.0, side="buy")
            actual_bid = bid_price * (1.0 - impact)
            s.inventory += 1.0
            s.cash -= actual_bid + mid * TRANSACTION_COST
            market_impact_cost += impact * mid
            
        if ask_filled:
            # Market impact: execution price worse than quoted
            impact = self._compute_market_impact(1.0, side="sell")
            actual_ask = ask_price * (1.0 - impact)
            s.inventory -= 1.0
            s.cash += actual_ask - mid * TRANSACTION_COST
            market_impact_cost += impact * mid
            
            if s.inventory < 0.0 and bid_filled:
                realized_pnl = actual_ask - actual_bid - 2 * mid * TRANSACTION_COST

        # Mark-to-market PnL change
        next_cursor = min(self._cursor + 1, len(self.prices) - 1)
        next_mid = float(self.prices[next_cursor])
        mtm_pnl = s.inventory * (next_mid - mid)

        total_pnl = mtm_pnl + realized_pnl
        inventory_skew = abs(s.inventory) / INVENTORY_LIMIT
        spread_capture = (ask_price - bid_price) * float(bid_filled and ask_filled)

        pnl_norm = total_pnl / (mid + 1e-8)
        spread_norm = spread_capture / (mid + 1e-8)
        impact_norm = market_impact_cost / (mid + 1e-8)
        
        inventory_penalty = self.inventory_lambda * (inventory_skew ** self.inventory_penalty_power) * 0.2
        raw_reward = pnl_norm + spread_norm + self.survival_bonus - inventory_penalty - impact_norm

        current_equity = self._equity(s.cash, s.inventory, next_mid)
        s.equity_peak = max(s.equity_peak, current_equity)
        drawdown = 0.0
        if s.equity_peak > 0:
            drawdown = max(0.0, (s.equity_peak - current_equity) / (s.equity_peak + 1e-8))

        # Penalize catastrophic behavior harder than normal inventory carrying.
        if drawdown > 0.20:
            raw_reward -= (drawdown - 0.20) * 0.35

        bankrupt = current_equity < (1.0 - self.max_drawdown_terminate) * self._initial_equity
        shaped = self._signed_power(raw_reward, self.alpha_pos, self.alpha_neg) * self.reward_scale
        reward = 0.0 if s.step < self.warmup_steps else float(np.clip(shaped, -2.0, 2.0))
        if bankrupt:
            reward = float(np.clip(reward - 1.5, -2.0, 2.0))

        s.pnl_history.append(total_pnl)
        
        # Update funding rate and OI
        if len(self.funding_rates) > next_cursor:
            s.funding_rate = float(self.funding_rates[next_cursor])
        if len(self.open_interests) > next_cursor:
            s.open_interest = float(self.open_interests[next_cursor])
        
        s.step += 1
        self._cursor = next_cursor

        done = s.step >= s.episode_length or self._cursor >= len(self.prices) - 1 or bankrupt
        info = {
            "inventory": s.inventory,
            "cash": s.cash,
            "mid": mid,
            "pnl": total_pnl,
            "raw_reward": raw_reward,
            "spread_capture": spread_capture,
            "market_impact": market_impact_cost,
            "drawdown": drawdown,
            "equity": current_equity,
            "bankrupt": bankrupt,
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
        }
        return self._get_obs(), reward, done, info


class SyncVectorMarketMakingEnv:
    """Simple synchronous vectorized wrapper for MarketMakingEnv."""

    def __init__(self, env_fns: list[Callable[[], MarketMakingEnv]]):
        if not env_fns:
            raise ValueError("env_fns must not be empty")
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self) -> np.ndarray:
        obs = [env.reset() for env in self.envs]
        return np.stack(obs, axis=0).astype(np.float32)

    def step(self, actions: np.ndarray, discrete: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        next_obs: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[bool] = []
        infos: list[dict] = []
        for idx, env in enumerate(self.envs):
            action = actions[idx]
            obs, rew, done, info = env.step(action, discrete=discrete)
            if done:
                obs = env.reset()
            next_obs.append(obs)
            rewards.append(float(rew))
            dones.append(bool(done))
            infos.append(info)
        return (
            np.stack(next_obs, axis=0).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )


class CurriculumWrapper:
    """
    Curriculum Learning wrapper for MM environment.
    
    Phases:
    - EASY: Trending data (low inventory risk)
    - MEDIUM: Normal/balanced regimes (default)
    - HARD: Chaotic/sideways regimes (high noise)
    
    The wrapper filters or weights episodes to emphasize the target difficulty.
    """

    def __init__(
        self,
        price_series: np.ndarray,
        curriculum_phase: TrainingCurriculum = TrainingCurriculum.MEDIUM,
        volatility_window: int = 20,
    ):
        self.price_series = price_series.astype(np.float32)
        self.curriculum_phase = curriculum_phase
        self.volatility_window = volatility_window
        
        # Compute volatility profile
        self.volatilities = self._compute_volatilities()
        self.valid_start_indices = self._get_valid_indices()

    def _compute_volatilities(self) -> np.ndarray:
        """Compute rolling volatility for all positions."""
        vols = np.zeros(len(self.price_series))
        for i in range(self.volatility_window, len(self.price_series)):
            recent = self.price_series[i - self.volatility_window : i + 1]
            vols[i] = float(np.std(np.diff(recent) / (recent[:-1] + 1e-8)))
        return vols

    def _get_valid_indices(self) -> np.ndarray:
        """
        Get list of valid episode start indices based on curriculum phase.
        
        Returns:
            Array of valid start indices
        """
        q_low = np.percentile(self.volatilities, 33)
        q_high = np.percentile(self.volatilities, 67)
        
        valid = []
        for i in range(self.volatility_window, len(self.price_series) - self.volatility_window - 1):
            vol = self.volatilities[i]
            
            if self.curriculum_phase == TrainingCurriculum.EASY:
                # Low volatility (trending)
                if vol < q_low:
                    valid.append(i)
            elif self.curriculum_phase == TrainingCurriculum.MEDIUM:
                # Any volatility (all regimes)
                valid.append(i)
            else:  # HARD
                # High volatility (chaotic)
                if vol > q_high:
                    valid.append(i)
        
        return np.array(valid, dtype=int)

    def create_env(
        self,
        episode_length: int = 200,
        inventory_lambda: float = 0.1,
        seed: int = 42,
        **env_kwargs
    ) -> MarketMakingEnv:
        """
        Create a MarketMakingEnv configured for the curriculum phase.
        
        Args:
            episode_length: Length of episode
            inventory_lambda: Inventory risk penalty
            seed: Random seed
            **env_kwargs: Additional arguments for MarketMakingEnv
            
        Returns:
            Configured MarketMakingEnv instance
        """
        env = MarketMakingEnv(
            price_series=self.price_series,
            episode_length=episode_length,
            inventory_lambda=inventory_lambda,
            seed=seed,
            curriculum=self.curriculum_phase,
            **env_kwargs
        )
        return env

    def sample_episode_start(self, rng: Optional[np.random.Generator] = None) -> int:
        """
        Sample a valid episode start index for the curriculum phase.
        
        Args:
            rng: Random number generator (default: new RNG with seed 42)
            
        Returns:
            Valid start index
        """
        if rng is None:
            rng = np.random.default_rng(42)
        
        if len(self.valid_start_indices) == 0:
            # Fallback: use any index
            return int(rng.integers(self.volatility_window, len(self.price_series) - 200))
        
        return int(rng.choice(self.valid_start_indices))
