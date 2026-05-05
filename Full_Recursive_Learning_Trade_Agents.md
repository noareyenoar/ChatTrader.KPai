# Full_Recursive_Learning_Trade_Agents (RLTA) v3.0 - Production Master Plan

## 1. Vision and Executive Summary
The RLTA framework transitions static quantitative models into a self-evolving AI organization [cite: 18]. Version 3.0 addresses critical real-world bottlenecks: LLM latency, analysis paralysis, journal drift, and hardware constraints. To ensure profitability and manageability, the architecture relies on strict risk control layers, error decomposition, and a phased rollout strategy.

## 2. Phased Rollout Strategy

To prevent hardware crashes and debug efficiently, the system is divided into two implementation phases.

### Phase 1: Experimental (Minimal Viable System)
*   **Goal:** Test pipeline connectivity, latency, and the ground-truth learning loop.
*   **Agent Roster (2+1+1):** 2 Analysts (Trend vs. Reversion) + 1 Shadow Agent + 1 Portfolio Manager.
*   **Execution:** Test the Fast Path vs. Slow Path logic.
*   **Compute:** Single machine, using highly quantized LLMs (e.g., Llama 3 8B 4-bit GGUF).

### Phase 2: Production (Full Scale)
*   **Goal:** Maximize strategy diversity and full multi-agent debate.
*   **Agent Roster (6+1+1):** 6 Archetype Analysts + 1 Shadow Agent + 1 Portfolio Manager [cite: 18].
*   **Execution:** Full dynamic routing based on market volatility and timeframe.
*   **Compute:** Distributed setup (Separate Inference Machine and Training/Database Machine).

---

## 3. The Dual-Path Execution Engine (Latency Management)

Local LLM debates introduce latency. The system dynamically routes decisions based on the required trading timeframe to prevent slippage.

*   **FAST PATH (Low Latency / Scalping):**
    *   Bypasses the LLM Orchestrator debate.
    *   Directly aggregates raw model signals. If `Signal Confidence > Threshold`, the Portfolio Manager instantly calculates sizing and executes.
*   **SLOW PATH (High Latency / Swing & Position):**
    *   Full debate loop activated.
    *   Analysts present theses, Shadow Agent critiques, and Orchestrator synthesizes.

*   **Anti-Overthinking Rule:** If `Consensus Score < 60%` during the Slow Path debate, the Orchestrator strictly outputs `NO TRADE`. Analysis paralysis is penalized.

---

## 4. Agent Roles & Responsibilities (6+1+1 Structure)

1.  **The Analysts (6 Archetypes):** Generate raw signals and theses based on their specific models (Trend, Reversion, Breakout, etc.) [cite: 18].
2.  **The Shadow Agent (Devil's Advocate):** Systematically attacks the Analysts' logic to prevent groupthink [cite: 18].
3.  **The Portfolio Manager (Risk & Sizing):** The final gatekeeper. Controls exposure, checks correlations, and determines position size. It can veto an Orchestrator decision if risk limits are breached [cite: 18].
4.  **The Orchestrator:** Chairs the debate and makes the final directional call [cite: 18].
5.  **The Regime Detector:** Identifies the market state (Trending, Ranging, etc.) to weight the Analysts' credibility [cite: 18].
6.  **The Journaler:** Manages the Vector DB memory layer [cite: 18].

---

## 5. Recursive Learning & Combating Journal Drift

Agents will naturally drift toward reinforcing their own biases if memory is not strictly anchored to reality.

### 5.1 Ground Truth & Error Decomposition
Every closed trade must undergo decomposition [cite: 18]:
$Loss = Signal\_Error + Decision\_Error + Execution\_Error$

### 5.2 Strict Penalty System
*   **If Model Wrong (Signal Error):** Reduce the specific Analyst's base weight for that market regime.
*   **If Reasoning Wrong (Decision Error):** If the Orchestrator ignored a correct Shadow Agent warning, penalize the Orchestrator's specific logic path in `AGENT_STRATEGY.md` and increase the Shadow Agent's weight.
*   **Journal Integrity:** The Journaler must append the `Actual_Outcome` and `Error_Attribution` to every pre-trade hypothesis [cite: 18]. Agents are forced to read their exact failures before the next debate.

---

## 6. Hardware & Compute Constraints

Running 18 models, LLM debates, and a Vector DB locally requires strict resource separation.

*   **Process Isolation:**
    *   **Node A (Inference & Execution):** Runs the quantized Ollama models, raw signal generation, and the execution API.
    *   **Node B (Memory & Training):** Handles the Vector DB (ChromaDB), heavy neural network retraining, and batch post-trade error decomposition.
*   **LLM Optimization:** Use GGUF/AWQ quantized models to ensure the Orchestrator can generate responses within acceptable timeframes.

---

## 7. Acceptance Criteria for Production Readiness
1.  **Latency:** Fast Path executes in < 200ms. Slow Path completes debate in < 5 seconds.
2.  **Discipline:** The system successfully outputs `NO TRADE` in sideways, low-consensus markets.
3.  **Adaptability:** The Credibility Scoring visibly shifts power between agents when the Regime Detector flags a market change (e.g., from Trend to Range) [cite: 18].

---
*Framework Version: 3.0 (Production & Control Layer)*
"""