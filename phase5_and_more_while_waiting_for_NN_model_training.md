# Phase 5+: Orchestration, Journaling & Simulation Engine (NN Mock Mode)

**Objective:** Develop the complete Multi-Agent framework, LLM Orchestration, and Research Journaling systems while Phase 4 Neural Network models are still training. All NN inferences will use deterministic Mock Models until integration.

**Global Condition:** Do not write any actual PyTorch training code here. Assume `TrendModelInterface.predict_with_confidence()` and similar archetype interfaces return mock tensors.

---

## TODO 1: Implement Multi-Agent Framework Core & Mock NN Interfaces
**Prompt Persona:** `Act as a Senior Software Architect` (Expert in System Design and SOLID principles)
*   **Task 1.1:** Create `MockTraderAgent` classes for all 6 archetypes (Trend, Mean Reversion, Scalper, Stat Arb, Discretionary, Market Maker)[cite: 30].
*   **Task 1.2:** Design the `Evidence Packet` data structure. It must encapsulate the mock NN confidence, directional alignment, and historical credibility score[cite: 31].
*   **Task 1.3:** Build the fast-path vs slow-path routing logic. Ensure that if 'Signal Confidence > Threshold', it bypasses the debate (Fast Path)[cite: 30].

---

## TODO 2: LLM Debate Orchestrator & Ollama Integration
**Prompt Persona:** `Act as a Senior Python/Backend Developer` (Expert in AsyncIO, LLM API integration, and concurrent processing)
*   **Task 2.1:** Implement the `Orchestrator` debate loop. It must receive evidence packets, allow Analysts to present theses, and have the `Shadow Agent` systematically critique them[cite: 30].
*   **Task 2.2:** Connect the loop to local Ollama. Enable parallel LLM calls to strictly control and reduce latency[cite: 31].
*   **Task 2.3:** Implement retry and self-correct logic for any malformed Ollama JSON outputs or logic breakdowns[cite: 31].
*   **Task 2.4:** Implement the "Anti-Overthinking Rule". If consensus is below 60%, the Orchestrator must immediately output `NO TRADE`[cite: 30].

---

## TODO 3: Research Logger & Vector DB (Journaling System)
**Prompt Persona:** `Act as a Data Engineer` (Expert in ChromaDB, Vector Databases, and Data Pipelines)
*   **Task 3.1:** Setup the Vector DB memory layer (ChromaDB) for the `Journaler` agent. This must run on an isolated node (Node B) from the execution logic[cite: 30].
*   **Task 3.2:** Build the Research Data Logger. Every thought, debate transcript, Shadow Agent critique, and final sizing decision must be serialized into a structured JSON/Markdown format specifically designed for writing future academic research papers.
*   **Task 3.3:** Implement the ground-truth anchoring. The Journaler must automatically append the `Actual_Outcome` and `Error_Attribution` to pre-trade hypotheses after a trade closes[cite: 30]. 

---

## TODO 4: End-to-End Simulation & Risk Evaluation Engine
**Prompt Persona:** `Act as a Senior QA/Simulation Engineer` (Expert in Backtesting, Event-driven systems, and Financial Risk Modeling)
*   **Task 4.1:** Build the E2E simulation runner pipeline: Data -> Features -> Mock Inference -> LLM Debate -> Trade Execution[cite: 31].
*   **Task 4.2:** Integrate realistic slippage, exchange fees, and debate latency delays into the backtest loop[cite: 31].
*   **Task 4.3:** Implement the `Portfolio Manager` gatekeeper logic. It must check correlations, determine position size, and have the authority to veto the Orchestrator if risk limits (Max Drawdown, exposure) are breached[cite: 30].
*   **Task 4.4:** Generate the final run report computing Sharpe, profit factor, max drawdown, and regime breakdowns[cite: 31]. Ensure all logs are routed to the Research Logger.

---
**Next Step after completion:** Await NN Model Checkpoints (Phase 4) and swap `MockTraderAgent` with production `TraderAgent`[cite: 31].