ChatTraderKPai-Fork-Guidelines.md
1. THE ARCHITECTURAL PIVOT
We are repurposing the ChatDev "Software House" concept into a "Quant Trading Firm." Instead of CEO, Coder, and Reviewer, our hierarchy is built on Specialized Alpha Generation and Consensus Reasoning.

Role Mapping
ChatDev Original	ChatTrader.KPai Role	Data/Model Access
CEO	Agent 0: Trading Orchestrator	Full Debate Logs, Risk Metrics, Final PnL
CTO / Coder	The 6 Archetype Traders	3 Specific Trained PyTorch Models per Agent
Reviewer / Tester	Risk Compliance Agent	Drawdown Limits, Position Sizing Rules

2. AGENT DEBATE & EVIDENCE LOGIC
Unlike standard LLM agents that just "talk," every Trader Agent in this system must back their claims with Hard Evidence from their respective model catalogs.

The Inference Hook
Each Trader Agent must have a method to trigger inference:

Load Models: Agent loads its 3 unique .pt models from /models/{archetype}/.

Generate Signals: For the current market window, the agent runs inference.

Construct "Evidence Packet":

Model Confidence: Average probability across the 3 models.

Signal Alignment: Do the LSTM, Transformer, and TCN agree?

Historical Performance: Reference the model_performance_summary.md for its specific models.

The Debate Protocol
The Orchestrator initiates a Reasoning Loop:

Step 1 (Presentation): Each of the 6 agents presents their Bias (Long/Short/Neutral) and their Confidence Score.

Step 2 (Rebuttal): Agents with opposing biases (e.g., Trend vs. Mean Reversion) must "debate" based on current volatility regimes.

Step 3 (Final Judgement): The Orchestrator weighs the arguments. If the Scalper is 90% confident but the Trend Follower sees a regime shift, the Orchestrator decides the final Action and Size.

3. VISUALIZATION & UI REQUIREMENTS
We are extending the existing ChatDev visualization to support Real-Time Strategy Monitoring.

Required UI Components:
The Debate Theater: Keep the ChatDev-style avatar conversation UI, but style it as a trading floor.

Signal Heatmap: A visual grid showing the real-time bias of all 6 archetypes.

Decision Tree: A visualization of how the Orchestrator moved from 6 different opinions to 1 final trade.

Backtest Playback: Integration of matplotlib or plotly charts within the visualization folder to show PnL curves generated during the simulation.

4. OLLAMA INTEGRATION (LOCAL LLM BRAIN)
All agent communication must be handled via the Ollama API to ensure privacy and local execution.

Model Selection: Default to Llama3 or Mistral (configured in YAML).

Prompt Engineering: Use the specific system instructions defined in /coder_agent_system_prompts/generative-ai/.

Stability: Implement a retry mechanism. If Ollama times out or produces non-JSON output, the agent must self-correct based on the debugging-quality prompts.

5. CODER AGENT IMPLEMENTATION STEPS (TODO)
Fork Environment: Copy the ChatDev core logic into /ChatTrader.KPai/.

Custom Agent Classes: Create a new TraderAgent class that inherits from ChatDev's BaseAgent but adds model_inference capabilities.

Environment Bridge: Build a MarketEnv class that feeds the same slice of Binance data to all agents simultaneously.

Modified Phase Handler: Rewrite the ChatDev WareHouse or Phase logic to support the "Debate-to-Trade" workflow instead of "Code-to-Software."

Validation Script: Create test_debate_flow.py to ensure Agent A can read Model A and send a valid JSON signal to the Orchestrator.

⚠️ CRITICAL FAILURE MODES TO AVOID
Echo Chambers: Ensure agents don't just agree with each other. Use the "Debate with Reasoning" prompt to force agents to look for flaws in others' arguments.

Latency Creep: If 7 agents talk to Ollama sequentially, the system will be too slow. Implement Parallel LLM Calls where possible.

Data Leakage: Ensure agents only see the data up to the current timestamp of the simulation. No "looking into the future" during the debate.

STOP: Coder Agent, verify you have read the models_by_archtype_catalog.md before attempting to write the TraderAgent inference code. Report your understanding of the "Evidence Packet" before proceeding.