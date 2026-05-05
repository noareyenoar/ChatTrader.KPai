🧠 1. Trend Follower (สายตามเทรนด์)
🔹 Premise
“ราคาเคลื่อนที่เป็นแนวโน้ม (trend persists)”
สิ่งสำคัญ: momentum > mean
🔹 Hypothesis
ถ้าราคาเริ่มขึ้น → มีโอกาสขึ้นต่อ
ใช้ concept: autocorrelation / regime persistence
🔹 Tools
Moving Average (SMA / EMA)
MACD
Ichimoku Cloud
Breakout (Donchian channel)
Volume confirmation
🔹 Positioning
Long เมื่อ breakout / trend confirm
Hold นาน (ride trend)
Cut loss สั้น
🔹 NN Approach
Model: LSTM / Transformer (sequence-based)
Input:
price sequence
moving averages
volatility
Label:
future return (multi-horizon)

👉 Key:

model ต้อง detect “regime” ว่าตลาด trending หรือไม่

⚖️ 2. Mean Reversion Trader (สายเด้งกลับค่าเฉลี่ย)
🔹 Premise
“ราคาจะกลับสู่ค่าเฉลี่ย (mean reverting)”
ตลาด overreact เสมอ
🔹 Hypothesis
ราคาที่ deviation สูง → มีโอกาส revert
🔹 Tools
Bollinger Bands
RSI
Z-score
VWAP deviation
🔹 Positioning
Short เมื่อ overbought
Long เมื่อ oversold
TP ใกล้ mean
🔹 NN Approach
Model: MLP / shallow NN / LightGBM + NN hybrid
Input:
z-score
distance from mean
volatility regime
Label:
short-term reversal (1–10 bars)

👉 Key:

ต้อง normalize ดีมาก (stationary features)

⚡ 3. Scalper / Microstructure Trader
🔹 Premise
“ตลาดมี inefficiency ในระดับ microsecond–second”
edge มาจาก order flow
🔹 Hypothesis
order imbalance → short-term move
🔹 Tools
Order Book (Level 2)
Tape reading
Latency advantage
Spread analysis
🔹 Positioning
เข้าเร็ว ออกเร็ว
กำไรเล็ก แต่บ่อย
🔹 NN Approach
Model:
CNN (order book image)
Transformer (event stream)
Input:
bid/ask depth
trade flow
Label:
next tick direction

👉 Key:

latency สำคัญกว่า accuracy ในบางกรณี

🧩 4. Statistical Arbitrage (Quant / Pair Trading)
🔹 Premise
“สินทรัพย์บางตัวมีความสัมพันธ์กัน”
mispricing = opportunity
🔹 Hypothesis
spread จะ revert (cointegration)
🔹 Tools
Cointegration test
PCA
Spread modeling
🔹 Positioning
Long asset A / Short asset B
🔹 NN Approach
Model:
Autoencoder
Graph Neural Network
Input:
multi-asset correlation
Label:
spread convergence

👉 Key:

ไม่ได้ predict price → predict “relationship”

🧠 5. Discretionary Trader (สายมนุษย์ล้วน + pattern)
🔹 Premise
ตลาดมี behavior + psychology
pattern repeat แต่ไม่ deterministic
🔹 Hypothesis
price action + context สำคัญกว่า indicator
🔹 Tools
Price action
Support / Resistance
Liquidity zones
Narrative / News
🔹 Positioning
flexible
context-based
🔹 NN Approach
Model:
Vision Transformer (chart image)
Multimodal (price + news sentiment)
Input:
chart image
NLP sentiment
Label:
discretionary decision (imitate human)

👉 Key:

imitation learning (learn from trader behavior)

🏗️ 6. Market Maker
🔹 Premise
“กำไรจาก spread ไม่ใช่ direction”
🔹 Hypothesis
ราคาจะแกว่งรอบ fair value
🔹 Tools
Order book
Inventory model
Volatility estimation
🔹 Positioning
วาง bid/ask ตลอดเวลา
hedge inventory
🔹 NN Approach
Model:
Reinforcement Learning
Input:
inventory
order flow
Reward:
PnL + risk penalty

👉 Key:

problem นี้ = control system ไม่ใช่ prediction