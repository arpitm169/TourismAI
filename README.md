# 🌍 TourismAI — Multi-Agent RAG + Predictive Analytics System

> Final Year Project | Intelligent Tourism Management & Revenue Optimisation.

---

## 📁 Project Structure

```
tourism_rag_system/
├── app.py                  # Main Streamlit dashboard (entry point)
├── data_preprocessing.py   # EDA, cleaning, feature engineering
├── model_training.py       # XGBoost/LightGBM classifier + regressors
├── agents.py               # Multi-agent system (LangGraph/LangChain)
├── rag_pipeline.py         # ChromaDB vectorstore + RAG retrieval
├── visualizations.py       # All Plotly chart builders
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── data/                   # Place your tourism_dataset.csv here
├── models/                 # Saved .joblib model files
└── vectorstore/            # ChromaDB persistent storage
```

---

## ⚡ Quick Start

### 1. Clone / set up environment

```bash
# Python 3.10+ required
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API keys (optional but recommended)

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
# The app works without a key in intelligent mock mode
```

### 3. Place your dataset

```bash
cp /path/to/tourism_dataset.csv data/tourism_dataset.csv
```

### 4. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🚀 Usage Walkthrough

| Step | Action |
|------|--------|
| 1 | In the **sidebar**, click **Use Synthetic Dataset** OR upload your CSV |
| 2 | Click **🚀 Train All Models** (sidebar) — takes ~60–90 seconds |
| 3 | Browse **EDA** for charts and statistics |
| 4 | Visit **Models** to review performance gauges |
| 5 | Use **Predict** to run inference on custom inputs |
| 6 | Chat with **Agent Chat** — ask anything about the data |
| 7 | Explore **Recommendations** for destination strategy |

---

## 🤖 ML Model Details

### Classification — `High_Revenue_Potential`

| Component     | Detail |
|---------------|--------|
| Algorithm     | Soft-voting ensemble: XGBoost + LightGBM + RandomForest |
| Imbalance handling | SMOTETomek (combined over+undersampling) |
| Target metric | ≥ 95% Accuracy / ≥ 96% Recall |
| Key features  | Revenue_per_Visitor, Popularity_Score, Rating, Accommodation |

### Regression — Revenue & Visitors

| Target   | Algorithm | Transform |
|----------|-----------|-----------|
| Revenue  | XGBoost + LightGBM voting regressor | log1p target |
| Visitors | XGBoost + LightGBM voting regressor | log1p target |

---

## 🤖 Agent Architecture

```
User Query
    │
    ▼
Supervisor (keyword/LLM routing)
    │
    ├── 📊 Data Analyst Agent     → EDA & statistical insights
    ├── 🔮 Prediction Agent       → Forecasts & model interpretation
    ├── 🎯 Recommendation Agent   → Development strategy
    └── 🌱 Sustainability Agent   → Eco & carrying capacity
         │
         ▼
    RAG Retrieval (ChromaDB)
         │
         ▼
    LLM Response (Google Gemini / mock)
```

---

## 🗃️ Dataset Format

Expected columns:

| Column | Type | Description |
|--------|------|-------------|
| `Location` | string | Destination name |
| `Country` | string | Country name |
| `Category` | string | e.g. Beach, Mountain, Cultural |
| `Visitors` | int | Annual visitor count |
| `Rating` | float | 0.0 – 5.0 |
| `Revenue` | int/float | Annual revenue in USD |
| `Accommodation_Available` | Yes/No or 1/0 | Accommodation flag |

---

## 🐳 Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t tourism-ai .
docker run -p 8501:8501 --env-file .env tourism-ai
```

---

## 📊 Expected Model Performance (on synthetic 1200-row dataset)

| Model | Metric | Expected |
|-------|--------|---------|
| Classifier | Accuracy | ≥ 95% |
| Classifier | Recall | ≥ 96% |
| Classifier | ROC-AUC | ≥ 97% |
| Revenue Reg | R² | ≥ 92% |
| Visitors Reg | R² | ≥ 88% |

*Performance on your real dataset may vary based on data quality and size.*

---

## 🔧 Troubleshooting

**`chromadb` install fails on ARM/Mac**: `pip install chromadb --no-binary :all:`

**`lightgbm` not found**: The system automatically falls back to XGBoost + RandomForest.

**Agents respond in mock mode**: Add your `GOOGLE_API_KEY` to `.env` for full LLM responses.

**Training is slow**: Reduce `n_estimators` in `model_training.py` (e.g., 200 instead of 600).

---

## 📄 License
MIT — free to use for academic and commercial purposes.
