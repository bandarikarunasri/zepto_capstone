\# Zepto Data \& AI Platform - End-to-End Capstone



One single repo with 3 linked modules: Data Pipeline, Analytics Pipeline, GenAI Support Assistant. Total Marks: 100



\## Project Structure

/zepto\_capstone

\- /data\_pipeline -> Module 1 (25 marks)

\- /analytics -> Module 2 (50 marks)

\- /support\_assistant -> Module 3 (25 marks)

\- README.md



\## Setup

pip install -r data\_pipeline/requirements.txt

pip install -r analytics/requirements.txt

pip install -r support\_assistant/requirements.txt



\## Module 1: /data\_pipeline - Data Engineering



\*\*Run:\*\*

cd data\_pipeline

python scrape\_and\_load.py



\*\*What it does:\*\*

\- Scrapes 60+ books from books.toscrape.com across 3+ categories (title, price\_gbp, rating text, availability, category)

\- Cleaning: price\_gbp float, rating int 1-5 (One=1 to Five=5), in\_stock bool, median-imputation for failed parses

\- Conversion: price\_inr = price\_gbp \* 105.50 (Fixed project rate: 1 GBP = 105.50 INR)

\- Normalized SQLite: categories(category\_id PK, category\_name UNIQUE), books(book\_id PK, title, price\_gbp, price\_inr, rating, in\_stock, category\_id FK)

\- 5+ SQL queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, JOIN

\- pd.read\_sql() and pd.merge() produce same JOIN result



\*\*Design Decisions:\*\* requests+BeautifulSoup, fixed INR conversion as graded baseline, 2-table PK/FK design.



\## Module 2: /analytics - EDA \& Modeling



\*\*Run:\*\*

cd analytics

jupyter notebook 01\_eda.ipynb

jupyter notebook 02\_modeling.ipynb



\*\*Flow:\*\*

\- sns.load\_dataset('titanic') loaded ONCE, saved as titanic.csv as offline fallback. All steps use pd.read\_csv('titanic.csv')

\- Profiling: info(), describe(), shape, missing % per column

\- Cleaning: <5% missing -> drop rows, 5-30% -> impute median/mode, >30% deck column -> drop with justification

\- Univariate: hist+box for age/fare, IQR outlier counts, mean/median/mode for fare = right-skewed

\- Bivariate: survival rate by sex, pclass, sex\&pclass with boolean masking. Correlation on 6 cols only: survived, pclass, age, sibsp, parch, fare (exclude adult\_male, alone). Heatmap + top 2 abs correlations interpreted.

\- Multivariate: 4+ charts each with 2-4 sentence interpretation

\- Standardization check: z = (x-mean)/std for age/fare, before/after mean 0 std 1 proof

\- Modeling: Stratified split (class imbalance), ColumnTransformer + Pipeline fit only on train, Logistic Regression, Decision Tree with plot\_tree, Random Forest

\- Metrics: Confusion matrix, accuracy, precision, recall, F1, ROC/AUC side-by-side table

\- Imbalance: baseline vs class\_weight=balanced vs SMOTE on train only + conclusion

\- Tuning: GridSearchCV on n\_estimators, max\_depth, max\_features, RandomForestClassifier(oob\_score=True) report best params + OOB score

\- Regression: Predict fare, MAE, RMSE, R2, Adjusted R2, residual plot + heteroscedasticity check

\- Save: joblib.dump(full\_pipeline) -> full\_pipeline.joblib usable on raw data



\## Module 3: /support\_assistant - RAG Service



\*\*Run:\*\*

cd support\_assistant

uvicorn main:app --reload --port 8000

Test: curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\\"query\\": \\"What is your return policy?\\"}"



Docker:

docker build -t zepto-assistant .

docker run -p 7860:7860 zepto-assistant



\*\*Architecture Ingestion->Embedding->Retrieval->Generation:\*\*

\- Ingestion: 8 docs from /docs, simple chunking

\- Embedding: sentence-transformers all-MiniLM-L6-v2 local, no API

\- Retrieval: ChromaDB collection, top-3 cosine similarity, always runs real in both modes

\- Generation: LangGraph StateGraph TypedDict + 3 nodes:

&#x20;   - classify\_intent: keyword heuristic (delivery, return, refund, membership, tracking, cancel, gift card, support hours) -> policy\_question else general\_question. Mock=heuristic, MOCK\_LLM=0=LLM call

&#x20;   - retrieve\_and\_answer: policy\_question -> retrieve top-3, mock answer = "Based on the retrieved context: {top\_chunk\_snippet}", real = LLM with structured prompt (role-context-task-format-length + negative constraint + few-shot)

&#x20;   - direct\_answer: general\_question -> fixed canned "I can only answer questions about Zepto policies right now.", real = LLM direct

\- Edge: classify\_intent -> retrieve\_and\_answer / direct\_answer

\- Output: Pydantic JSON {answer, sources, confidence}. Mock deterministic (sources=retrieved ids, confidence=1.0). Real path retry 2x on validation fail.

\- FastAPI: POST /ask {"query": str} returns validated model



\*\*Mock Mode Example (Graded Baseline MOCK\_LLM=1):\*\*

\- Q: "What is delivery fee?" -> policy\_question -> "Based on the retrieved context: Zepto delivers..." Sources \[doc\_01] Confidence 1.0

\- Q: "Tell me joke" -> general\_question -> "I can only answer questions about Zepto policies right now." Sources \[] Confidence 1.0



\*\*MOCK\_LLM Toggle:\*\* Generation stage branches on MOCK\_LLM env var. Default mock = no network. MOCK\_LLM=0 = optional Groq free tier real LLM.



\## Git Workflow

Branch feature/final-readme created, committed twice, merged to main visible via git log --graph --all



\## No Paid Services

books.toscrape.com free, seaborn+CSV fallback, sentence-transformers local, ChromaDB local, Mock LLM baseline, Docker local

