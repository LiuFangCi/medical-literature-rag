# Medical Literature RAG Assistant

基於醫學文獻(CKD / Kidney Transplant)的檢索增強生成與答案品質評估系統。
非臨床診斷工具,僅供文獻研究輔助。

## 目前進度

- [x] Phase 1 — 資料蒐集(PubMed 文獻)
- [x] Phase 2 — 文字清理、Chunking
- [x] Phase 3a — Embedding + FAISS 語意檢索
- [x] Phase 3b — LLM 生成答案(OpenAI,含來源引用)
- [ ] Phase 4 — Hybrid Search + Reranker
- [ ] Phase 5 — Evaluation(Recall@k、MRR、Faithfulness)
- [ ] Phase 6 — Streamlit Demo

目前系統已能做到:輸入一個研究問題 → 檢索最相關的文獻段落 → 由 LLM 根據段落生成有引用來源(PMID)的完整答案,且回答會忠實於檢索到的內容,不會使用檢索範圍外的資訊回答。

### 快速開始

```bash
# 1. 建立虛擬環境
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. 安裝套件
pip install -r requirements.txt
# Windows 如果安裝 torch 時出現 WinError 126 / shm.dll 相關錯誤,
# 先安裝 Microsoft Visual C++ Redistributable (x64):
# https://aka.ms/vs/17/release/vc_redist.x64.exe,安裝後重開機再重試

# 3. 設定環境變數
cp .env.example .env
# 打開 .env,填入:
#   NCBI_EMAIL     PubMed API 要求填的 email(必填)
#   OPENAI_API_KEY 生成答案用(Phase 3b 才需要)

# 4. 抓一批文獻(Phase 1)
python ingestion/pubmed_loader.py \
  --query "chronic kidney disease AND kidney transplantation" \
  --max_results 50

# 5. 清理文字、切成小段(Phase 2)
python preprocessing/chunking.py

# 6. 建立向量索引(Phase 3a,本地跑,不需要 API Key)
python embedding/embedder.py

# 7a. 純測試檢索(可選,不呼叫 LLM,不花錢)
python retrieval/dense_retriever.py \
  --query "What are the risk factors for mortality after kidney transplantation?" \
  --top_k 5

# 7b. 檢索 + 生成完整答案(Phase 3b,會呼叫 OpenAI API)
python generation/answer_generator.py \
  --query "What are the risk factors for mortality after kidney transplantation?" \
  --top_k 5
```

### 資料流程

```
PubMed API (ingestion/pubmed_loader.py)
        ↓  data/raw/*.json
文字清理 + Chunking (preprocessing/chunking.py)
        ↓  data/processed/chunks.json
Embedding + FAISS 索引 (embedding/embedder.py)
        ↓  data/processed/faiss.index, chunks_meta.json
語意檢索 (retrieval/dense_retriever.py)
        ↓  Top-k 相關段落
LLM 生成答案 (generation/answer_generator.py)
        ↓  含 PMID 引用的完整答案
```

## 後續 Phase(規劃中)

| Phase | 內容 | 產出資料夾 | 狀態 |
|---|---|---|---|
| 4 | BM25 + Hybrid Search + Reranker | `retrieval/` | 規劃中 |
| 5 | Evaluation Dataset、Recall@k、MRR、Faithfulness | `evaluation/` | 規劃中 |
| 6 | Streamlit Demo | `app/` | 規劃中 |

## 授權與資料來源

資料來自 PubMed(NCBI E-utilities)公開 Metadata(標題、摘要),
使用時請遵守 [NCBI 使用條款](https://www.ncbi.nlm.nih.gov/home/about/policies/)。
若要抓全文,需另外確認 PMC Open Access 子集的授權範圍。
