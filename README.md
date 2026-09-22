# Medical Literature RAG Assistant

基於醫學文獻(CKD / Kidney Transplant)的檢索增強生成與答案品質評估系統。
非臨床診斷工具,僅供文獻研究輔助。

## 目前進度:Phase 1 — 資料蒐集

### 快速開始

```bash
# 1. 建立虛擬環境
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. 安裝套件
pip install -r requirements.txt

# 3. 設定環境變數
cp .env.example .env
# 打開 .env,填入你的 email(NCBI 要求)與 API Key(選填)

# 4. 抓第一批文獻
python ingestion/pubmed_loader.py \
  --query "chronic kidney disease AND kidney transplantation" \
  --max_results 50
```

執行成功後,`data/raw/` 底下會多一個 JSON 檔,每一筆記錄長得像:

```json
{
  "pmid": "12345678",
  "title": "...",
  "abstract": "...",
  "year": "2023",
  "authors": ["..."],
  "journal": "...",
  "source": "pubmed"
}
```

終端機最後也會印出第一筆記錄的 PMID / 標題 / 摘要前 200 字,
這就是 Phase 1 的里程碑檢查點。

## 後續 Phase(規劃中)

| Phase | 內容 | 產出資料夾 |
|---|---|---|
| 2 | 文字清理、Chunking | `preprocessing/` |
| 3 | Embedding + FAISS,做出 Naive RAG | `embedding/`, `retrieval/`, `generation/` |
| 4 | BM25 + Hybrid Search + Reranker | `retrieval/` |
| 5 | Evaluation Dataset、Recall@k、MRR、Faithfulness | `evaluation/` |
| 6 | Streamlit Demo | `app/` |

## 授權與資料來源

資料來自 PubMed(NCBI E-utilities)公開 Metadata(標題、摘要),
使用時請遵守 [NCBI 使用條款](https://www.ncbi.nlm.nih.gov/home/about/policies/)。
若要抓全文,需另外確認 PMC Open Access 子集的授權範圍。
