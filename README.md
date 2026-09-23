# Medical Literature RAG Assistant

基於醫學文獻(CKD / Kidney Transplant)的檢索增強生成系統,並以系統化的消融實驗(Ablation Study)與統計檢定,比較不同檢索策略的實際效果。
非臨床診斷工具,僅供文獻研究輔助。

## 專案完成度

- [x] Phase 1 — 資料蒐集(PubMed E-utilities API,50 篇 CKD / 腎臟移植文獻,保留段落結構)
- [x] Phase 2 — 結構感知切段(依真實的 Background/Methods/Results/Conclusion 段落標籤切,而非猜測)
- [x] Phase 3 — MedCPT Embedding(醫學專用雙編碼器)+ FAISS 向量索引
- [x] Phase 4 — Hybrid 檢索(Dense + BM25,RRF 融合)+ MedCPT Cross-Encoder Reranker
- [x] Phase 5 — 評估:25 題人工標記評估集、Recall@5/MRR、RAGAS 四指標、雙因子重複測量 ANOVA + Friedman/Wilcoxon 統計檢定
- [ ] Phase 6 — Streamlit Demo(規劃中,未實作)

## 系統架構

```
PubMed API
    │  ingestion/pubmed_loader.py(保留段落 Label)
    ▼
data/raw/*.json ──▶ preprocessing/chunking.py(結構感知切段)
    │
    ▼
data/processed/chunks.json ──▶ embedding/embedder.py(MedCPT Article Encoder)
    │
    ▼
FAISS 向量索引 + chunks_meta.json
    │
    ├──▶ retrieval/dense_retriever.py(MedCPT Query Encoder)
    ├──▶ retrieval/bm25_retriever.py(關鍵字比對)
    ├──▶ retrieval/hybrid_retriever.py(RRF 融合 Dense + BM25)
    └──▶ retrieval/reranker.py(MedCPT Cross-Encoder)
              │
              ├──▶ generation/answer_generator.py(OpenAI,附 PMID 引用,只根據檢索段落回答)
              │
              └──▶ evaluation/(見下方「消融實驗與評估」)
```

## 快速開始

```bash
# 1. 建立虛擬環境
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. 安裝套件
pip install -r requirements.txt
# Windows 若安裝 torch 時出現 WinError 126 / shm.dll 相關錯誤,
# 先安裝 Microsoft Visual C++ Redistributable (x64):
# https://aka.ms/vs/17/release/vc_redist.x64.exe,安裝後重開機再重試

# 3. 設定環境變數
cp .env.example .env
# 打開 .env,填入:
#   NCBI_EMAIL     PubMed API 要求填的 email(必填)
#   OPENAI_API_KEY 生成答案 / RAGAS 評估用

# 4. 抓文獻(保留段落結構標籤)
python ingestion/pubmed_loader.py \
  --query "chronic kidney disease AND kidney transplantation" \
  --max_results 50

# 5. 結構感知切段
python preprocessing/chunking.py

# 6. 建立 MedCPT 向量索引
python embedding/embedder.py

# 7. 測試單次問答(檢索 + 生成)
python generation/answer_generator.py \
  --query "What are the risk factors for mortality after kidney transplantation?" \
  --top_k 5
```

## 消融實驗與評估

四組設定的 2×2 因子設計:

| | Reranker 關 | Reranker 開 |
| --- | --- | --- |
| **Dense only** | A_dense_only | D_dense_rerank |
| **Hybrid** | B_hybrid | C_hybrid_rerank |

```bash
# 建立 25 題人工標記評估集(候選論文從 A/B/C/D 四組合併取樣,避免偏袒任一組)
python evaluation/build_eval_set.py

# 跑 A/B/C/D 四組的 Recall@5、MRR,並存下逐題原始分數
python evaluation/retrieval_eval.py

# 幫每題生成參考答案全文(RAGAS Context Recall 需要)
python evaluation/build_reference_answers.py

# 跑 RAGAS 四指標:Faithfulness、Answer Relevancy、Context Precision、Context Recall
python evaluation/ragas_eval.py

# 統計顯著性檢定:常態性檢定、雙因子重複測量 ANOVA、Friedman + Wilcoxon
python evaluation/statistical_analysis.py
```

### 主要發現

- **確定成立**:A/B/C/D 四組的 Recall@5 整體存在顯著差異(Friedman p = 0.013);**Reranker 對 Recall@5 有顯著的正面效果,且不受檢索方式(Dense/Hybrid)影響**(雙因子 ANOVA 主效應 p = 0.018)——這是整個實驗裡最站得住腳的結論
- **數字上有差異、但統計上未達顯著**:Dense 平均 Recall@5(0.824)高於 Hybrid 平均(0.780),但 ANOVA 主效應 p = 0.163,25 題樣本量下尚不能排除這只是抽樣雜訊
- **交互作用不顯著**(p = 0.240):換用領域專用的 MedCPT 之後,Reranker 帶來的邊際效益在 Dense/Hybrid 之間是否有差異,目前證據不足以下定論——這點在通用 Embedding 模型(all-MiniLM-L6-v2)與 MedCPT 兩輪實驗中觀察到方向相反的交互作用模式,顯示消融實驗的結論會受底層 Embedding 模型選擇影響,不能直接類推
- **檢索結果多樣性不足**:觀察到同一篇論文的多個段落有時會佔滿 top-5,尚未透過 MMR 等方法解決(見下方限制)

## 專案限制與未來工作

- **MMR(Maximal Marginal Relevance)未實作**:top-5 結果有時被同一篇論文的多個段落佔滿,理論上會壓縮生成答案引用來源的多樣性
- **語料庫規模較小(50 篇)**:部分主題(如 AKI 預測因子、病患服藥遵從性、活體 vs 屍體捐贈比較)在評估集建立過程中確認完全沒有相關文獻覆蓋,對應問題已從評估集中排除,而非強行標記
- **評估集標準答案的候選來源仍有天花板**:候選論文以 pooling 方式從 A/B/C/D 四組合併取樣,避免偏袒任一組,但若某篇論文四組都排不進候選池,仍無法被標記為相關(TREC 式 pooling 方法共通的限制)
- **未做 Query Fusion / HyDE**:規劃階段評估後認為對目前的語料庫規模效益有限,故未實作
- **統計檢定力受限於樣本數**:25 題在小樣本消融實驗中屬合理規模,但部分邊緣顯著的比較(如 p = 0.058)仍需更大樣本才能下定論

## 專案資料夾

```
medical-literature-rag/
├── ingestion/          # Phase 1:PubMed 資料蒐集
├── preprocessing/       # Phase 2:結構感知切段
├── embedding/            # Phase 3:MedCPT 向量化
├── retrieval/            # Phase 3-4:Dense / BM25 / Hybrid / Reranker
├── generation/            # LLM 生成答案
├── evaluation/            # 評估集建立、Recall/MRR、RAGAS、統計檢定
├── data/
│   ├── raw/               # 原始論文
│   ├── processed/          # 切段後資料、FAISS 索引
│   └── evaluation/          # eval_set.json、逐題分數、RAGAS 分數
└── app/, experiments/, tests/   # 未來擴充用(Phase 6 尚未實作)
```

## 授權與資料來源

資料來自 PubMed(NCBI E-utilities)公開 Metadata(標題、摘要),
使用時請遵守 [NCBI 使用條款](https://www.ncbi.nlm.nih.gov/home/about/policies/)。
若要抓全文,需另外確認 PMC Open Access 子集的授權範圍。
