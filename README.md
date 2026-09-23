# Medical Literature RAG

一個檢索增強生成(RAG)系統,用來查詢慢性腎臟病(CKD)與腎臟移植相關的 PubMed 醫學文獻,並用系統化的消融實驗(Ablation Study)與統計檢定,量化比較不同檢索策略的實際效果。非臨床診斷工具,僅供文獻研究輔助。

---

---

## TL;DR — 這個專案做了什麼

| 環節 | 做了什麼 | 佐證 / 存放位置 |
| --- | --- | --- |
| 資料 | 50 篇 PubMed 論文(CKD / 腎臟移植),結構感知切段:29 篇結構化、21 篇非結構化,共切出 272 段 | `ingestion/`、`preprocessing/`、`data/raw/`、`data/processed/chunks.json` |
| 檢索 | Dense(MedCPT)+ BM25(RRF 融合)+ MedCPT Cross-Encoder Reranker,四組設定(A/B/C/D)可透過參數切換 | `retrieval/*` |
| 生成 | GPT-5-mini,Faithfulness 導向 System Prompt,預設採用統計證實效果最好的 Dense+Reranker 組合 | `generation/answer_generator.py` |
| 評估 | 25 題人工標記評估集(候選論文四組合併取樣、避免偏袒單一組)+ Recall@5/MRR + RAGAS 四指標 | `data/evaluation/eval_set.json`、`retrieval_scores.csv`、`ragas_scores.csv` |
| 統計檢定 | 常態性檢定(Shapiro-Wilk)→ 雙因子重複測量 ANOVA + Friedman/Wilcoxon 交叉驗證 | `evaluation/statistical_analysis.py`、`data/evaluation/statistical_analysis_report.txt` |

**關鍵數字：**

| 指標 | 數值 |
| --- | --- |
| 語料庫規模 | 50 篇論文、272 個切段(結構化 29 篇 / 非結構化 21 篇) |
| 最佳組合(D_dense_rerank)Recall@5 | **0.887** |
| 最佳組合(D_dense_rerank)MRR | **0.813** |
| Reranker 主效應(Recall@5,雙因子 ANOVA) | p = 0.018 ✅ 顯著 |
| 檢索方式主效應(Recall@5,雙因子 ANOVA) | p = 0.163,未顯著 |
| A/B/C/D 整體差異(Recall@5,Friedman) | p = 0.013 ✅ 顯著 |
| A/B/C/D 整體差異(MRR,Friedman) | p = 0.214,未顯著 |
| Reranker 主效應(MRR,雙因子 ANOVA) | p = 0.206,未顯著 |
| RAGAS Faithfulness(A_dense_only 最高) | 0.972 |
| RAGAS Context Recall(D_dense_rerank 最高) | 0.525 |

---

## 為什麼做這個專案

醫學研究者要從大量論文裡找到特定問題的答案,傳統關鍵字搜尋常常抓不到語意上相關、但用字不同的段落;直接丟給通用型 LLM 又容易讓它憑自己的訓練知識回答,而不是真正根據最新文獻——這在醫學領域尤其危險,因為答案可能過時或無法追溯來源。

這個專案的目標,不是「做出一個能回答問題的 AI」,而是做出一個**答案有憑有據、可以追溯回原始論文,而且每一個設計決策都經過量化驗證**的系統。跟坊間常見的 RAG 教學專案不同的地方在於:我沒有停在「做出一個能跑的版本」就結束,而是進一步問「這個版本真的比較好嗎?」——用 25 題人工標記的評估集、Recall@5/MRR、RAGAS 四項生成品質指標,以及雙因子重複測量 ANOVA 搭配無母數檢定,把「感覺比較好」變成「有統計證據支持」。

## Architecture

```
                    使用者問題
                        │
        ┌───────────────┼───────────────┐
        │                               │
   Dense 檢索                      BM25 檢索
 (MedCPT Query Encoder)          (關鍵字比對)
        │                               │
        └───────────────┬───────────────┘
                         │
                RRF 融合(Hybrid)
                         │
              MedCPT Cross-Encoder
                  Reranker
                         │
                    Top-5 段落
                         │
              LLM 生成答案(附 PMID 引用)
                         │
                   Faithfulness 導向
                    System Prompt
```

資料準備階段(離線,執行一次):

```
PubMed E-utilities API
        │  ingestion/pubmed_loader.py(保留段落 Label)
        ▼
   data/raw/*.json
        │  preprocessing/chunking.py(結構感知切段)
        ▼
data/processed/chunks.json
        │  embedding/embedder.py(MedCPT Article Encoder)
        ▼
  FAISS 向量索引 + chunks_meta.json
```

## Tech Stack

| 環節 | 技術選擇 | 選擇理由 |
| --- | --- | --- |
| 資料來源 | PubMed E-utilities API | 免費、官方、可程式化查詢的醫學文獻資料庫 |
| 切段 | 自製結構感知切段器 | 依論文真實的 Background/Methods/Results/Conclusion 段落標籤切,而非固定字數硬切;文獻證實比固定長度切段更少破壞語意完整性 |
| Embedding | MedCPT(`ncbi/MedCPT-Article-Encoder` + `Query-Encoder`) | NCBI 用 2.55 億筆真實 PubMed 查詢-文章配對訓練的雙編碼器,文獻證實在 PubMed 檢索任務上優於通用型模型,且本地免費執行 |
| 向量索引 | FAISS(`IndexFlatIP`) | 精確搜尋,對這個規模的語料庫(50~100 篇)零額外運維成本 |
| 關鍵字檢索 | `rank-bm25`(BM25Okapi) | 補足語意搜尋在專有名詞、藥物代號上的弱點 |
| 檢索融合 | Reciprocal Rank Fusion(RRF) | 不需要正規化分數尺度,融合 Dense 與 BM25 兩種排序的標準做法 |
| Reranker | MedCPT Cross-Encoder | 與 Embedding 同樣走醫學專用路線,對候選段落重新精算相關程度 |
| 生成 | OpenAI(`gpt-5-mini`) | 成本效益佳,搭配嚴格的 Faithfulness 導向 System Prompt |
| 生成面評估 | 自製 RAGAS 四指標實作 | 不依賴第三方套件版本更新,直接對應可解釋的評估邏輯 |
| 統計分析 | `pingouin` + `pandas` | 雙因子重複測量 ANOVA、Friedman、Wilcoxon 一致的 API |

## 安裝與快速上手

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
# 打開 .env,填入 NCBI_EMAIL(PubMed API 要求)與 OPENAI_API_KEY(生成/評估用)

# 4. 建立資料與索引
python ingestion/pubmed_loader.py --query "chronic kidney disease AND kidney transplantation" --max_results 50
python preprocessing/chunking.py
python embedding/embedder.py

# 5. 問一個問題(預設用 Dense + Reranker,本專案統計檢定證實效果最好的組合)
python generation/answer_generator.py \
  --query "What are the risk factors for mortality after kidney transplantation?" \
  --top_k 5
```

完整的消融實驗與統計檢定流程(建立評估集 → 算 Recall@5/MRR → RAGAS 四指標 → 統計檢定),見 `evaluation/` 資料夾內各腳本的說明。

## 評估結果 / 消融實驗發現

四組設定的 2×2 因子設計:Dense / Hybrid 檢索 ×開 / 關 Reranker。25 題人工標記評估集,候選論文從四組合併取樣以避免偏袒任一組。

| Config | Recall@5 | MRR |
| --- | --- | --- |
| A_dense_only | 0.760 | 0.750 |
| B_hybrid | 0.753 | 0.723 |
| C_hybrid_rerank | 0.807 | 0.767 |
| **D_dense_rerank** | **0.887** | **0.813** |

**統計檢定結果(雙因子重複測量 ANOVA + Friedman/Wilcoxon 交叉驗證)：**

四組的 Recall@5、MRR 分數經 Shapiro-Wilk 檢定,均顯著偏離常態分佈,因此同時採用母數(ANOVA)與無母數(Friedman/Wilcoxon)兩條路線交叉驗證,而非只依賴其中一種。

-  **確定成立**:A/B/C/D 四組在 **Recall@5** 上整體存在顯著差異(Friedman p = 0.013);**Reranker 對 Recall@5 有顯著的正面效果,且不受檢索方式影響**(ANOVA 主效應 p = 0.018)——這是整組實驗裡最站得住腳的結論。個別配對比較中,A vs D(Dense 底下的 Reranker 效果)p = 0.0579,最接近顯著但未跨過門檻
-  **數字有差異、統計未達顯著**:Dense 平均 Recall@5 高於 Hybrid,但 ANOVA 主效應 p = 0.163,25 題樣本量下尚不能排除抽樣雜訊;交互作用同樣不顯著(p = 0.240)
-  **MRR 上沒有觀察到同樣的顯著性**:同一套四種檢定方法(常態性檢定、ANOVA、Friedman、Wilcoxon)套用在 MRR 上,**全部未達顯著**(Friedman p = 0.214;ANOVA 的 Reranker 主效應 p = 0.206)。Recall@5 與 MRR 這兩個指標的結論不完全一致,不是檢定出錯,而是提醒:同一組資料換一個評估角度,顯著性結論可能不同,不宜只憑單一指標下定論

完整的逐項檢定結果(含全部四組配對比較的 p 值),見 `data/evaluation/statistical_analysis_report.txt`。

RAGAS 生成面評估(四組平均):

| Config | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| --- | --- | --- | --- | --- |
| A_dense_only | 0.972 | 0.782 | 0.875 | 0.434 |
| B_hybrid | 0.966 | 0.782 | 0.865 | 0.419 |
| C_hybrid_rerank | 0.956 | 0.788 | 0.893 | 0.405 |
| D_dense_rerank | 0.958 | 0.778 | **0.906** | **0.525** |

## 常見問題 / 使用限制

**這是臨床診斷工具嗎?** 不是。這是文獻研究輔助工具,所有答案都應搭配原始論文核實,不能取代醫療專業判斷。

**為什麼有些常見問題(如 AKI 預測因子、病患服藥遵從性)在評估集裡找不到?** 這些主題在建立評估集的過程中,經過候選論文合併取樣(pooling)後確認語料庫(50 篇)完全沒有覆蓋,對應問題已從評估集排除,而非強行標記錯誤答案。這是語料庫規模的限制,不是系統缺陷。

**檢索結果會不會被同一篇論文佔滿?** 目前偶爾會發生——MMR(Maximal Marginal Relevance)這類強制結果多樣性的機制尚未實作,是已知限制。

**Dense 真的比 Hybrid 好嗎?** 目前數據顯示有差異但統計上未達顯著,不建議直接下定論。

**評估集的候選論文怎麼保證公平?** 候選論文以 pooling 方式從 A/B/C/D 四組合併取樣。但若某篇論文四組都排不進候選池,仍無法被標記為相關——這是所有基於「檢索池」建立標準答案的方法(包含 TREC 這類大型評測)共同的天花板。

**有做 Query Fusion(如 RAG-Fusion、HyDE)嗎?** 沒有。評估後認為對目前語料庫規模(50~100 篇)的邊際效益有限,故未實作。

## Project Structure

```
medical-literature-rag/
├── ingestion/          # PubMed 資料蒐集
├── preprocessing/       # 結構感知切段
├── embedding/            # MedCPT 向量化
├── retrieval/            # Dense / BM25 / Hybrid / Reranker
├── generation/            # LLM 生成答案
├── evaluation/            # 評估集建立、Recall/MRR、RAGAS、統計檢定
├── data/
│   ├── raw/               # 原始論文
│   ├── processed/          # 切段後資料、FAISS 索引
│   └── evaluation/          # eval_set.json、逐題分數、RAGAS 分數、統計報告
└── app/, experiments/, tests/   # 未來擴充用
```

## 授權與資料來源

程式碼採 MIT 授權。資料來自 PubMed(NCBI E-utilities)公開 Metadata(標題、摘要),使用時請遵守 [NCBI 使用條款](https://www.ncbi.nlm.nih.gov/home/about/policies/)。若要抓全文,需另外確認 PMC Open Access 子集的授權範圍。
