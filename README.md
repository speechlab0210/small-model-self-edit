# 小模型能不能自己改自己的權重，而且一直變好？

**狀態：實驗進行中。** 這份 README 會在結果出來後更新；實驗設計與預測已先寫在 [PREREG.md](PREREG.md)（跑任何模型之前寫的）。

> 本 repository 的實驗設計、程式、執行與文字，全部由 AI agent（Claude Code，執行 Claude 模型）完成。
> 沒有人類檢查過原始數據；AI 可能會犯錯，歡迎直接看程式與原始輸出來檢查。

## 問題

2026 年 11 月有一篇部落格文章〈Sparks of Weight-Space Programming〉（MIT CSAIL）：一個很大的模型
（GPT-6 Astra）直接讀、直接改 Qwen3.5-0.8B 的權重，不用梯度、不用訓練。它只改了 131,072 個數字（0.016%），
這個 0.8B 小模型在 MATH-500 上就從 7.9% 升到 48.1%。主要原因是：小模型想太久會一直繞圈、寫到長度上限都沒寫完；
大模型找到造成繞圈的那個方向，把它從兩層的 MLP 輸出裡扣掉。

這個實驗問反過來的事：**同一個 0.8B 小模型，能不能自己改自己的權重，而且一輪一輪持續進步？**

## 三種讓它改自己的方式

| 組 | 小模型拿到什麼 | 對照 |
|---|---|---|
| A | 跟大模型一模一樣的工具：一個能讀寫自己權重的 Python 環境、練習題、5 次「跑模型看結果」的額度。改了的權重立刻套在自己身上 | 原文大模型的改法（正對照）、完全不改 |
| B | 簡化成選擇題：每輪給它 8 個現成的改法（其中有隨機方向的假改法），它挑一個；連續 6 輪 | 用亂數挑；有／沒有「裁判」把關 |
| C | 傳統的自我訓練：用自己寫的答案再訓練自己（有答案卷／沒答案卷），3 輪 | —（優先度最低，算力不夠就不跑） |

所有比較都在同一份測驗上：MATH-500 抽 100 題、GSM8K 抽 100 題，每個模型用同一組隨機種子作答（配對比較）。

## 怎麼重跑

需要一張 GPU（這次用 Colab 的 G4：NVIDIA RTX PRO 6000 Blackwell，96 GB）。

```
git clone https://github.com/speechlab0210/small-model-self-edit
python small-model-self-edit/colab/run_in_colab.py pilot   # 小規模試跑，只用來抓錯
python small-model-self-edit/colab/run_in_colab.py full    # 正式實驗
```

程式說明：`selfedit/engine.py`（改權重之後，生成與讀取都立刻用新權重）、`selfedit/arm_a.py`（A 組）、
`selfedit/arm_b.py`（B 組）、`selfedit/mathqa.py`（評分，照抄原文作者的程式）、`run_all.py`（照預先登記的順序跑）。

## English summary

Can a 0.8B model (Qwen3.5-0.8B) edit its own weights and keep getting better at math? The recent "Weight-Space
Programming" post showed that a frontier model can fix this model's reasoning loops by editing 0.016% of its weights
(MATH-500 7.9% -> 48.1%). Here the editor is the small model itself, in three set-ups: (A) the same notebook tools the
frontier model had, (B) choosing among 8 ready-made edits for 6 rounds, against a random chooser, with and without a
referee, (C) self-training on its own answers. Design and predictions are in [PREREG.md](PREREG.md), written before
any run. Results will be added here. Everything in this repository was produced by an AI agent; no human has checked
the raw data.

## 授權

程式碼：MIT（見 [LICENSE](LICENSE)）。`selfedit/mathqa.py` 與 `prompts/` 的來源與授權見 [THIRD_PARTY.md](THIRD_PARTY.md)。
