# 小模型能不能自己改自己的權重，而且一直變好？

**狀態：實驗已完成（2026-11-04）。** 設計與預測在跑任何模型之前就寫在 [PREREG.md](PREREG.md)。
跑的過程中改過的地方，記在 PREREG.md 第 6 節。
每一個數字都由 `analysis/make_report.py` 從原始輸出算出來；完整表格在 [results/RESULTS.md](results/RESULTS.md)。

> 本 repository 的實驗設計、程式、執行與文字，全部由 AI agent（Claude Code，執行 Claude 模型）完成。
> 沒有人類檢查過原始數據；AI 可能會犯錯，歡迎直接看程式與原始輸出來檢查。

## 結論

- **小模型自己寫程式改權重：沒有成功。** A 組拿到跟原文大模型一樣的工具，跑了 6 次、共 180 輪，沒有一個改動真的套上去。
- **讓它從 8 個現成改法裡挑：挑得跟亂數差不多。** 60 次裡有 33 次它回答「1」，也就是清單上的第一個。B 組最好的結果是 +3.5 分，在誤差範圍內。
- **唯一明顯進步的是 C 組：用自己寫完的答案再訓練自己。** 不看答案卷的 C2 三輪都在進步：10.5% → 27.0% → 40.0% → 44.5%。這已經接近原文大模型改的版本（45.5%）。
- **但 C 組的權重是梯度訓練改的，不是小模型決定怎麼改的。** 它學到的主要是「寫完，不要繞圈」：寫完的題目從 22 題變成 124 題，寫完但答錯的也從 2 題變成 35 題。
- 預先登記的成功標準是「小模型自己的選擇讓它進步」。照這個標準，**沒有任何一組達成**。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/test_by_round_dark.png">
  <img src="results/figures/test_by_round_light.png" alt="兩張折線圖，縱軸是 200 題測驗的正確率。左圖 B 組四條線在 6 輪裡都停在 6% 到 14% 之間。右圖 C2 從 10.5% 一路升到 44.5%，接近大模型改的版本 45.5%；C1 先掉到 1.5%，第 3 輪跳到 38%。">
</picture>

## 問題

2026 年 11 月有一篇部落格文章〈Sparks of Weight-Space Programming〉（MIT CSAIL）：一個很大的模型
（GPT-6 Astra）直接讀、直接改 Qwen3.5-0.8B 的權重，不用梯度、不用訓練。它只改了 131,072 個數字（0.016%），
這個 0.8B 小模型在 MATH-500 上就從 7.9% 升到 48.1%。主要原因是：小模型想太久會一直繞圈、寫到長度上限都沒寫完；
大模型找到造成繞圈的那個方向，把它從兩層的 MLP 輸出裡扣掉。

這個實驗問反過來的事：**同一個 0.8B 小模型，能不能自己改自己的權重，而且一輪一輪持續進步？**

## 三種讓它改自己的方式

| 組 | 小模型拿到什麼 | 對照 |
|---|---|---|
| A | 跟大模型一模一樣的工具：一個能讀寫自己權重的 Python 環境、練習題、5 次「跑模型看結果」的額度。改了的權重立刻套在自己身上。30 輪 | 原文大模型的改法（正對照）、完全不改 |
| B | 簡化成選擇題：每輪給它 8 個現成的改法，它挑一個；連續 6 輪 | 用亂數挑；有／沒有「裁判」把關 |
| C | 傳統的自我訓練：用自己寫的答案再訓練自己，3 輪。C1 只留答案卷說對的；C2 不看答案卷，只留自己寫完的 | — |

所有比較都在同一份測驗上：MATH-500 抽 100 題、GSM8K 抽 100 題，共 200 題。
每題最多寫 16,384 個 token。每個模型用同一組隨機種子作答（配對比較）。

## 結果

### 起點與正對照

| 模型 | 200 題正確率 | MATH-500 | GSM8K | 寫到上限被切斷 |
|---|---|---|---|---|
| 原模型（不改） | 10.5% | 8.0% | 13.0% | 89.0% |
| 原文大模型改的版本（正對照） | 45.5% | 32.0% | 59.0% | 47.5% |

- 原模型 200 題裡有 178 題寫到上限都沒寫完，其中 172 題結尾在重複同樣的內容。
- 正對照比原模型多 35.0 分（SE 3.6）。所以這套量法抓得到「改權重讓它進步」。
- 開跑前也先檢查過改權重的機制：改了之後兩個引擎（生成用的 vLLM、讀取用的 transformers）都用新權重，還原之後輸出跟原本一樣。

### A 組：用原文的工具，自己寫程式改自己

6 次（不思考、思考兩種模式，各 3 個隨機種子），每次 30 輪。沒有一次有改動生效，所以測驗成績都等於原模型。

| 這一次 | 發生什麼事 |
|---|---|
| 不思考 s1 | 把規則（不准用梯度訓練）讀成「不准改權重」，結論是做不到；從第 3 輪起每輪都回同一段話 |
| 不思考 s2 | 回答「不能提供改權重的程式」；從第 2 輪起每輪都回同一段話 |
| 不思考 s3 | 30 輪都寫了程式，但每次都去讀一個不存在的張量 `edited`，出錯 |
| 思考 s1 | 30 輪的程式都「沒有出錯」，但程式只定義函式、從來沒有呼叫，所以什麼都沒做 |
| 思考 s2 | 29 輪都用了沒有載入的 `AutoModel`，出錯 |
| 思考 s3 | 26 輪想配置 1 到 24 GB 的記憶體，超過沙盒的記憶體上限而失敗；另外 4 輪去讀不存在的檔案或欄位 |

- 工具本身是好的：開跑前用腳本（不是模型）把每個工具用過一次。存的改動有套到引擎上、凍結的張量會被擋、沙盒讀不到實驗檔案、配置 64 GB 會失敗而且環境還活著。
- 第一次跑思考 s3 時，它的程式把整台機器的記憶體吃光，主程式被系統殺掉。之後加了記憶體上限，從頭重跑這一次。當機那次的紀錄保留在 `results/raw/arm_a_crashed/`。

### B 組：從 8 個現成改法裡挑一個，連續 6 輪

每輪的 8 個改法由固定程序產生，不是模型想的：5 個沿著「繞圈方向」改（從原模型繞圈和寫完的答案估出來），
2 個沿著隨機方向改（假改法，當內建對照），1 個關掉一個注意力頭。改法會一輪一輪累積。

| 組 | 誰來挑 | 有沒有裁判 | 3 輪後（比原模型） | 6 輪後（比原模型） |
|---|---|---|---|---|
| B1 | 它自己 | 有 | +3.5（SE 2.8） | +3.5（第 3 輪之後沒再收任何改動） |
| B2 | 亂數 | 有 | 0（全部被裁判擋下） | 0 |
| B3 | 它自己 | 沒有 | −3.0（SE 2.4） | −4.5（SE 2.3） |
| B4 | 亂數 | 沒有 | −1.0（SE 2.6） | +0.5（SE 2.6） |

- 第 1 輪先把 8 個改法都在同一組 64 題練習題上量過：全部在 −4.7 到 0 分之間。選單裡本來就沒有好選項，所以 B 組沒有機會顯示「挑得好」。
- 它挑的時候，60 次裡有 33 次回答「1」（55%）。選單每次都重新洗牌；如果它看內容挑，回答 1 的機會只有 1/8（12.5%）。所以它主要在挑「第一個」，不是在挑內容。
- 裁判的規則：在 64 題新的練習題上，淨多答對至少 2 題、而且至少一個標準誤，才收下這個改動。B1 只在第 2 輪收了一個（第 17 層，練習題 +4.7 分）。到了 200 題測驗，它是 +3.5 分，z = 1.26，不顯著。

### C 組：用自己的答案再訓練自己，3 輪

每一輪：目前的模型回答 256 題新的練習題 → 留下一部分答案 → 用這些答案訓練一遍（每個答案用一次，每 8 個答案更新一次權重，學習率 1e-5）→ 變成下一輪的模型。

| 輪 | C1 留下的答案（答案卷說對的） | C1 測驗正確率 | C2 留下的答案（自己寫完的，不看答案卷） | C2 測驗正確率 |
|---|---|---|---|---|
| 0 | — | 10.5% | — | 10.5% |
| 1 | 35 | 5.0% | 29 | 27.0% |
| 2 | 22 | 1.5% | 70 | 40.0% |
| 3 | 3 | 38.0% | 129 | 44.5% |

- **C2 每一輪都進步**，三輪後 44.5%，比原模型多 34.0 分（SE 3.6）。
- 進步主要來自「寫完」。寫完的題目：22 → 59 → 109 → 124。一題寫完的答案，長度中位數從約 4,600 個 token 變成約 2,000 個。
- 但寫完而答錯的也變多：2 → 5 → 31 → 35。正對照寫完 105 題，答錯 14 題；C2 寫完得更多，錯得也更多。
- **C1 第 1 輪只比 C2 多留了 6 個答案**：答案卷說它們對，但它們都是寫到上限被切斷的答案（其中 4 個結尾在重複同樣的內容）。結果就從 C2 的 +16.5 分翻成 −5.5 分。之後 C1 越來越常被切斷：89% → 96.5% → 97.5%。
- **C1 第 3 輪只剩 3 個答案，只更新了一次權重，正確率卻從 1.5% 跳到 38.0%。** 這表示這個模型的成績對很小的權重變化非常敏感。單一輪、單一個種子的結果要小心看。
- 跑完後加做一個檢查：把 C 組訓練後的權重放進兩個引擎，讀出來的 log-prob 一樣（最大差 0.124 和 0.125，原模型本身是 0.128）。所以這些分數不是引擎沒載入新權重造成的。訓練改到了 47%（C1）和 64%（C2）的權重。

### 預先登記的預測（PREREG.md 第 5 節，由 `make_report.py` 判定）

| 預測 | 內容 | 結果 | 數字 |
|---|---|---|---|
| P0 | 原模型很差、正對照大幅進步 | 成立 | 原模型 MATH-500 8.0%、切斷 89.0%；正對照 MATH-500 +24.0、GSM8K +46.0 分，切斷少 41.5 點 |
| PA1 | A 組兩種模式，能跑的程式都不到一半輪次 | 成立 | 思考 33.3%（而且那些程式什麼都沒做），不思考 0% |
| PA2 | A 組 6 次都不會顯著進步 | 成立 | 0 次 |
| PA3 | 思考模式至少一半輪次會繞圈到上限、寫不出程式 | 不成立 | 90 輪裡 0 輪。它很快就寫出程式，只是程式是錯的 |
| PB1 | 它挑的不比亂數好 | 成立 | 挑「繞圈方向」類 60%（選單裡佔 62.5%）；挑中的平均 −1.7 分，全部 8 個平均 −1.6 分 |
| PB2 | B1 和 B2 最後差不到 10 分 | 成立 | +3.5 和 0 |
| PB3 | 沒有裁判的 B3、B4 不會進步 5 分以上，而且 B4 會退步 | 不成立 | B3 −4.5；B4 +0.5，沒有退步 |
| PB4 | B1、B2 後 3 輪的進步比前 3 輪小 | 照字面不成立 | B1：+3.5，然後 0。B2：0，然後 0。「0 比 0 小」不成立，所以照寫好的判準算不成立；但兩組後 3 輪都沒有進步，預測想說的「不會持續進步」是對的 |
| PC1 | C1 每一輪都進步 | 不成立 | 10.5 → 5.0 → 1.5 → 38.0 |
| PC2 | C2 也會少切斷，但進步比 C1 少 | 不成立 | C2 44.5% 高於 C1 38.0% |

成功標準只用在 A、B 兩組：小模型自己的選擇讓測驗進步至少 5 分、z ≥ 2，而且比亂數對照好。結果是沒有一組達成。
C 組不在這個標準裡：它沒有亂數對照，改權重的也是梯度訓練。

### 探索性分析（沒有預先登記，跑完才加的）

為什麼 B 組的現成改法沒有用？（`analysis/exploratory_direction.py`）

- B 組估出來的「繞圈方向」跟大模型改的方向相當接近。在第 15 層和第 19 層，兩個方向的 |cos| 是 0.70 和 0.73；在所有層裡，都是同一層最像。
- 照大模型的做法（同樣兩層、同樣只改 64 欄、同樣強度），但換成 B 組的方向：64 題練習題 +4.7 分（SE 4.7）。大模型自己的改法在同一組練習題是 +10.9 分（SE 5.9）。
- 這組練習題當量尺很弱：大模型的改法在 200 題測驗上 +35 分，在練習題上只看得到 +10.9 分，誤差又大。練習題每題只給 8,192 個 token，是測驗的一半，題數也只有 64 題。

## 限制

1. B 組的選單在第 1 輪就沒有好選項（8 個都 ≤ 0 分），所以 B 組只能說明「它沒有挑對」，不能說明「有好選項時它也挑不對」。
2. 練習題（64 題、8,192 token）當量尺很弱。B 組的裁判靠它判斷，可能擋掉有用的改法，也可能收下沒用的。
3. C 組只試了一種訓練設定、一個隨機種子，也沒有「用別人的答案訓練」的對照，所以不知道進步有多少是因為「用自己的答案」。
4. 測驗每題上限 16,384 token（原文 32,768），所以絕對分數比原文低。例如正對照的 MATH-500：這裡 32.0%，原文 48.1%。
5. 只有一個模型（Qwen3.5-0.8B）、一種題目（數學）、200 題測驗；每個比較的標準誤大約 3 分。
6. A 組的提示和工具照原文。換一種提示、給更多輪，結果可能不同。
7. 正式跑在 A 組思考 s3 第 8 輪當機一次（它自己的程式吃光記憶體）。修法與續跑記在 PREREG.md 第 6 節第 6 條，當機那次的紀錄也保留。
8. 沒有人類檢查過原始數據。

## 怎麼檢查

所有原始輸出都在 [`results/raw/`](results/raw/)：

| 位置 | 內容 |
|---|---|
| `test/` | 原模型與正對照在 200 題測驗上的完整答案 |
| `arm_a/` | A 組每一輪的回覆、程式、執行結果與錯誤訊息（`turns.jsonl`）；開跑前的工具檢查 |
| `arm_a_crashed/` | 當機那次的紀錄 |
| `arm_b/` | B 組每一輪的選單、它的回答、練習題答案、測驗的完整答案 |
| `arm_c/` | C 組每一輪的練習題答案、測驗的完整答案、訓練紀錄 |
| `exploratory/` | 跑完才加的兩個檢查 |
| `run.log`、`console.log` | 執行紀錄 |

練習題答案太長時，只存開頭 300 字和結尾 1,500 字；測驗答案都是完整的。

重算 RESULTS.md、summary.json 和圖（只需要 Python 與 matplotlib）：

```
python analysis/make_report.py results/raw results
```

沒有放進來的：C 組訓練後的權重（每份約 1 GB）、B 組的方向、正對照改動的 `.pt` 檔。這些可以用程式重新產生；正對照的權重在 Hugging Face 上公開（見 `selfedit/config.py`）。

## 怎麼重跑

需要一張 GPU（這次用 Colab 的 G4：NVIDIA RTX PRO 6000 Blackwell，96 GB）。正式實驗大約跑了 2 小時 15 分（當機前後兩段加起來）。

```
git clone https://github.com/speechlab0210/small-model-self-edit
python small-model-self-edit/colab/run_in_colab.py pilot   # 小規模試跑，只用來抓錯
python small-model-self-edit/colab/run_in_colab.py full    # 正式實驗
```

程式說明：`selfedit/engine.py`（改權重之後，生成與讀取都立刻用新權重）、`selfedit/arm_a.py`（A 組）、
`selfedit/arm_b.py`（B 組）、`selfedit/arm_c.py`（C 組）、`selfedit/mathqa.py`（評分，照抄原文作者的程式）、
`run_all.py`（照預先登記的順序跑）、`analysis/make_report.py`（從原始輸出算出所有數字和圖）。
`analysis/check_arm_c_weights.py` 和 `analysis/exploratory_direction.py` 是跑完才加的檢查。

## English summary

Can a 0.8B model (Qwen3.5-0.8B) edit its own weights and keep getting better at math? The recent "Weight-Space
Programming" post showed that a frontier model can fix this model's reasoning loops by editing 0.016% of its weights
(MATH-500 7.9% -> 48.1%). Here the editor is the small model itself. Design and predictions are in
[PREREG.md](PREREG.md), written before any run; every number comes from `analysis/make_report.py` run on the raw
outputs in `results/raw/` (tables: [results/RESULTS.md](results/RESULTS.md)).

Test set: 100 MATH-500 + 100 GSM8K problems, at most 16,384 tokens, paired seeds. Unedited model 10.5% (89% of answers
cut at the cap, mostly looping). The published edit (positive control) 45.5%, +35.0 points (SE 3.6).

- **Arm A, the frontier model's own notebook tools:** 6 runs x 30 turns; no edit ever took effect. Two runs refused
  and repeated one reply; the others wrote code that failed (mostly with the same error every turn), or that only
  defined functions and never called them.
- **Arm B, pick 1 of 8 ready-made edits for 6 rounds:** its picks were no better than random. It answered "1" (the
  first option of a shuffled list) in 33 of 60 samples, where a content-based choice would land there 12.5% of the
  time. Best result +3.5 points (SE 2.8, z 1.26), with the referee. The round-1 menu had no helpful option (-4.7 to 0.0
  points on 64 practice problems), so this arm could not show good choosing.
- **Arm C, self-training on its own answers:** keeping the answers it finished, with no answer key (C2), improved
  every round: 10.5% -> 27.0% -> 40.0% -> 44.5%, close to the positive control. It mainly learned to finish: finished
  answers 22 -> 124, but finished-and-wrong answers 2 -> 35. Keeping answers the key marks correct (C1) went
  5.0% -> 1.5% -> 38.0%: its first round added 6 answers that were cut at the cap (4 ending in repetition), and its
  last round was a single optimizer step on 3 answers, which moved test accuracy by 36.5 points.
- **The pre-registered criterion** (the model's own choices give >= 5 points with z >= 2, beating the random
  chooser; defined for Arms A and B) **was met by no arm.** Arm C changes the weights by gradient descent, not by the
  model's decisions, and has no random control.
- Predictions that held: P0, PA1, PA2, PB1, PB2. Did not hold: PA3, PB3, PB4 (as written; B2 gained 0 in both halves),
  PC1, PC2.
- Exploratory (not pre-registered): Arm B's looping direction is close to the frontier model's (|cos| 0.70 at layer
  15, 0.73 at layer 19), but the 64-problem, 8,192-token practice set is a weak instrument: the published edit shows
  only +10.9 points (SE 5.9) on it.

Limitations: one model, one task family, 200 test problems, one training recipe and seed for Arm C, a 16,384-token
cap (the post used 32,768). Everything in this repository was produced by an AI agent; no human has checked the raw
data.

## 授權

程式碼：MIT（見 [LICENSE](LICENSE)）。`selfedit/mathqa.py` 與 `prompts/` 的來源與授權見 [THIRD_PARTY.md](THIRD_PARTY.md)。
