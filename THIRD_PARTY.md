# Third-party material

## From "LLMs can understand and write neural network weights" (Takehi, Teoh, Zhang, Khattab; MIT CSAIL)

Logs repository: https://github.com/RikiyaT/lm_understand-logs @ `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
Blog post: https://rikiyat.github.io/blog/weight-space-programming/

- `selfedit/mathqa.py`: functions copied from `exp8_math/code/mathqa.py` (listed in the file's docstring), under the
  MIT License below. `repetitive_end()` and the shortened `mv()` are not from that file.
  The 2026-11-04 audit additionally adapted `_timeouts()` for platforms without POSIX alarms.
- `prompts/authors_message_p50c.txt`: the task message of their run `exp8_math/runs/p50c` (`message.txt`), unchanged,
  under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). `prompts/arm_a_task.txt` is an adaptation of it
  (changed: the editor is told the weights are its own; budgets; paths; the parts of the forward-pass tool this harness
  does not implement; the evaluation's token cap). Compare the two files to see every change.
- The positive control downloads the edited weights `RikiyaT/Qwen3.5-0.8B-weight-space-programming`
  @ `d1671a178705bcab7644806d23c535517f35723b` from Hugging Face at run time; nothing of it is stored here.

```
MIT License

Copyright (c) 2026 the authors of "LLMs can understand and write neural network weights"

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Model and data (downloaded at run time, not stored here)

- Qwen3.5-0.8B (`Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`), under its own license.
- MATH-500 (`HuggingFaceH4/MATH-500`), GSM8K (`openai/gsm8k`), MATH training split
  (`DigitalLearningGmbH/MATH-lighteval` @ `0530c78699ea5e8eb5530600900e1f328b48acad`), each under its own license.
  `data/ids.json` lists only the ids of the problems used.
