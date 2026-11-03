"""Every fixed setting of the experiment (PREREG.md sections 2-3). Runs read their numbers from here only."""
import os

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REV = "2fc06364715b967f1860aea9cf38778875588b17"
# the positive control: the edit GPT-6 Astra made in the blog post's 50-pass run (two tensors)
ASTRA_ID = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
ASTRA_REV = "d1671a178705bcab7644806d23c535517f35723b"
ASTRA_TENSORS = ("model.language_model.layers.15.mlp.down_proj.weight",
                 "model.language_model.layers.19.mlp.down_proj.weight")

WORK = os.environ.get("SELFEDIT_WORK", "/content/selfedit")
MODEL_DIR = os.path.join(WORK, "model")
OUT = os.path.join(WORK, "out")
DATA = os.path.join(WORK, "data")

LM_PREFIX = "model.language_model."
N_LAYERS, HIDDEN, MLP_WIDTH = 24, 1024, 3584
FULL_ATTN_LAYERS = (3, 7, 11, 15, 19, 23)
N_HEADS, HEAD_DIM = 8, 256
FROZEN = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")

# sampling: the model card's thinking-mode benchmark setting (also the blog post's evaluation) and its non-thinking one
THINK = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)
NOTHINK = dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)
TEST_MAX_TOKENS = 16384          # the post: 32,768 (halved for compute; the same for every model)
DEV_MAX_TOKENS = 8192
SUBSET_SEED = 20261103
N_TEST_EACH = 100                # MATH-500 and GSM8K test problems each

# engine
GPU_MEM = float(os.environ.get("SELFEDIT_GPU_MEM", 0.55))
MAX_MODEL_LEN = 34000
MAX_NUM_SEQS = 128

# Arm A (the authors' tools, the small model as its own editor)
A_VARIANTS = ("nothink", "think")
A_SEEDS = (1, 2, 3)
A_TURNS, A_PY_RUNS, A_PY_TIMEOUT, A_PASSES = 30, 30, 120, 5
A_GEN_MAX, A_TF_MAX_TEXTS, A_TF_MAX_TOKENS = 64, 64, 2048
A_TURN_TOKENS = {"think": 8192, "nothink": 4096}
A_PROMPT_BUDGET = 24000          # tokens of conversation kept (the task message is always kept)
A_OUTPUT_CHARS = 4000            # kernel output shown back per run

# Arm B (choose from a menu, 6 rounds)
B_ROUNDS, B_K, B_DEV_N, B_DIAG_N = 6, 8, 64, 192
B_WINDOW = 1024
B_DIR_TEXTS = 48                 # at most this many looping / productive texts per direction estimate
B_LAYERS = tuple(range(8, 23))
B_CHOICE_SAMPLES, B_TABLE_SAMPLES = 5, 20
B_TEST_AFTER = (3, 6)

# Arm C (self-training, gradient updates)
C_ROUNDS, C_GEN_N, C_LR, C_EPOCHS = 3, 256, 1e-5, 1
