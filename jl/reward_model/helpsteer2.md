# HelpSteer2-Preference → Bradley–Terry Reward Model Training (Recipe)

This note explains how to load and preprocess the **HelpSteer2 preference annotations** (“HelpSteer2-Preference”) from the Hugging Face dataset **`nvidia/HelpSteer2`**, and how to convert them into pairwise data suitable for **Bradley–Terry (BT)** reward-model training.

## What “HelpSteer2-Preference” is

- **Paper**: *HelpSteer2-Preference: Complementing Ratings with Preferences* (arXiv:2410.01257). ([arxiv.org](https://arxiv.org/abs/2410.01257?utm_source=chatgpt.com))  
- **Data**: Preference annotations are released inside the Hugging Face dataset repo **`nvidia/HelpSteer2`** under a **`preference/`** subdirectory (loaded via `data_dir="preference"`). ([huggingface.co](https://huggingface.co/datasets/nvidia/HelpSteer2))

---

## 1) Load the preference data from Hugging Face

### Key point: preference data is a *subdirectory* (“data_dir”), not a HF split
The preference annotations are not loaded via `split="preference"`. Instead, load the dataset with `data_dir="preference"`, then use the `split` column inside the table to separate train/validation. ([huggingface.co](https://huggingface.co/datasets/nvidia/HelpSteer2))

python
from datasets import load_dataset

pref = load_dataset("nvidia/HelpSteer2", data_dir="preference", split="train")

> Note: this returns one Dataset that includes a `split` column with values like `"train"` and `"validation"`. ([huggingface.co](https://huggingface.co/datasets/nvidia/HelpSteer2))

### Split into train/validation using the `split` column

python
pref_train = pref.filter(lambda x: x["split"] == "train")
pref_val   = pref.filter(lambda x: x["split"] == "validation")

The dataset card states the preference annotations are matched to the original HelpSteer2 train/validation prompts/responses, and `split` is included for clarity. ([huggingface.co](https://huggingface.co/datasets/nvidia/HelpSteer2))

---

## 2) Schema and sign convention you’ll use

Each preference example (per dataset card) includes at least:

- `prompt`
- `response_1`
- `response_2`
- `preference_strength` ∈ {-3,-2,-1,0,1,2,3}
- `preference_statement`, `preference_elaboration` (human-written, post-processed)
- `three_most_similar_preferences` (subset used to compute overall preference)
- `all_preferences_unprocessed` (raw preferences; NVIDIA notes these weren’t used in their experiments) ([huggingface.co](https://huggingface.co/datasets/nvidia/HelpSteer2))

### `preference_strength` sign convention
- -3/-2/-1: **Response 1 is preferred over Response 2** (stronger magnitude = stronger preference)
- 0: tie / about the same
- +1/+2/+3: **Response 2 is preferred over Response 1** ([huggingface.co](https://huggingface.co/datasets/nvidia/HelpSteer2))

---

## 3) Preprocess into BT pairs (winner/loser)

### Recommended: drop ties
For standard BT training, discard ambiguous pairs with `preference_strength == 0`.

python
pref_train = pref_train.filter(lambda x: x["preference_strength"] != 0)
pref_val   = pref_val.filter(lambda x: x["preference_strength"] != 0)

The paper discusses filtering based on agreement/outliers (e.g., using the three most similar preferences and discarding high-spread cases). ([arxiv.org](https://arxiv.org/html/2410.01257v2?utm_source=chatgpt.com))

### Map to `chosen`/`rejected` and optional strength weight

python
def to_bt_pair(ex):
    s = ex["preference_strength"]
    if s < 0:
        chosen, rejected = ex["response_1"], ex["response_2"]
    else:
        chosen, rejected = ex["response_2"], ex["response_1"]

    ex["chosen"] = chosen
    ex["rejected"] = rejected
    ex["weight"] = abs(s)  # 1..3 (optional)
    return ex

pref_train = pref_train.map(to_bt_pair)
pref_val   = pref_val.map(to_bt_pair)

---

## 4) Bradley–Terry reward-model objective

Train a scalar reward model:

- `rθ(prompt, response) → ℝ`

For each pair `(x, y⁺, y⁻)`:

- Δr = rθ(x, y⁺) − rθ(x, y⁻)
- BT probability: σ(Δr)

### Standard BT loss (binary preference)

- L = -log σ(Δr)

### Strength-aware BT (recommended for this dataset)

Use `w = |preference_strength| ∈ {1,2,3}`:

- L = -w * log σ(Δr)

HelpSteer2-Preference explicitly provides preference strength levels. ([huggingface.co](https://huggingface.co/datasets/nvidia/HelpSteer2))

---

## 5) Practical checklist for implementation

1. Load: `load_dataset("nvidia/HelpSteer2", data_dir="preference", split="train")` ([huggingface.co](https://huggingface.co/datasets/nvidia/HelpSteer2))  
2. Split: filter by `row["split"] in {"train","validation"}`  
3. Filter: drop ties (`preference_strength == 0`)  
4. Map: negative ⇒ response_1 wins; positive ⇒ response_2 wins  
5. Train: BT loss on Δr; optionally weight by `abs(preference_strength)`  
6. Eval: pairwise accuracy: fraction where `Δr > 0` matches label sign; optionally report by strength bucket (|s|=1/2/3)

---

## Appendix: Why you probably shouldn’t use `all_preferences_unprocessed`

The dataset card notes that the raw preferences are released, but NVIDIA’s reported experiments used the post-processed preference signal built from the “three most similar preferences” to reduce outliers and disagreement. ([huggingface.co](https://huggingface.co/datasets/nvidia/HelpSteer2))  
The paper describes preference preprocessing/filters intended to improve agreement. ([arxiv.org](https://arxiv.org/html/2410.01257v2?utm_source=chatgpt.com))
