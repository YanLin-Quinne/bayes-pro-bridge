# Backbone contract used for every reported run (sourced by the other scripts).
# Pinned Hugging Face revisions, the training and evaluation script, and backbone specific arguments.
case "${1:?backbone}" in
  llama3.2-1b) BASE=meta-llama/Llama-3.2-1B; REV=4e20de362430cd3b72f300e6b0f18e50e7166e08
    TRAINER=train_llama.py; EVALUATOR=evaluate_llama.py; TRAIN_ARGS=(); EVAL_ARGS=(--model_dtype bf16) ;;
  llama3.2-3b) BASE=meta-llama/Llama-3.2-3B; REV=13afe5124825b4f3751f836b40dafda64c1ed062
    TRAINER=train_llama.py; EVALUATOR=evaluate_llama.py; TRAIN_ARGS=(); EVAL_ARGS=(--model_dtype bf16) ;;
  qwen2-1.5b-instruct) BASE=Qwen/Qwen2-1.5B-Instruct; REV=ba1cf1846d7df0a0591d6c00649f57e798519da8
    TRAINER=train_qwen2_instruct.py; EVALUATOR=evaluate_qwen2_instruct.py
    TRAIN_ARGS=(--model_revision "$REV" --attn_implementation sdpa --seed 42 --experiment_family release)
    EVAL_ARGS=(--model_revision "$REV" --model_dtype bf16) ;;
  qwen3-4b-base) BASE=Qwen/Qwen3-4B-Base; REV=906bfd4b4dc7f14ee4320094d8b41684abff8539
    TRAINER=train_qwen3.py; EVALUATOR=evaluate_qwen3.py
    TRAIN_ARGS=(--model_revision "$REV" --attn_implementation sdpa --seed 42 --experiment_family release)
    EVAL_ARGS=(--model_revision "$REV") ;;
  *) echo "unknown backbone $1 (llama3.2-1b, llama3.2-3b, qwen2-1.5b-instruct, qwen3-4b-base)" >&2; exit 2 ;;
esac
# The Llama scripts load the model by repository id: pin it with `hf download $BASE --revision $REV` first.
