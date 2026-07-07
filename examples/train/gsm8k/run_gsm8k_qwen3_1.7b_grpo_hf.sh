set -x

# DEBUG wrapper: runs the STOCK examples/train/gsm8k/run_gsm8k.sh byte-for-byte,
# only handling the managed-compute prerequisites the stock script assumes are
# already in place (uv on PATH, GSM8K parquets present) and steering logging +
# checkpoints to ephemeral, console-visible sinks. No edit to run_gsm8k.sh.

: "${DATA_DIR:="$HOME/data/gsm8k"}"

# HF/Modal base images may not ship `uv`; install it and put it on PATH.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        pip install --no-cache-dir uv || python -m pip install --no-cache-dir uv
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version

# Prep GSM8K parquets if missing.
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/validation.parquet" ]]; then
    echo "GSM8K parquets not found in ${DATA_DIR} — running prep script..."
    uv run --isolated examples/train/gsm8k/gsm8k_dataset.py --output_dir "${DATA_DIR}"
fi

# Console logging so metrics land in `orx logs` (stock default is wandb).
export LOGGER=console

# DEBUG fix B: force vLLM V0 engine — avoids the V1 mp EngineCore entirely.
export VLLM_USE_V1=0

# Run the stock script untouched. It forwards "$@", so override only the
# checkpoint path (stock writes under $HOME; keep it ephemeral) via a trailing
# CLI arg — the script body itself is unchanged.
bash examples/train/gsm8k/run_gsm8k.sh \
  trainer.ckpt_path="/tmp/skyrl-ckpts/gsm8k_1.5B_ckpt"
