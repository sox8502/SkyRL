set -x

# --- Fix vLLM V1 EngineCore silent death (SIGBUS from a tiny /dev/shm) ---
# vLLM V1 runs EngineCore as an mp child and mmaps a multi-hundred-MB ring
# buffer into /dev/shm. HF Jobs containers default /dev/shm to ~64MB, so the
# mmap faults with SIGBUS and the child is killed before it prints anything —
# the exact signature we saw: child goes silent at the shm handshake, parent
# reports "Engine core initialization failed ... Failed core proc(s): {}".
# Enlarge /dev/shm (loud, so the log shows the before/after) before any Python.
echo "=== /dev/shm before ==="; df -h /dev/shm || true
if mount -o remount,size=16g /dev/shm 2>/dev/null; then
    echo "remounted /dev/shm to 16g"
else
    echo "in-place remount of /dev/shm denied; falling back to a tmpfs under /tmp"
    export VLLM_SHM_DIR=/tmp/vllm_shm
    mkdir -p "$VLLM_SHM_DIR"
    # If we can mount a fresh tmpfs there, do so; otherwise /tmp is usually a
    # large tmpfs already and vLLM honoring TMPDIR still helps.
    mount -t tmpfs -o size=16g tmpfs "$VLLM_SHM_DIR" 2>/dev/null || true
fi
echo "=== /dev/shm after ==="; df -h /dev/shm || true

# Colocated GRPO training+generation for Qwen3-1.7B-Base on GSM8K.
# Mirrors examples/train/gsm8k/run_gsm8k.sh (the known-good colocated config),
# changing only: model -> Qwen3-1.7B-Base, single-GPU (NUM_GPUS=1) to fit one
# large HF GPU, console logging so metrics land in `orx logs`, and ephemeral
# checkpoints (HF Jobs has no persistent volume).

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MODEL:=Qwen/Qwen3-1.7B-Base}"

# The HF Jobs base image does not ship `uv`; install it and put it on PATH.
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

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=3 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=64 \
  trainer.micro_train_batch_size_per_gpu=64 \
  trainer.ckpt_interval=100000 \
  trainer.hf_save_interval=100000 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k" \
  trainer.run_name="gsm8k_qwen3_1.7b_grpo" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.ckpt_path="/tmp/skyrl-ckpts/gsm8k_qwen3_1.7b" \
  $@
