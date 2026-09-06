#!/usr/bin/env bash
# SmolVLA 训练脚本：顶部变量可改，下方传入 lerobot-train。
# 默认值对齐源码配置（SmolVLAConfig / TrainPipelineConfig / DatasetConfig / WandBConfig）。
# 默认数据集与 scripts/record_xiebang_grasp.sh 一致。
# 微调推荐用 --policy.path=lerobot/smolvla_base；从零训练可改 POLICY_PATH="" 并设 POLICY_TYPE=smolvla。
#
# Prerequisites:
#   uv sync --locked --extra smolvla
#
# Usage (from repo root):
#   ./scripts/train_smolvla_xiebang.sh
#   BATCH_SIZE=8 STEPS=40000 ./scripts/train_smolvla_xiebang.sh
#   FREEZE_VISION_ENCODER=false TRAIN_EXPERT_ONLY=false ./scripts/train_smolvla_xiebang.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -x .tools/bin/uv ]]; then
  export PATH="$ROOT/.tools/bin:$PATH"
fi

##############################
# 基础 / 输出
##############################
DATASET_REPO_ID="${DATASET_REPO_ID:-bsll/pnp_xiebang_grasp_state}"
# 本地数据集目录（含 meta/info.json）；空则去 $HF_LEROBOT_HOME/<repo_id> 或 Hub 下载
DATASET_ROOT="${DATASET_ROOT:-$HOME/.cache/huggingface/lerobot/${DATASET_REPO_ID}}"
DATASET_REVISION="${DATASET_REVISION:-}"              # 数据集 revision，空则默认
DATASET_EPISODES="${DATASET_EPISODES:-}"              # 例如 "[0,1,2]"，空则用全部 episode
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/smolvla/pnp_xiebang_grasp_state}"
JOB_NAME="${JOB_NAME:-smolvla_xiebang_grasp}"
RESUME="${RESUME:-false}"                             # true 时需配合 CONFIG_PATH
CONFIG_PATH="${CONFIG_PATH:-}"                        # resume 时指向 train_config.json
SEED="${SEED:-42}"
CUDNN_DETERMINISTIC="${CUDNN_DETERMINISTIC:-false}"

##############################
# Policy 加载方式
##############################
# 微调预训练权重（推荐）
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
# 从零训练时：POLICY_PATH="" 且 POLICY_TYPE=smolvla
POLICY_TYPE="${POLICY_TYPE:-smolvla}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_USE_AMP="${POLICY_USE_AMP:-false}"
POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-false}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"                  # push_to_hub=true 时必填
POLICY_PRIVATE="${POLICY_PRIVATE:-}"
POLICY_TAGS="${POLICY_TAGS:-}"                        # 例如 '["smolvla","xiebang"]'
POLICY_LICENSE="${POLICY_LICENSE:-}"

##############################
# 训练超参（TrainPipelineConfig 默认）
##############################
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-100000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
# 仿真环境评测频率；真机数据集无 --env 时请保持 0
ENV_EVAL_FREQ="${ENV_EVAL_FREQ:-0}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
TOLERANCE_S="${TOLERANCE_S:-1e-4}"
USE_POLICY_TRAINING_PRESET="${USE_POLICY_TRAINING_PRESET:-true}"

##############################
# 数据集相关
##############################
USE_IMAGENET_STATS="${USE_IMAGENET_STATS:-true}"
STREAMING="${STREAMING:-false}"
VIDEO_BACKEND="${VIDEO_BACKEND:-}"                    # 空则用系统默认 codec
# 图像增强（默认关闭可改 false）
IMAGE_TRANSFORMS_ENABLE="${IMAGE_TRANSFORMS_ENABLE:-true}"
IMAGE_TRANSFORMS_MAX_NUM="${IMAGE_TRANSFORMS_MAX_NUM:-3}"
IMAGE_TRANSFORMS_RANDOM_ORDER="${IMAGE_TRANSFORMS_RANDOM_ORDER:-true}"

# 相机/状态 key 重命名。采集用 front/wrist，对齐 smolvla_base 的 camera1/camera2：
RENAME_MAP="${RENAME_MAP:-{\"observation.images.front\":\"observation.images.camera1\",\"observation.images.wrist\":\"observation.images.camera2\"}}"
# 若不需要重命名：RENAME_MAP='{}'

##############################
# SmolVLA 结构 / 输入输出（SmolVLAConfig 默认）
##############################
N_OBS_STEPS="${N_OBS_STEPS:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
MAX_STATE_DIM="${MAX_STATE_DIM:-32}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-32}"
RESIZE_IMGS="${RESIZE_IMGS:-[512,512]}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-0}"                   # 数据相机数少于预训练期望时可补空相机
ADAPT_TO_PI_ALOHA="${ADAPT_TO_PI_ALOHA:-false}"
USE_DELTA_JOINT_ACTIONS_ALOHA="${USE_DELTA_JOINT_ACTIONS_ALOHA:-false}"
TOKENIZER_MAX_LENGTH="${TOKENIZER_MAX_LENGTH:-48}"
NUM_STEPS="${NUM_STEPS:-10}"                          # 推理 denoising steps
USE_CACHE="${USE_CACHE:-true}"

##############################
# 微调开关
##############################
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
TRAIN_STATE_PROJ="${TRAIN_STATE_PROJ:-true}"

##############################
# Optimizer / Scheduler（policy preset 默认）
# use_policy_training_preset=true 时生效
##############################
OPTIMIZER_LR="${OPTIMIZER_LR:-2e-5}"
OPTIMIZER_BETAS="${OPTIMIZER_BETAS:-[0.9,0.95]}"
OPTIMIZER_EPS="${OPTIMIZER_EPS:-1e-8}"
OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-1e-10}"
OPTIMIZER_GRAD_CLIP_NORM="${OPTIMIZER_GRAD_CLIP_NORM:-10}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-100000}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"

##############################
# VLM / Expert 结构
##############################
VLM_MODEL_NAME="${VLM_MODEL_NAME:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"
# 从零训 expert 时通常 false；从 pretrained smolvla_base 加载时由 checkpoint 决定
LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS:-false}"
ADD_IMAGE_SPECIAL_TOKENS="${ADD_IMAGE_SPECIAL_TOKENS:-false}"
ATTENTION_MODE="${ATTENTION_MODE:-cross_attn}"
PREFIX_LENGTH="${PREFIX_LENGTH:--1}"
PAD_LANGUAGE_TO="${PAD_LANGUAGE_TO:-longest}"         # longest | max_length
NUM_EXPERT_LAYERS="${NUM_EXPERT_LAYERS:-16}"
NUM_VLM_LAYERS="${NUM_VLM_LAYERS:-16}"
SELF_ATTN_EVERY_N_LAYERS="${SELF_ATTN_EVERY_N_LAYERS:-2}"
EXPERT_WIDTH_MULTIPLIER="${EXPERT_WIDTH_MULTIPLIER:-0.75}"
MIN_PERIOD="${MIN_PERIOD:-4e-3}"
MAX_PERIOD="${MAX_PERIOD:-4.0}"
COMPILE_MODEL="${COMPILE_MODEL:-false}"
COMPILE_MODE="${COMPILE_MODE:-max-autotune}"

##############################
# Sample weighting / RA-BC（默认关闭）
# 当前仓库用 --sample_weighting.*，不再支持 --use_rabc / --rabc_*
##############################
USE_RABC="${USE_RABC:-false}"
RABC_PROGRESS_PATH="${RABC_PROGRESS_PATH:-}"
RABC_KAPPA="${RABC_KAPPA:-0.01}"
RABC_EPSILON="${RABC_EPSILON:-1e-6}"
RABC_HEAD_MODE="${RABC_HEAD_MODE:-sparse}"

##############################
# Eval（无 env 时一般不跑评测，仅保留可调项）
##############################
EVAL_N_EPISODES="${EVAL_N_EPISODES:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"
EVAL_USE_ASYNC_ENVS="${EVAL_USE_ASYNC_ENVS:-false}"

##############################
# WandB / SwanLab（WandBConfig；enable=true 时走 SwanLab）
##############################
# 默认关闭：未登录 / 无 entity 权限时 wandb.init 会直接 permission denied
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
WANDB_ENTITY="${WANDB_ENTITY:-}"                      # 有权限时再填，例如你的 wandb team/user
WANDB_NOTES="${WANDB_NOTES:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_MODE="${WANDB_MODE:-}"                          # online | offline | disabled | local
WANDB_ADD_TAGS="${WANDB_ADD_TAGS:-true}"

##############################
# 组装命令
##############################
CMD=(
  lerobot-train
  --dataset.repo_id="${DATASET_REPO_ID}"
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
  --resume="${RESUME}"
  --seed="${SEED}"
  --cudnn_deterministic="${CUDNN_DETERMINISTIC}"
  --num_workers="${NUM_WORKERS}"
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --env_eval_freq="${ENV_EVAL_FREQ}"
  --log_freq="${LOG_FREQ}"
  --tolerance_s="${TOLERANCE_S}"
  --save_checkpoint="${SAVE_CHECKPOINT}"
  --save_freq="${SAVE_FREQ}"
  --use_policy_training_preset="${USE_POLICY_TRAINING_PRESET}"
  --rename_map="${RENAME_MAP}"
  --dataset.use_imagenet_stats="${USE_IMAGENET_STATS}"
  --dataset.streaming="${STREAMING}"
  --dataset.image_transforms.enable="${IMAGE_TRANSFORMS_ENABLE}"
  --dataset.image_transforms.max_num_transforms="${IMAGE_TRANSFORMS_MAX_NUM}"
  --dataset.image_transforms.random_order="${IMAGE_TRANSFORMS_RANDOM_ORDER}"
  --policy.device="${POLICY_DEVICE}"
  --policy.use_amp="${POLICY_USE_AMP}"
  --policy.push_to_hub="${POLICY_PUSH_TO_HUB}"
  --policy.n_obs_steps="${N_OBS_STEPS}"
  --policy.chunk_size="${CHUNK_SIZE}"
  --policy.n_action_steps="${N_ACTION_STEPS}"
  --policy.max_state_dim="${MAX_STATE_DIM}"
  --policy.max_action_dim="${MAX_ACTION_DIM}"
  --policy.resize_imgs_with_padding="${RESIZE_IMGS}"
  --policy.empty_cameras="${EMPTY_CAMERAS}"
  --policy.adapt_to_pi_aloha="${ADAPT_TO_PI_ALOHA}"
  --policy.use_delta_joint_actions_aloha="${USE_DELTA_JOINT_ACTIONS_ALOHA}"
  --policy.tokenizer_max_length="${TOKENIZER_MAX_LENGTH}"
  --policy.num_steps="${NUM_STEPS}"
  --policy.use_cache="${USE_CACHE}"
  --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}"
  --policy.train_expert_only="${TRAIN_EXPERT_ONLY}"
  --policy.train_state_proj="${TRAIN_STATE_PROJ}"
  --policy.optimizer_lr="${OPTIMIZER_LR}"
  --policy.optimizer_betas="${OPTIMIZER_BETAS}"
  --policy.optimizer_eps="${OPTIMIZER_EPS}"
  --policy.optimizer_weight_decay="${OPTIMIZER_WEIGHT_DECAY}"
  --policy.optimizer_grad_clip_norm="${OPTIMIZER_GRAD_CLIP_NORM}"
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}"
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"
  --policy.vlm_model_name="${VLM_MODEL_NAME}"
  --policy.load_vlm_weights="${LOAD_VLM_WEIGHTS}"
  --policy.add_image_special_tokens="${ADD_IMAGE_SPECIAL_TOKENS}"
  --policy.attention_mode="${ATTENTION_MODE}"
  --policy.prefix_length="${PREFIX_LENGTH}"
  --policy.pad_language_to="${PAD_LANGUAGE_TO}"
  --policy.num_expert_layers="${NUM_EXPERT_LAYERS}"
  --policy.num_vlm_layers="${NUM_VLM_LAYERS}"
  --policy.self_attn_every_n_layers="${SELF_ATTN_EVERY_N_LAYERS}"
  --policy.expert_width_multiplier="${EXPERT_WIDTH_MULTIPLIER}"
  --policy.min_period="${MIN_PERIOD}"
  --policy.max_period="${MAX_PERIOD}"
  --policy.compile_model="${COMPILE_MODEL}"
  --policy.compile_mode="${COMPILE_MODE}"
  --eval.n_episodes="${EVAL_N_EPISODES}"
  --eval.batch_size="${EVAL_BATCH_SIZE}"
  --eval.use_async_envs="${EVAL_USE_ASYNC_ENVS}"
  --wandb.enable="${WANDB_ENABLE}"
  --wandb.disable_artifact="${WANDB_DISABLE_ARTIFACT}"
  --wandb.project="${WANDB_PROJECT}"
  --wandb.add_tags="${WANDB_ADD_TAGS}"
)

# RA-BC：仅在启用时追加新版 sample_weighting 参数
if [[ "${USE_RABC}" == "true" ]]; then
  CMD+=(
    --sample_weighting.type=rabc
    --sample_weighting.kappa="${RABC_KAPPA}"
    --sample_weighting.epsilon="${RABC_EPSILON}"
    --sample_weighting.head_mode="${RABC_HEAD_MODE}"
  )
  [[ -n "${RABC_PROGRESS_PATH}" ]] && CMD+=(--sample_weighting.progress_path="${RABC_PROGRESS_PATH}")
fi

# policy 加载：优先 path（微调），否则 type（从零）
if [[ -n "${POLICY_PATH}" ]]; then
  CMD+=(--policy.path="${POLICY_PATH}")
else
  CMD+=(--policy.type="${POLICY_TYPE}")
fi

# 可选字符串参数：非空才追加，避免传空覆盖默认
[[ -n "${DATASET_ROOT}" ]] && CMD+=(--dataset.root="${DATASET_ROOT}")
[[ -n "${DATASET_REVISION}" ]] && CMD+=(--dataset.revision="${DATASET_REVISION}")
[[ -n "${DATASET_EPISODES}" ]] && CMD+=(--dataset.episodes="${DATASET_EPISODES}")
[[ -n "${VIDEO_BACKEND}" ]] && CMD+=(--dataset.video_backend="${VIDEO_BACKEND}")
[[ -n "${CONFIG_PATH}" ]] && CMD+=(--config_path="${CONFIG_PATH}")
[[ -n "${POLICY_REPO_ID}" ]] && CMD+=(--policy.repo_id="${POLICY_REPO_ID}")
[[ -n "${POLICY_PRIVATE}" ]] && CMD+=(--policy.private="${POLICY_PRIVATE}")
[[ -n "${POLICY_TAGS}" ]] && CMD+=(--policy.tags="${POLICY_TAGS}")
[[ -n "${POLICY_LICENSE}" ]] && CMD+=(--policy.license="${POLICY_LICENSE}")
[[ -n "${WANDB_ENTITY}" ]] && CMD+=(--wandb.entity="${WANDB_ENTITY}")
[[ -n "${WANDB_NOTES}" ]] && CMD+=(--wandb.notes="${WANDB_NOTES}")
[[ -n "${WANDB_RUN_ID}" ]] && CMD+=(--wandb.run_id="${WANDB_RUN_ID}")
[[ -n "${WANDB_MODE}" ]] && CMD+=(--wandb.mode="${WANDB_MODE}")

echo "========== SmolVLA train =========="
echo "dataset : ${DATASET_REPO_ID}"
echo "root    : ${DATASET_ROOT:-"(hub/default)"}"
echo "policy  : ${POLICY_PATH:-type=${POLICY_TYPE}}"
echo "output  : ${OUTPUT_DIR}"
echo "job     : ${JOB_NAME}"
echo "batch   : ${BATCH_SIZE}  steps: ${STEPS}  lr: ${OPTIMIZER_LR}"
echo "rename  : ${RENAME_MAP}"
echo "==================================="
printf '%q ' "${CMD[@]}"
echo

exec "${CMD[@]}"
