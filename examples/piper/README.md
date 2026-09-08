# AgileX Piper 快速使用

主从遥操采集数据集，并用 LeRobot 训练。支持两种硬件接法。

## 环境

```bash
uv sync --extra piper
```

## 两种接法怎么选

| | 单 CAN 硬件主从 | 双 CAN 软件主从 |
|---|---|---|
| 接线 | 主从已在上位机配好，两臂共一路 CAN（如 `can0`） | 主臂 / 从臂各一路 CAN |
| 遥操谁做 | 硬件完成，不必跑 `2_teleop.sh` | PC 读主臂再写从臂 |
| 录数据 | [`3b_record_singleport.sh`](./3b_record_singleport.sh) | [`3_record.sh`](./3_record.sh) |
| action 来源 | 从臂当前关节状态 | 主臂指令 |

---

## A. 单 CAN 硬件主从（推荐你当前硬件）

### 1. 拉起 CAN

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

### 2. 录数据集

用主臂做动作（硬件带动从臂），PC 只连从臂侧 CAN 采数：

```bash
export DATASET_NAME=piper_demo
export CAN_PORT=can0          # 按实际接口名改
bash examples/piper/3b_record_singleport.sh
```

要点：`--direct_record=true`，不传 `--teleop.*`，且**不会**向机械臂发控制指令。  
数据默认写到仓库内 `train_data/<DATASET_NAME>/`，**不上传** Hugging Face Hub（可用 `DATASET_ROOT=...` 改路径）。

### 3. 训练

```bash
export DATASET_NAME=piper_demo
bash examples/piper/4_train.sh
```

---

## B. 双 CAN 软件主从

### 1. 初始化 CAN 名字

改 [`1_init_can.sh`](./1_init_can.sh) 里的 `USB_PORTS`，再：

```bash
bash examples/piper/1_init_can.sh
```

默认名：`can_leader` / `can_follower`。

### 2. 遥操

```bash
bash examples/piper/2_teleop.sh
```

### 3. 录数据集

```bash
export HF_USER=your_HF_id
export DATASET_NAME=piper_demo
bash examples/piper/3_record.sh
```

### 4. 训练

同 A 节，跑 `4_train.sh`。

---

## 评估 / 回放（可选）

策略部署请用 `lerobot-rollout`。回放某一 episode：

```bash
lerobot-replay \
  --robot.type=piper_follower \
  --robot.port=can0 \
  --robot.id=follower \
  --dataset.repo_id=${HF_USER}/piper_demo \
  --dataset.episode=0
```

（双 CAN 时把 `--robot.port` 改成 `can_follower`。）

## 注意

- 单 CAN：确认硬件主从已配好，PC 只监听、不抢控
- 双 CAN：上电前确认口名；不要和硬件主从模式混用
- 当前示例相机：`front=/dev/video4`，`wrist=/dev/video10`；换机时改脚本里的路径
- 长时间录制前确认 `dataset.repo_id`

## 代码位置

| 路径 | 作用 |
| --- | --- |
| `src/lerobot/motors/piper/` | CAN 电机总线 |
| `src/lerobot/robots/piper_follower/` | 从臂 Robot |
| `src/lerobot/teleoperators/piper_leader/` | 主臂 Teleoperator（双 CAN） |
| `--direct_record` | 单 CAN 直录开关（`lerobot-record`） |
