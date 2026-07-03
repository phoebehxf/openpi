# Piper 策略离线验证 & 诊断记录

面向 `pi05_piper_pick_and_place` 微调任务的离线评测工具和一次完整的问题排查记录。

核心思路:**拿训练集里的 episode 喂给微调后的模型,比较预测的 action chunk 和真值(ground truth)的误差。**
- 训练集上都拟合不了 → 训练侧问题(delta / norm / 欠训练)。
- 训练集拟合得很好但实机不行 → 部署侧问题(transform / 相机映射 / 执行频率 / 夹爪)。

---

## 1. 两个评测脚本

两个脚本从**不同路径**测同一件事(预测动作 vs 真值),互相印证。

### 1.1 `scripts/offline_eval_action_mse.py` —— 走部署路径

用 `policy.infer()` 跑,和实机部署(websocket server)**完全相同**的变换链:
`repack → LiberoInputs → Normalize → 模型 → Unnormalize → LiberoOutputs`。
输出是**原始动作空间(弧度)**的预测,直接和数据集里的真值 action 比。

```bash
uv run scripts/offline_eval_action_mse.py \
  --config-name pi05_piper_pick_and_place \
  --checkpoint-dir checkpoints/pi05_piper_pick_and_place/piper_pick_pi05_lora_v1/9999 \
  --num-samples 200
```

主要参数:
| 参数 | 默认 | 说明 |
|---|---|---|
| `--config-name` | `pi05_piper_pick_and_place` | 训练 config 名 |
| `--checkpoint-dir` | .../9999 | checkpoint 目录 |
| `--num-samples` | 200 | 从数据集随机抽多少帧 |
| `--seed` | 0 | 随机种子(固定才能复现/比较) |
| `--num-infer-samples` | 1 | 每帧采样几次取平均(flow 有随机性) |

**怎么读输出**(每一维:0–5 是关节[弧度],6 是夹爪):
- `model RMSE` —— 模型预测误差,越小越好。
- `baseline hold-state` —— "保持当前姿态不动"的误差。**模型必须显著低于它**,否则等于"还不如不动"。
- `baseline repeat-gt[0]` —— "重复第一帧动作"的误差,反映 chunk 内动作变化幅度(可达到的下限)。
- `nRMSE(/std)` —— RMSE ÷ 该维动作 std。**> 1.0 表示比直接输出均值还差。**
- `per-horizon-step RMSE` —— 误差是否随 chunk 步数增长(判断漂移 vs 固定偏移)。

### 1.2 `scripts/offline_eval_modelspace.py` —— 走训练路径

用**训练那条原生数据管线**(`create_torch_data_loader`)取 batch,直接调模型的
`sample_actions` / `compute_loss`。用来:
1. 复现训练 loss —— 验证 checkpoint 权重加载正确;
2. 在**归一化空间**和**原始弧度空间**同时报采样动作 RMSE;
3. 排除"我的部署路径评测脚本本身有 bug"的可能(两条路径结果一致 = 离线管线没问题)。

```bash
# 注意:如果 GPU 0 被占,换一张空闲卡,并用单卡避免 sharding 问题
CUDA_VISIBLE_DEVICES=1 uv run scripts/offline_eval_modelspace.py \
  --checkpoint-dir checkpoints/pi05_piper_pick_and_place/piper_pick_pi05_lora_v1/9999 \
  --num-batches 25 --batch-size 8
```

主要参数:`--num-batches`(25)、`--batch-size`(8)、`--num-sample-steps`(10,去噪步数)、`--seed`(0)。

**怎么读输出**:
- `mean compute_loss` —— 应≈ wandb 上的 loss(~0.017),对上了说明权重没加载错。
- `raw-space RMSE` —— 应该和 `offline_eval_action_mse.py` 的数值接近(两条路径互印证)。
- `per-dim: norm_RMSE / raw_RMSE / hold_RMSE` —— 同上,重点看每维是否赢过 hold 基线。

> ⚠️ **坑**:这台机器有 3 张 GPU,训练数据加载器默认把 batch 切到所有卡上(要求 batch 能被卡数整除)。评测时用 `CUDA_VISIBLE_DEVICES=<单卡>` 最省事。多卡 vs 单卡结果只差浮点级(~1e-3),不影响结论。

---

## 2. 今天的发现(2026-07-03)

### 结论:是**欠训练**,不是部署 transform 问题

排查链:

1. **两条离线路径结果一致** —— 部署路径 RMSE 0.228 ≈ 训练路径 RMSE 0.213(弧度)。
   → 离线管线的 transform / 映射**没问题**,不是这里的 bug。

2. **norm stats 正常** —— checkpoint 的 `norm_stats.json` 和数据集 `stats.json` 完全一致。

3. **delta/绝对量正常** —— action 存的是绝对关节值(均值≈0.96,非≈0),训练和推理都是
   `use_delta_joint_actions=False`,一致。

4. **loss 小 ≠ 动作准**(关键)—— flow-matching 训练 loss 是"随机噪声水平上预测速度场"的
   MSE,大部分时间步最优解就是预测均值,很好学,所以 loss 会很低(0.013)却和"采样动作一般"共存。
   **判断扩散策略好坏必须看采样动作 RMSE,不能只看 loss。**

5. **RMSE 还在下降** —— 5000 步 → 9999 步,采样 RMSE 从 0.247 → 0.213 还在线性下降,
   说明训练在"LR 刚爬到有效区间"就被停掉了,远没收敛。

6. **根因**:配置 `warmup_steps=10_000` 正好等于 `num_train_steps=10_000` ——
   学习率整段都在爬坡(平均 ~2.5e-5,前几千步接近冻结),只有最后才碰到峰值。
   (openpi 默认本来就是 warmup 1000 / decay 30k。)

### 关键数据

| checkpoint | compute_loss | 采样 RMSE(弧度) | hold 基线 | 夹爪 dim6 | hold(夹爪) |
|---|---|---|---|---|---|
| 5000 | 0.0181 | 0.247 | 0.196 | 0.224 | 0.086 |
| 9999 | 0.0132 | **0.213** | 0.196 | **0.172** | 0.086 |

**⚠️ 现在的模型(0.213)仍高于"不动"基线(0.196),即比完全不动还差 → 实机表现成"什么都不是"完全说得通。夹爪那一维(0.172 vs 0.086)差了一倍,是重灾区。**

### 验收线(重训后)

不是"RMSE 下降"就算好 —— 必须**显著低于 hold 基线**,尤其 **dim 6(夹爪)必须压到 0.086 以下**。

---

## 3. 已做的改动

- **`scripts/train.py`**:训练循环里加了采样动作评测,每 `eval_interval` 步往 wandb 记
  `eval/sample_rmse_norm` 和 `eval/gripper_rmse_norm`(夹爪单独盯);评测 seed 固定,
  使不同 checkpoint 可比。**训练时盯这个,不要只盯 loss。**
- **`src/openpi/training/config.py`**:新增 `eval_interval`(1000)、`eval_num_batches`(2)、
  `eval_num_sample_steps`(10);LR 调度已改为 warmup 1000 / decay 30k / 30k 步。

---

## 4. 重训 & 后续建议

### 重训(从 10001 续训,不用从 pretrained 重来)

```bash
CUDA_VISIBLE_DEVICES=1,2 uv run scripts/train.py pi05_piper_pick_and_place \
  --exp-name piper_pick_pi05_lora_v1 --resume \
  --eval-interval 1000 --eval-num-batches 4
```

### 待办(尚未实施,按优先级)

1. **[高] 部署客户端(`scripts/piper_remote_client.py`)有硬伤,和模型质量无关**:
   - 夹爪被禁用(`disable_gripper=True`),且 `get_state()` 把夹爪状态写死成常数 1.0
     (对 pick-and-place 是致命的,且喂给模型 OOD 状态)。→ 重新启用并读真实夹爪。
   - 绝对模式下 `±0.03 rad/step` 钳制 + `action_alpha=0.15` 把动作压到近乎不动。
   - `open_loop_horizon=1` 只执行 chunk 第一步,且 `control_hz=3` vs 训练 30fps 的频率错配。
   - **重训后如果 RMSE 过线但实机仍不行,先回来查这里,别再怀疑训练。**
2. **[高] held-out 留出集评测**:从 182 条里留 ~10 条不进训练集,RMSE 在留出集上打
   (训练集上低于基线只证明拟合,不能预测实机)。需要改数据加载支持按 episode 排除。
3. **[中] checkpoint 策略**:`save_interval=5000`,最后按 held-out RMSE 选,不要默认拿最后一个;
   留出曲线若 20k 后回升就提前停。
4. **[中] 若换全量微调**:`peak_lr` 降到 ~2.5e-5(全量在小数据上 5e-5 容易 NaN)。
5. **[中] 夹爪**:若重训后 dim6 还压不下,先查数据里夹爪信号质量(开合是否干脆、有无半开/抖动),
   再考虑部署时对夹爪输出做二值化(阈值开/闭)。

---

## 5. 已知事故

做 resume 冒烟测试时,orbax 保留策略删掉了 **`9999`**。现存 `10001`(= 9999 + 2 步,实质等同)
和 `5000`。教训:resume 续训时,非 `keep_period` 倍数的 checkpoint 会被自动清理 ——
要保留特定里程碑需设好 `keep_period`。
