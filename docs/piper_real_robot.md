# Piper 真机部署 & 实验流程

面向 `pi05_piper_pick_and_place` 微调模型的真机运行手册。配套离线评测见
[`piper_offline_eval.md`](piper_offline_eval.md)。

真机运行是**两个进程**:
- **Policy server**(GPU 机):加载 checkpoint,通过 websocket 提供 `infer`。
- **Client**([`scripts/piper_remote_client.py`](../scripts/piper_remote_client.py)):在机器人这端采相机 + 读关节,
  组 observation 发给 server,拿回 action chunk 逐步下发给 Piper。

```
[相机 D435/D405] ─┐
                  ├─> client(piper_remote_client)  ──websocket──>  serve_policy(GPU, ckpt)
[Piper 关节反馈] ─┘         │ 逐步 send_action                              │ sample_actions
                           └<──────────────── action chunk ────────────────┘
```

> ⚠️ **机械臂无机械刹车**。`RobotArm.close()` 会 `disable()` 切电机力矩,机械臂会**因重力下坠**。
> 所有脚本默认**不**在退出时掉电(`--disable-on-exit` 默认 False)。断电前先用手扶稳。

---

## 0. 前置检查

- GPU 机能加载 checkpoint(先跑过离线评测最稳,见 offline_eval 文档)。
- 两台相机接好:global = **D435**,wrist = **D405**(client 按型号名自动找 serial,
  也可用 `--global-camera-serial` / `--wrist-camera-serial` 指定)。
- `bci_piper` 仓库可被找到:client 用 `--bci-piper-root`(默认 `/home/huix/bci_robot/bci_piper`);
  goto_home 会自动探测 `$BCI_PIPER_ROOT` / 同级目录 / Linux 默认路径。
- 确认 server 机 IP。若 server 和 client 同机,用 `--server-host localhost`。

---

## 1. 起 policy server(GPU 机)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_piper_pick_and_place \
  --policy.dir checkpoints/pi05_piper_pick_and_place/piper_pick_pi05_lora_v2/29999 \
```

看到监听 8000、打印 server metadata 即就绪。换 checkpoint 只改 `--policy.dir`。

---

## 2. 回 home 位(每次实验前必做)

策略假设从**训练起始位姿**开始。起点偏了(尤其 joint1/joint2 超出训练范围),模型看到 OOD
状态就会 hedge / 不动。所以先把臂开到 home,再跑 client。
在本地机器上运行：

```bash
python scripts/piper_goto_home.py --real
```

- ~~ home = 全 182 条训练 episode 起止关节的中位姿(脚本里 `DATASET_HOME_RAD`)。~~
- home = [-1.53°, -1.46°, 3.238°, 2.289°, 21.699°, -1.254°]，是收集数据的pipeline的NEUTRAL_JOINT_ANGLES
- `move_j` 是点到点命令,脚本发一次后按 `command_rate_hz`(50Hz)插值刷新到位,`tol_rad`(0.02)判定到达。
- 交互确认可用 `--yes` 跳过。到位后**保持使能**(默认不掉电)。

主要参数:`--speed-percent`(10)、`--max-joint-vel-rad-s`(0.15)、`--arrive-timeout-s`(30)、
`--disable-on-exit`(默认 False,仅在臂被支撑好时才加)。

---

## 3. 跑 client

### 3.1 先 dry-run(强制,验证 pipeline 与安全)

`--dry-run` **必须**配 `--real`:它读真实关节 + 相机、正常调用 server、算出目标关节,
**但不下发 move_j / 夹爪指令**。用来在真动之前确认数值合理。

```bash
python scripts\piper_remote_client.py --server-host 127.0.0.1 --server-port 8000   --bci-piper-root D:\hxf\code\piper\bci_piper   --real   --prompt "Pick up the bottle of calcium tablets and place it in the large container" --control-hz 30 --open-loop-horizon 10 --max-joint-delta-rad 0.05 --action-alpha 1.0 --action-mode absolute --no-disable-gripper --show-preview --print-state-debug --print-every 1
```

默认max_steps是2000，如果debug可以设小一点

看调试输出(`--print-state-debug` 默认开):
- `obs_state` —— 喂给模型的状态(6 关节 + 夹爪)。
- `model_action` —— 模型原始输出。
- `sent_action` —— 经插值/钳制后**将要**下发的目标。dry-run 下不真发。

确认 `sent_action` 每步相对 `obs_state` 平滑、没有乱跳、方向合理后,再真跑。

### 3.2 真跑

去掉 `--dry-run`:

```bash
uv run python scripts/piper_remote_client.py \
  --server-host <SERVER_IP> --server-port 8000 \
  --real \
  --prompt "pick up the object and place it in the basket"
```

`--show-preview`(默认开)会开一个 OpenCV 窗口,上排原始相机、下排喂给模型的 224×224 图,
用来核对相机没插反、画面正常。

---

## 4. 执行逻辑与关键参数

### 4.1 一步是怎么走出来的([`send_action`](../scripts/piper_remote_client.py#L167))

每个控制步:读当前关节 `current`,模型给 `raw_action`(6 关节),按 `action-mode` 求目标:

- **`absolute`(默认)**:`target = current + action_alpha*(raw_action - current)`,再 clip 到
  `current ± max_joint_delta_rad`。→ 向模型目标插值一小步,且每步位移被硬性限幅。
- **`delta`**:`target = current + clip(raw_action, -1, 1) * max_joint_delta_rad`。
  → 把模型输出当增量方向,乘以步长。

拿到 chunk 后按 `open-loop-horizon` 决定执行几步再重新推理(=1 即每步都重新采图推理,闭环)。

### 4.2 参数表([Args](../scripts/piper_remote_client.py#L15))

| 参数 | 默认 | 说明 |
|---|---|---|
| `--server-host` | (必填) | policy server 地址;同机用 `localhost` |
| `--server-port` | 8000 | 与 serve_policy 一致 |
| `--prompt` | "pick up the object and place it in the basket" | 任务指令(训练用 task 文本) |
| `--real` | False | **不加则是假状态(全 0)**,只调通链路,不碰机器 |
| `--dry-run` | False | 需配 `--real`;读真值算目标但不发运动 |
| `--action-mode` | `absolute` | `absolute` 插值到绝对目标 / `delta` 增量 |
| `--action-alpha` | 0.15 | absolute 模式插值系数,越小越保守 |
| `--max-joint-delta-rad` | 0.03 | **每步单关节最大位移(安全上限)**,初期别调大 |
| `--speed-percent` | 10 | 机械臂速度,实验初期保持低 |
| `--control-hz` | 3.0 | 控制频率 |
| `--open-loop-horizon` | 1 | 每次执行 chunk 的前几步(1=全闭环) |
| `--max-steps` | 300 | 本次 rollout 最大步数 |
| `--disable-gripper` | True | **默认锁夹爪**;测抓取需 `--disable-gripper False` |
| `--gripper-open-fraction` | 1.0 | 锁夹爪时的固定开合值 |
| `--freeze-observation` | False | 冻结首帧观测(调试模型对输入的敏感性用) |
| `--show-preview` | True | OpenCV 预览窗 |
| `--disable-on-exit` | False | 退出**不**掉电(防下坠) |
| `--image-size` | 224 | 送模型的图像尺寸 |

---

## 5. 已知硬伤 / 排查顺序

> 这些是部署侧问题,和模型 checkpoint 质量**无关**。重训后如果离线 RMSE 过线但实机仍不行,
> **先回来查这里,别再怀疑训练**。

1. **夹爪状态写死**:`get_state()` 把夹爪那一维恒填 `--gripper-open-fraction`(默认 1.0),
   不读真实夹爪。对 pick-and-place 会给模型 OOD 状态。测抓取时需要接真实夹爪反馈。
2. **夹爪默认禁用**:`--disable-gripper True`。只验证移动轨迹时可以锁;测完整抓取要打开。
   (离线评测里 dim6 夹爪本就是唯一输给"不动"基线的维度,是重灾区,实机重点观察开合时机。)
3. **动作可能被压到近乎不动**:absolute 模式下 `±0.03 rad/step` 钳制 + `alpha=0.15` 会把每步
   位移压得很小。若实机"几乎不动",先确认不是这两个参数把动作吃掉了。
4. **控制频率错配**:`control-hz=3` vs 训练数据 30fps 采集,时间尺度不一致,可能影响 chunk 语义。

排查建议:`--dry-run` 下先只看 `sent_action` 是否被钳制吃光 → 打开夹爪单测开合 →
逐步放宽 `max-joint-delta-rad` / 调 `action-alpha`。

---

## 6. 一次标准实验流程(推荐顺序)

1. **server** 起目标 ckpt(如 v2/29999)。
2. `piper_goto_home.py --real` 回 home。
3. **client `--real --dry-run`**,看 `sent_action` 数值正常、方向合理。
4. 去掉 dry-run,`--disable-gripper True` 先只验证**移动轨迹**靠不靠谱。
5. 轨迹 OK 后 `--disable-gripper False`(并接真实夹爪反馈)测**完整抓取**,重点看夹爪开合时机。
6. 收工前扶稳机械臂再考虑掉电。
