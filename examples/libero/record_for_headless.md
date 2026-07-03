下面是跑通 LIBERO（远程、无头、Docker+GPU）的命令清单，按顺序整理好了。

1. 禁用出问题的 Grafana apt 源（临时）
```bash
sudo mv /etc/apt/sources.list.d/apt_grafana_com.list /etc/apt/sources.list.d/apt_grafana_com.list.disabled
sudo apt-get update
```

2. 安装并配置 Docker 的 NVIDIA runtime
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

3. 验证 Docker 能调用 GPU
```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

4. 无头启动 LIBERO（关键成功命令）
```bash
sudo env SERVER_ARGS="--env LIBERO" MUJOCO_GL=egl docker compose \
  -f examples/libero/compose.yml \
  -f examples/libero/compose.headless.yml \
  up --build
```

5. 可选：小规模测试（先快速验证）
```bash
sudo env SERVER_ARGS="--env LIBERO" CLIENT_ARGS="--args.num-trials-per-task 2" MUJOCO_GL=egl docker compose \
  -f examples/libero/compose.yml \
  -f examples/libero/compose.headless.yml \
  up --build
```

6. 可选：查看服务端日志排错
```bash
sudo docker compose -f examples/libero/compose.yml -f examples/libero/compose.headless.yml logs --tail=300 openpi_server
```

7. 可选：查看无头生成的视频回放
```bash
ls -lh data/libero/videos | tail -n 20
```

## 一键脚本（已添加）

脚本路径：`examples/libero/run_libero_headless.sh`

默认完整跑：
```bash
bash examples/libero/run_libero_headless.sh
```

快速验证（默认 `--args.num-trials-per-task 2`）：
```bash
bash examples/libero/run_libero_headless.sh --quick
```

自定义参数示例：
```bash
SERVER_ARGS="--env LIBERO" \
CLIENT_ARGS="--args.task-suite-name libero_10 --args.num-trials-per-task 5" \
MUJOCO_GL=egl \
bash examples/libero/run_libero_headless.sh
```
