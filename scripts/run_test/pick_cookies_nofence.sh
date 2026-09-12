# 诊断用：与 pick_cookies.sh 完全相同，只去掉 --workspace-box 和 --workspace-floor-margin-m。
# 二者都不传时 _apply_workspace_safety() 会直接 return target，整个围栏（含桌面高度保护）短路。
# 危险：没有任何笛卡尔空间保护，手必须放在急停上。确认无误后请改回 pick_cookies.sh。
python scripts\\piper_remote_client.py --server-host 127.0.0.1 --server-port 8000   --bci-piper-root D:\hxf\code\piper\bci_piper   --real   --prompt "Pick up the cookies and place it in the large container" --control-hz 30 --open-loop-horizon 10 --max-joint-delta-rad 0.15 --action-alpha 1.0 --speed-percent 10 --action-mode absolute --no-disable-gripper --show-preview --print-state-debug --print-every 1 --max-steps 1800 --participant-id SYSTEM --session-id S002 --method-id M01 --task-id T03 "$@"
