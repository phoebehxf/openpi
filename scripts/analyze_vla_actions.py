from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    src = Path("vla_actions.csv")
    out = Path("outputs/vla_actions_analysis")
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    cols = [f"a{i}" for i in range(7)]
    joint_cols = cols[:6]
    executed = df[df["source"].eq("executed")].copy().sort_values(["step"])
    chunks = df[~df["source"].eq("executed")].copy().sort_values(["step", "chunk_index"])

    stats_lines = [
        f"rows={len(df)} executed_rows={len(executed)} chunk_preview_rows={len(chunks)}",
        "sources=" + ", ".join(f"{k}:{v}" for k, v in df["source"].value_counts().items()),
        f"step_range={int(df.step.min())}..{int(df.step.max())}",
    ]

    if not executed.empty:
        deltas = executed[cols].diff().dropna()
        joint_deltas = deltas[joint_cols]
        l2 = np.linalg.norm(joint_deltas.to_numpy(), axis=1)
        stats_lines.append(
            f"executed_delta_l2 mean={l2.mean():.6f} p95={np.percentile(l2, 95):.6f} max={l2.max():.6f}"
        )
        for col in cols:
            stats_lines.append(
                f"{col}: min={executed[col].min():.6f} max={executed[col].max():.6f} "
                f"mean={executed[col].mean():.6f} "
                f"step_delta_abs_mean={deltas[col].abs().mean():.6f} "
                f"step_delta_abs_max={deltas[col].abs().max():.6f}"
            )

    boundary_rows = []
    for step, group in chunks.groupby("step"):
        prev_exec = executed[executed["step"].lt(step)].tail(1)
        if prev_exec.empty:
            continue
        start_idx = int(group["chunk_index"].min())
        start_action = group[group["chunk_index"].eq(start_idx)].iloc[0]
        prev_action = prev_exec.iloc[0]
        delta = start_action[cols].to_numpy(dtype=float) - prev_action[cols].to_numpy(dtype=float)
        boundary_rows.append(
            {
                "step": int(step),
                "source": str(group["source"].iloc[0]),
                "start_idx": start_idx,
                "joint_l2_jump": float(np.linalg.norm(delta[:6])),
                **{f"delta_{col}": float(delta[i]) for i, col in enumerate(cols)},
            }
        )

    boundary = pd.DataFrame(boundary_rows)
    if not boundary.empty:
        boundary.to_csv(out / "boundary_jumps.csv", index=False)
        stats_lines.append(
            f"boundary_jumps count={len(boundary)} "
            f"joint_l2 mean={boundary.joint_l2_jump.mean():.6f} "
            f"p95={np.percentile(boundary.joint_l2_jump, 95):.6f} "
            f"max={boundary.joint_l2_jump.max():.6f}"
        )

    (out / "summary.txt").write_text("\n".join(stats_lines), encoding="utf-8")

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    axes = axes.ravel()
    for i in range(6):
        ax = axes[i]
        ax.plot(executed["step"], executed[f"a{i}"], linewidth=1.2)
        ax.set_title(f"a{i} joint action")
        ax.set_ylabel("action")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("control step")
    fig.suptitle("Executed VLA Actions: joints a0-a5", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "executed_joints.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(executed["step"], executed["a6"], linewidth=1.2, color="tab:green")
    ax.set_title("Executed VLA gripper action a6")
    ax.set_xlabel("control step")
    ax.set_ylabel("gripper")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "executed_gripper.png", dpi=160)
    plt.close(fig)

    if len(executed) > 1:
        deltas = executed[cols].diff()
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        for i in range(6):
            axes[0].plot(executed["step"], deltas[f"a{i}"], linewidth=0.9, label=f"a{i}")
        axes[0].set_title("Executed per-step joint deltas")
        axes[0].set_ylabel("delta")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(ncol=6, fontsize=8)
        joint_l2 = np.linalg.norm(deltas[joint_cols].fillna(0).to_numpy(), axis=1)
        axes[1].plot(executed["step"], joint_l2, linewidth=1.0, color="tab:red")
        axes[1].set_title("Executed per-step joint delta L2")
        axes[1].set_xlabel("control step")
        axes[1].set_ylabel("L2")
        axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "executed_deltas.png", dpi=160)
        plt.close(fig)

    if not boundary.empty:
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(boundary["step"], boundary["joint_l2_jump"], marker="o", markersize=3, linewidth=1.0)
        ax.set_title("Chunk boundary jump: previous executed action -> new chunk selected action")
        ax.set_xlabel("swap step")
        ax.set_ylabel("joint L2 jump")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "boundary_jumps.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=False)
    axes = axes.ravel()
    preview_steps = list(chunks["step"].drop_duplicates().head(12))
    for i in range(6):
        ax = axes[i]
        ax.plot(executed["step"], executed[f"a{i}"], color="black", linewidth=1.0, alpha=0.6, label="executed")
        for step in preview_steps:
            group = chunks[chunks["step"].eq(step)].sort_values("chunk_index")
            xs = step + (group["chunk_index"] - group["chunk_index"].min())
            ax.plot(xs, group[f"a{i}"], linewidth=1.0, alpha=0.55)
        ax.set_title(f"a{i}: executed plus preview chunks")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("First preview chunks overlaid on executed timeline", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "chunk_previews_overlay.png", dpi=160)
    plt.close(fig)

    print("\n".join(stats_lines))
    print(f"OUTPUT_DIR={out.resolve()}")


if __name__ == "__main__":
    main()
