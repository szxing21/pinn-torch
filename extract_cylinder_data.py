import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
import argparse
import matplotlib.tri as mtri
from matplotlib.patches import Circle
import json
import re

from data import get_orig_dataset

# 防止中文乱码
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# OpenMP workaround
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_loss(root_dir="result", output_path="loss_curve.png"):
    """
    从 root_dir 下递归查找所有 loss_history.json 文件，
    将它们拼接成一条连续 loss 曲线。
    """

    # 存储 (epoch_index, path) 元组
    all_files = []

    # 搜索所有 loss_history.json
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f == "loss_history.json":
                full_path = os.path.join(dirpath, f)
                # 提取 epoch index
                match = re.search(r"ckpt-(\d+)", dirpath)
                if match:
                    epoch_idx = int(match.group(1))
                else:
                    epoch_idx = -1
                all_files.append( (epoch_idx, full_path) )

    if not all_files:
        print("没有找到任何 loss_history.json 文件！")
        return

    # 按 epoch index 排序
    all_files.sort()

    # 拼接所有 loss
    full_loss = []
    for epoch_idx, file_path in all_files:
        with open(file_path, "r") as f:
            losses = json.load(f)

        # 确保是 list
        if not isinstance(losses, list):
            print(f"不是 list，跳过：{file_path}")
            continue

        full_loss.extend(losses)

    # 画曲线
    if full_loss:
        plt.figure(figsize=(10,5))
        plt.plot(range(1, len(full_loss)+1), full_loss, color='b')
        plt.xlabel("Step (across all epochs)")
        plt.ylabel("Loss")
        plt.yscale("log")     # 如果跨度太大可以留着
        plt.grid(True)
        plt.title("Concatenated Loss Curve")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        print(f"拼接后 loss 曲线已保存: {output_path}")
    else:
        print("loss 是空的，未生成曲线。")
def plot_pinn_frames(
    train_data,
    view_mode="interp",
    total_frames=20,
    out_dir="frames_interp",
    gif_name="cylinder_flow_interp.gif"
):
    """
    根据 PINN 数据集绘制若干帧图像，并生成 GIF。

    Parameters
    ----------
    train_data : PinnDataset
        pinn-torch 中的 train_dataset 对象
    view_mode : str
        'scatter' 或 'interp'
    total_frames : int
        要绘制的帧数
    out_dir : str
        帧图像输出目录
    gif_name : str
        最终合成的 GIF 名称
    """

    arr = train_data.examples.detach().cpu().numpy()  # (N*T, 6): t,x,y,p,u,v

    # 提取所有 unique t
    unique_ts = np.unique(arr[:, 0])
    TOTAL = len(unique_ts)
    print("数据中总帧数:", TOTAL)

    # 圆柱参数
    xc, yc, r = 0.2, 0.0, 0.05

    # 全局 min/max
    u_min, u_max = arr[:, 4].min(), arr[:, 4].max()
    v_min, v_max = arr[:, 5].min(), arr[:, 5].max()
    p_min, p_max = arr[:, 3].min(), arr[:, 3].max()

    field_vminmax = {
        "u velocity": (u_min, u_max),
        "v velocity": (v_min, v_max),
        "pressure": (p_min, p_max)
    }

    # 全局 contour levels
    u_levels = np.linspace(u_min, u_max, 50)
    v_levels = np.linspace(v_min, v_max, 50)
    p_levels = np.linspace(p_min, p_max, 50)

    field_levels = {
        "u velocity": u_levels,
        "v velocity": v_levels,
        "pressure": p_levels
    }

    # 规则网格 (interp 用)
    unique_xs = np.unique(arr[:, 1])
    unique_ys = np.unique(arr[:, 2])
    X, Y = np.meshgrid(unique_xs, unique_ys)

    # 输出目录
    os.makedirs(out_dir, exist_ok=True)
    frame_files = []

    # 坐标范围
    xlim = (0, 8)
    ylim = (-2, 2)

    for frame_idx in range(min(TOTAL, total_frames)):
        t_val = unique_ts[frame_idx]
        mask = np.isclose(arr[:, 0], t_val, atol=1e-8)
        data_f = arr[mask]

        if data_f.shape[0] == 0:
            print(f"frame {frame_idx} no data, skipped")
            continue

        x_f, y_f = data_f[:, 1], data_f[:, 2]
        p_f = data_f[:, 3]
        u_f, v_f = data_f[:, 4], data_f[:, 5]

        if view_mode == "interp":
            u_grid = np.full_like(X, np.nan)
            v_grid = np.full_like(X, np.nan)
            p_grid = np.full_like(X, np.nan)
            for x_pt, y_pt, p_pt, u_pt, v_pt in data_f[:, 1:]:
                i = np.searchsorted(unique_xs, x_pt)
                j = np.searchsorted(unique_ys, y_pt)
                u_grid[j, i] = u_pt
                v_grid[j, i] = v_pt
                p_grid[j, i] = p_pt

        fig, axes = plt.subplots(
            3, 1,
            figsize=(6, 12),
            sharex=True,
            sharey=True,
            gridspec_kw={'hspace': 0.2}
        )

        for ax, name in zip(axes, ["u velocity", "v velocity", "pressure"]):
            vmin, vmax = field_vminmax[name]
            levels = field_levels[name]

            if view_mode == "scatter":
                field = {"u velocity": u_f,
                         "v velocity": v_f,
                         "pressure": p_f}[name]
                im = ax.scatter(
                    x_f, y_f,
                    c=field,
                    s=5,
                    cmap="jet",
                    vmin=vmin,
                    vmax=vmax
                )
            else:
                field_grid = {"u velocity": u_grid,
                              "v velocity": v_grid,
                              "pressure": p_grid}[name]
                im = ax.contourf(
                    X, Y, field_grid,
                    levels=levels,
                    cmap="jet",
                    vmin=vmin,
                    vmax=vmax
                )
                # ax.contour(
                #     X, Y, field_grid,
                #     levels=10,
                #     colors="k",
                #     linewidths=0.5
                # )

            circ = Circle((xc, yc), r, edgecolor='k', fill=False, lw=2)
            ax.add_patch(circ)

            ax.set_title(f"{name}, frame={frame_idx}")
            ax.set_aspect("equal")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

        fn = os.path.join(out_dir, f"frame_{frame_idx:04d}.png")
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        frame_files.append(fn)
        if frame_idx % 10 == 0:
            print(f"saved frame {frame_idx}/{total_frames}")

    print("Creating GIF...")
    with imageio.get_writer(gif_name, mode="I", duration=0.1) as writer:
        for fn in frame_files:
            writer.append_data(imageio.imread(fn))
    print(f"GIF done: {gif_name}")


# ===============================
# 当脚本直接运行时，自动执行
# ===============================
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        mode = "interp"
        frames = 200
        print(f"Running in IDE mode → mode={mode}, frames={frames}")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--mode",
            choices=["scatter", "interp"],
            default="interp",
            help="绘图模式：'scatter' 散点, 'interp' 三角剖分等值面"
        )
        parser.add_argument(
            "--frames",
            type=int,
            default=200,
            help="绘制前多少帧（默认20）"
        )
        args = parser.parse_args()
        mode = args.mode
        frames = args.frames

    train_dataset, _, _, _, _, _, _, _ = get_orig_dataset()

    plot_pinn_frames(
        train_data=train_dataset,
        view_mode=mode,
        total_frames=frames,
        out_dir=f"frames_{mode}",
        gif_name=f"cylinder_flow_{mode}.gif"
    )
