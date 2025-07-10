import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from scipy.interpolate import griddata
from data import PinnDataset

def plot_heatmap(x, y, values, title, filename, nx=200, ny=200, cmap='jet'):
    """
    x, y: 坐标向量 (N,)
    values: 对应值向量 (N,)
    """
    # 构造网格
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    X_grid, Y_grid = np.meshgrid(xi, yi)

    # 插值
    Z_grid = griddata(
        points=np.stack([x, y], axis=1),
        values=values,
        xi=(X_grid, Y_grid),
        method='cubic'
    )

    # 画图
    plt.figure(figsize=(8, 4))
    im = plt.imshow(
        Z_grid,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin='lower',
        cmap=cmap,
        aspect='auto'
    )
    plt.colorbar(im, label=title)
    plt.title(f"Heatmap of {title}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    # plt.show()
    print(f"✅ Saved heatmap: {filename}")
def generate_flow_animation(trainer, min_x, max_x, t_vals, save_dir="gif_frames"):
    """
    用训练好的 PINN 模型，预测不同时间点的 u，并绘制热力图，生成 GIF。
    
    参数:
        trainer: 已训练 Trainer 对象
        min_x, max_x: 用于归一化的坐标范围
        t_vals: list of float, 所有需要生成的时间帧
        save_dir: 保存 PNG 帧的文件夹
    """

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成空间网格
    x = np.linspace(0, 2, 100)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    filenames = []

    for t in t_vals:
        t_flat = np.full_like(x_flat, t)
        
        # 构造 shape (N,6)
        data_arr = np.stack([
            t_flat,
            x_flat,
            y_flat,
            np.zeros_like(t_flat),  # p
            np.zeros_like(t_flat),  # u
            np.zeros_like(t_flat),  # v
        ], axis=1)

        # ✅ 封装成 PinnDataset
        from data import PinnDataset
        fake_dataset = PinnDataset(data_arr)
        
        # 模型预测
        outputs = trainer.predict(fake_dataset)
        preds = outputs["preds"].detach().cpu().numpy()
        
        u_pred = preds[:, 1]   # u 分量

        # ✅ 直接调用 plot_heatmap
        fname = os.path.join(save_dir, f"u_pred_heatmap_t_{t:.4f}.png")

        plot_heatmap(
            x_flat,
            y_flat,
            u_pred,
            title=f"Predicted u velocity at t={t:.4f}",
            filename=fname
        )

        filenames.append(fname)

    # 拼接 GIF
    images = [imageio.v2.imread(f) for f in filenames]
    gif_path = os.path.join(save_dir, "flow.gif")
    imageio.mimsave(gif_path, images, duration=0.3)
    print(f"✅ GIF saved: {gif_path}")
