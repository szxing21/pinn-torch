import os
import numpy as np
import torch
from data import PinnDataset
from extract_cylinder_data import plot_pinn_frames
import pdb
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# %%
def generate_flow_animation(
    trainer,
    min_x,
    max_x,
    t_vals,
    save_dir="gif_frames"
):
    """
    用训练好的 PINN 模型预测不同时间点流场，并绘制 GIF。
    
    保留旧版 generate_flow_animation 的 predict 逻辑，
    不直接调用 model.forward。
    """

    os.makedirs(save_dir, exist_ok=True)

    # 生成空间网格
    x = np.linspace(1, 8, 100)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    all_data_list = []

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
        # pdb.set_trace()
        # 封装成 PinnDataset
        fake_dataset = PinnDataset(data_arr)
# %%
        # 用 trainer.predict
        with torch.enable_grad():
            outputs = trainer.predict(fake_dataset)

        preds = outputs["preds"].detach().cpu().numpy()

        # 从预测结果提取
        p_pred = preds[:, 0]
        u_pred = preds[:, 1]
        v_pred = preds[:, 2]

        # 拼成 (N, 6) 数组
        pred_arr = np.stack([
            t_flat,
            x_flat,
            y_flat,
            p_pred,
            u_pred,
            v_pred
        ], axis=1)

        all_data_list.append(pred_arr)

    # 合并所有帧
    final_arr = np.concatenate(all_data_list, axis=0)

    # 封装成 PinnDataset
    final_dataset = PinnDataset(final_arr)

    # 调用 plot_pinn_frames
    plot_pinn_frames(
        train_data=final_dataset,
        view_mode="interp",
        total_frames=len(t_vals),
        out_dir=save_dir,
        gif_name=os.path.join(save_dir, "flow.gif")
    )

# %%
