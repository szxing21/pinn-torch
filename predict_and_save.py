import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

from model import Pinn
from data import get_orig_dataset
from trainer import Trainer
from scipy.interpolate import griddata

def interpolate_to_grid(x, y, values, nx=300, ny=300, method="cubic"):
    """
    将散点数据插值到网格，用于热力图绘制。

    参数:
        x, y: 坐标数组 (N,)
        values: 值数组 (N,)
        nx, ny: 网格分辨率
        method: 插值方法，可选 'linear' 或 'cubic'

    返回:
        X_grid, Y_grid, Z_grid
    """
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    X_grid, Y_grid = np.meshgrid(xi, yi)
    
    Z_grid = griddata(
        points=np.stack([x, y], axis=1),
        values=values,
        xi=(X_grid, Y_grid),
        method=method
    )
    
    return X_grid, Y_grid, Z_grid
def plot_comparison(X, Y, Z_true, Z_pred, title, filename):
    """
    绘制真值、预测值、误差三幅热力图。

    参数:
        X, Y: 网格坐标
        Z_true: 真值
        Z_pred: 预测值
        title: 标题前缀 (str)
        filename: 保存文件名 (str)
    """
    Z_err = Z_pred - Z_true

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    im = axs[0].imshow(
        Z_true,
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        origin='lower',
        cmap='jet'
    )
    axs[0].set_title(f"True {title}")
    plt.colorbar(im, ax=axs[0])

    im = axs[1].imshow(
        Z_pred,
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        origin='lower',
        cmap='jet'
    )
    axs[1].set_title(f"Predicted {title}")
    plt.colorbar(im, ax=axs[1])

    im = axs[2].imshow(
        Z_err,
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        origin='lower',
        cmap='seismic',
        vmin=-np.max(np.abs(Z_err)),
        vmax=np.max(np.abs(Z_err))
    )
    axs[2].set_title(f"Error in {title}")
    plt.colorbar(im, ax=axs[2])

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"✅ Saved heatmap: {filename}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== 1. Load dataset =====
train_data, test_data, min_x, max_x = get_orig_dataset()

# Convert test_data.data to numpy array
test_arr = test_data.data

t = test_arr[:, 0]
x = test_arr[:, 1]
y = test_arr[:, 2]
p_true = test_arr[:, 3]
u_true = test_arr[:, 4]
v_true = test_arr[:, 5]

# ===== 2. Create model =====
model = Pinn(min_x, max_x).to(device)

# Load checkpoint
ckpt_path = Path("result/pinn-large-tanh-bs128-lr0.005-lrstep1-lrgamma0.8-epoch20/ckpt-39/ckpt.pt")
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ===== 3. Prepare inputs =====
t_tensor = torch.tensor(t, dtype=torch.float32).to(device)
x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

x_tensor.requires_grad_()
y_tensor.requires_grad_()
t_tensor.requires_grad_()


outputs = model(
    x_tensor,
    y_tensor,
    t_tensor,
    torch.tensor(p_true, dtype=torch.float32).to(device),
    torch.tensor(u_true, dtype=torch.float32).to(device),
    torch.tensor(v_true, dtype=torch.float32).to(device),
)
preds = outputs["preds"]

preds = preds.detach().cpu().numpy()

p_pred = preds[:, 0]
u_pred = preds[:, 1]
v_pred = preds[:, 2]

# ===== 4. Save predictions to .mat =====
scipy.io.savemat(
    "predicted_result.mat",
    {
        "X": np.stack([x, y, t], axis=1),
        "p_pred": p_pred,
        "u_pred": u_pred,
        "v_pred": v_pred,
    }
)
print("✅ Saved predicted_result.mat")

# ===== 5. Compute errors =====
err_u = np.mean(np.abs((u_true - u_pred) / u_true))
err_v = np.mean(np.abs((v_true - v_pred) / v_true))
err_p = np.mean(np.abs((p_true - p_pred) / p_true))

lambda1 = model.lambda1.item()
lambda2 = model.lambda2.item()

err_lambda1 = np.abs(lambda1 - 1.0)
err_lambda2 = np.abs(lambda2 - 0.01) / 0.01

print(f"lambda 1: {lambda1:.4f}, error: {err_lambda1:.2f}")
print(f"lambda 2: {lambda2:.6f}, error: {err_lambda2:.2f}")
print(f"Error in u: {err_u:.4e}")
print(f"Error in v: {err_v:.4e}")
print(f"Error in pressure: {err_p:.4e}")

# ===== 6. Plot results =====

# 可视化哪个时间
t_unique = np.unique(t)
print("可用 t 值:", t_unique)

# 假设想选 t = 0.02
target_t = 1

# 找最近的 t
t_slice = t_unique[np.argmin(np.abs(t_unique - target_t))]
print(f"✅ 实际选用 t = {t_slice}")

# Apply mask
mask = (t == t_slice)

if mask.sum() == 0:
    print(f"⚠️ 没有找到 t = {t_slice} 的数据！")
    exit()

x_sel = x[mask]
y_sel = y[mask]

# u
u_true_sel = u_true[mask]
u_pred_sel = u_pred[mask]

X_grid, Y_grid, Z_u_true = interpolate_to_grid(x_sel, y_sel, u_true_sel)
_, _, Z_u_pred = interpolate_to_grid(x_sel, y_sel, u_pred_sel)
plot_comparison(X_grid, Y_grid, Z_u_true, Z_u_pred, title=f"u velocity at t={t_slice}", filename=f"compare_u_t{t_slice}.png")

# v
v_true_sel = v_true[mask]
v_pred_sel = v_pred[mask]

X_grid, Y_grid, Z_v_true = interpolate_to_grid(x_sel, y_sel, v_true_sel)
_, _, Z_v_pred = interpolate_to_grid(x_sel, y_sel, v_pred_sel)
plot_comparison(X_grid, Y_grid, Z_v_true, Z_v_pred, title=f"v velocity at t={t_slice}", filename=f"compare_v_t{t_slice}.png")

# p
p_true_sel = p_true[mask]
p_pred_sel = p_pred[mask]

X_grid, Y_grid, Z_p_true = interpolate_to_grid(x_sel, y_sel, p_true_sel)
_, _, Z_p_pred = interpolate_to_grid(x_sel, y_sel, p_pred_sel)
plot_comparison(X_grid, Y_grid, Z_p_true, Z_p_pred, title=f"pressure at t={t_slice}", filename=f"compare_p_t{t_slice}.png")

