import torch
from torch import nn, autograd, Tensor
from torch.nn import functional as F


def calc_grad(y, x) -> Tensor:
    grad = autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad


class FfnBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inter_dim = 4 * dim
        self.fc1 = nn.Linear(dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, dim)
        self.act_fn = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x0 = x
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + x0


class Pinn(nn.Module):
    """
    `forward`: returns a tensor of shape (D, 3), where D is the number of
    data points, and the 2nd dim. is the predicted values of p, u, v.
    """

    def __init__(self, min_x: int, max_x: int, min_y: int, max_y: int, min_t: int, max_t: int):
        super().__init__()

        self.MIN_X = min_x
        self.MAX_X = max_x
        self.MIN_Y = min_y
        self.MAX_Y = max_y
        self.MIN_T = min_t
        self.MAX_T = max_t
        # Build FFN network
        self.hidden_dim = 128
        self.num_blocks = 8
        self.first_map = nn.Linear(3, self.hidden_dim)
        self.last_map = nn.Linear(self.hidden_dim, 2)
        self.ffn_blocks = nn.ModuleList([
            FfnBlock(self.hidden_dim) for _ in range(self.num_blocks)
        ])

        self.lambda1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.lambda2 = nn.Parameter(torch.tensor(0.01), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def ffn(self, inputs: Tensor) -> Tensor:
        x = self.first_map(inputs)
        for blk in self.ffn_blocks:
            x = blk(x)
        x = self.last_map(x)
        return x

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        p: Tensor = None,
        u: Tensor = None,
        v: Tensor = None,
    ):
        """
        All shapes are (b,)

        inputs: x, y, t
        labels: p, u, v
        """
        t1 = 2.0 * (t - self.MIN_T) / (self.MAX_T - self.MIN_T) - 1.0
        x1 = 2.0 * (x - self.MIN_X) / (self.MAX_X - self.MIN_X) - 1.0
        y1 = 2.0 * (y - self.MIN_Y) / (self.MAX_Y - self.MIN_Y) - 1.0
        inputs = torch.stack([x1, y1, t1], dim=1)
        # inputs = 2.0 * (inputs - self.MIN_X) / (self.MAX_X - self.MIN_X) - 1.0

        hidden_output = self.ffn(inputs)
        psi = hidden_output[:, 0]
        p_pred = hidden_output[:, 1]
        u_pred = calc_grad(psi, y)
        v_pred = -calc_grad(psi, x)

        preds = torch.stack([p_pred, u_pred, v_pred], dim=1)
        u_t = calc_grad(u_pred, t)
        u_x = calc_grad(u_pred, x)
        u_y = calc_grad(u_pred, y)
        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)

        v_t = calc_grad(v_pred, t)
        v_x = calc_grad(v_pred, x)
        v_y = calc_grad(v_pred, y)
        v_xx = calc_grad(v_x, x)
        v_yy = calc_grad(v_y, y)

        p_x = calc_grad(p_pred, x)
        p_y = calc_grad(p_pred, y)

        # This is the original implementation (I think this is incorrect)
        f_u = (
            u_t
            + self.lambda1 * (u_pred * u_x + v_pred * u_y)
            + p_x
            - self.lambda2 * (u_xx + u_yy)
        )
        f_v = (
            v_t
            + self.lambda1 * (u_pred * v_x + v_pred * v_y)
            + p_y
            - self.lambda2 * (v_xx + v_yy)
        )

        # Corrected
        # f_u = (
        #     self.lambda1 * (u_t + u_pred * u_x + v_pred * u_y)
        #     + p_x
        #     - self.lambda2 * (u_xx + u_yy)
        # )
        # f_v = (
        #     self.lambda1 * (v_t + u_pred * v_x + v_pred * v_y)
        #     # - self.lambda1 * 9.81
        #     + p_y
        #     - self.lambda2 * (v_xx + v_yy)
        # )

        loss, losses = self.loss_fn(
        x, y, t,
        u, v,
        u_pred, v_pred,
        f_u, f_v
         )
        return {
            "preds": preds,
            "loss": loss,
            "losses": losses,
        }

    def loss_fn(
        self,
        x, y, t,
        u, v,
        u_pred, v_pred,
        f_u_pred, f_v_pred,
    ):
        """
        Compute total loss including IC and BC losses,
        directly from the same training batch.
        """

        # ----------------------------
        # PDE residual loss
        # ----------------------------
        u_loss = F.mse_loss(u_pred, u)
        v_loss = F.mse_loss(v_pred, v)
        f_u_loss = F.mse_loss(f_u_pred, torch.zeros_like(f_u_pred))
        f_v_loss = F.mse_loss(f_v_pred, torch.zeros_like(f_v_pred))

        # ----------------------------
        # Initial Condition Loss
        # ----------------------------
        # mask_ic = (t == 0.0)
        # ic_u_loss = torch.tensor(0.0, device=u.device)
        # ic_v_loss = torch.tensor(0.0, device=u.device)
        # if mask_ic.any():
        #     ic_u_loss = F.mse_loss(u_pred[mask_ic], u[mask_ic])
        #     ic_v_loss = F.mse_loss(v_pred[mask_ic], v[mask_ic])

        # ----------------------------
        # Boundary Condition Loss
        # ----------------------------
        # eps = 1e-5
        # mask_bc = (
        #     (torch.isclose(x, torch.tensor(1.0, device=x.device), atol=eps)) |
        #     (torch.isclose(x, torch.tensor(8.0, device=x.device), atol=eps)) |
        #     (torch.isclose(y, torch.tensor(-2.0, device=x.device), atol=eps)) |
        #     (torch.isclose(y, torch.tensor(2.0, device=x.device), atol=eps))
        # )
        # bc_u_loss = torch.tensor(0.0, device=u.device)
        # bc_v_loss = torch.tensor(0.0, device=u.device)
        # if mask_bc.any():
        #     bc_u_loss = F.mse_loss(u_pred[mask_bc], u[mask_bc])
        #     bc_v_loss = F.mse_loss(v_pred[mask_bc], v[mask_bc])

        # ----------------------------
        # total loss
        # ----------------------------
        loss = (
            u_loss + v_loss +
            f_u_loss + f_v_loss
            # + ic_u_loss + ic_v_loss +
            # bc_u_loss + bc_v_loss
        )

        return loss, {
            "u_loss": u_loss,
            "v_loss": v_loss,
            "f_u_loss": f_u_loss,
            "f_v_loss": f_v_loss,
            # "ic_u_loss": ic_u_loss,
            # "ic_v_loss": ic_v_loss,
            # "bc_u_loss": bc_u_loss,
            # "bc_v_loss": bc_v_loss,
        }

