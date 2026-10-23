"""
Hard global budget constraint, shared by every paradigm.

After each model step the predicted content is shifted so the ocean-integrated
content change equals the surface flux integrated over that step. For the
column heat and freshwater content the global budget has no other term, so the
emulator can no longer drift globally, and the global trend follows the
forcing for any run length; the network only decides where the heat and
freshwater go.

The time step is the true month length. Between the centres of months a and b
the flux integral is F_a L_a / 2 + F_b L_b / 2 (L = month length): with a fixed
30-day month the 1970-2018 ACCESS-OM2 heat budget mis-closed by about half of
the OHC change, against about 5% with true month lengths.
"""

import torch
import torch.nn as nn


class GlobalBudgetConstraint(nn.Module):
    """
    forward(pred, previous, previous_index, target_index) -> pred, with every
    enforced budget closed exactly.

    pred, previous   : (B, P, H, W) z-scores: the predicted next state and the
                       state it was predicted from
    previous_index,
    target_index     : (B,) time indices of those two states (their months)

    For each budget, with content C = z * std + mean (physical, per cell):

        R = sum(area * (C_pred - C_previous)) - sign * (G_a L_a + G_b L_b) / 2

    where G is the ocean-integrated surface flux of a month. The correction
    adds k * std_target to C_pred with k = -R / sum(area * std_target): a
    uniform shift of the z-score over the ocean, so it is spread where the
    content varies and never divides by a small std. It is linear in pred, so
    gradients pass through it.
    """

    def __init__(self, budgets, content_std, mean_global, flux_global, month_seconds, area, mask):
        """
        budgets      : [(name, content_channel, flux_channel, flux_sign)]
        content_std  : {name: (T, H, W) std of the content, physical units}
        mean_global  : (T, P) ocean integral of the climatological mean of each prognostic variable
        flux_global  : (T, C) ocean integral of each forcing variable (physical)
        month_seconds: (T,) month lengths in seconds
        area, mask   : (H, W) cell area (m^2) and ocean mask
        """
        super().__init__()
        self.budgets = [(name, int(c), int(f), float(sign)) for name, c, f, sign in budgets]
        ocean_area = (area.double() * mask.double())
        # Non-persistent: rebuilt from the data, so they stay out of checkpoints.
        self.register_buffer("ocean_area", ocean_area, persistent=False)
        self.register_buffer("ocean", (mask > 0).float(), persistent=False)
        self.register_buffer("mean_global", mean_global.double(), persistent=False)
        self.register_buffer("flux_global", flux_global.double(), persistent=False)
        self.register_buffer("month_seconds", month_seconds.double(), persistent=False)
        for name, *_ in self.budgets:
            self.register_buffer(f"{name}_std", content_std[name].float(), persistent=False)

    def residual(self, pred, previous, previous_index, target_index, budget):
        """(B,) budget residual R (content units x m^2) of one budget, and sum(area * std_target)."""
        name, c, f, sign = budget
        std = getattr(self, f"{name}_std")
        std_previous = std[previous_index].double()
        std_target = std[target_index].double()
        area = self.ocean_area
        change = ((pred[:, c].double() * std_target - previous[:, c].double() * std_previous) * area).sum(dim=(-2, -1))
        change = change + self.mean_global[target_index, c] - self.mean_global[previous_index, c]
        flux = sign * 0.5 * (self.flux_global[previous_index, f] * self.month_seconds[previous_index]
                             + self.flux_global[target_index, f] * self.month_seconds[target_index])
        return change - flux, (std_target * area).sum(dim=(-2, -1))

    def forward(self, pred, previous, previous_index, target_index):
        previous_index = previous_index.to(pred.device)
        target_index = target_index.to(pred.device)
        out = pred.clone()
        for budget in self.budgets:
            residual, weight = self.residual(out, previous, previous_index, target_index, budget)
            k = (residual / weight.clamp_min(1e-30)).to(pred.dtype)
            out[:, budget[1]] = out[:, budget[1]] - k[:, None, None] * self.ocean
        return out


def enforced_budgets(cfg):
    """{budget: ClosureConfig} enforced exactly: enforce=True, content predicted, flux in the forcing."""
    d = cfg.data
    return {
        name: c for name, c in cfg.loss.closures.items()
        if c.enforce and c.content_variable in d.prognostic and c.flux_variable in d.forcing
    }


def build_constraint(cfg, data):
    """The GlobalBudgetConstraint for cfg's enforced budgets, or None if there are none."""
    budgets = enforced_budgets(cfg)
    if not budgets:
        return None
    d, f = cfg.data, data.fields
    spec, std = [], {}
    for name, closure in budgets.items():
        c = d.prognostic.index(closure.content_variable)
        spec.append((name, c, d.forcing.index(closure.flux_variable), closure.surface_flux_sign))
        std[name] = f["prognostic_std"][:, c].contiguous()
    return GlobalBudgetConstraint(spec, std, f["prognostic_mean_global"], f["forcing_global"],
                                  f["month_seconds"], f["area"], f["mask"])
