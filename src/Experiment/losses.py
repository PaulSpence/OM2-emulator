"""
Loss construction.

``build_losses(cfg, data)`` turns ``cfg.loss.terms`` into the list of callables
that ``Emulator.total_rollout_loss`` evaluates at every rollout step. The loss
functions themselves live in src/Emulator/om2_loss_functions.py.

Weights: the weight of term k at rollout step s is

    terms[k]      (a number, or a list with one absolute weight per step)
  x step_weights[s] / mean(step_weights)      (optional, shared by all terms)

so ``step_weights`` shifts emphasis between lead times without changing the
overall size of the loss.
"""

from Emulator import budget_closure_loss, local_mse_loss, spectral_loss

from .config import CLOSURE_SUFFIX, _is_active


def closure_loss(cfg, budget):
    """The unit-weight closure callable for cfg.loss.closures[budget]."""
    d, closure = cfg.data, cfg.loss.closures[budget]
    return budget_closure_loss(
        weight=1.0,
        budget=budget,
        content_channel_index=d.prognostic.index(closure.content_variable),
        flux_channel_index=d.forcing.index(closure.flux_variable),
        surface_flux_sign=closure.surface_flux_sign,
        min_scale=closure.min_scale,
    )


def closure_std_fields(cfg, data):
    """
    {budget: (content_std, flux_std)} for every active closure, each (T, H, W)
    in physical units: the rollout context's ``closure_std``.
    """
    d, f = cfg.data, data.fields
    return {
        budget: (
            f["prognostic_std"][:, d.prognostic.index(closure.content_variable)],
            f["forcing_std"][:, d.forcing.index(closure.flux_variable)],
        )
        for budget, closure in cfg.active_closures().items()
    }


# Fitting terms by name: (cfg, data) -> callable(**rollout_context) with unit
# weight. "<budget>_closure" terms are built by closure_loss.
LOSS_TERMS = {
    "local_mse": lambda cfg, data: local_mse_loss(weight=1.0),
    "spectral": lambda cfg, data: spectral_loss(weight=1.0),
}


def _term(cfg, data, name):
    if name.endswith(CLOSURE_SUFFIX):
        return closure_loss(cfg, name[: -len(CLOSURE_SUFFIX)])
    return LOSS_TERMS[name](cfg, data)


class WeightedLossTerm:
    """
    A loss term with per-step weights, plus a running total for logging.

    ``running`` accumulates the weighted value (detached) over one rollout; the
    Lightning module reads and resets it after every batch.
    """

    def __init__(self, name, fn, step_weights):
        self.name = name
        self.fn = fn
        self.step_weights = [float(w) for w in step_weights]
        self.running = None

    def __call__(self, **context):
        weight = self.step_weights[context["rollout_step"]]
        if weight == 0.0:
            return context["pred_t"].new_zeros(())
        value = weight * self.fn(**context)
        detached = value.detach()
        self.running = detached if self.running is None else self.running + detached
        return value

    def pop_running(self):
        value, self.running = self.running, None
        return value

    def __repr__(self):
        return f"WeightedLossTerm({self.name!r}, step_weights={self.step_weights})"


def build_losses(cfg, data):
    """List of WeightedLossTerm for every active term in cfg.loss.terms."""
    lo, n_steps = cfg.loss, cfg.window.posterior_steps
    if lo.step_weights is None:
        relative = [1.0] * n_steps
    else:
        mean = sum(lo.step_weights) / n_steps
        relative = [s / mean for s in lo.step_weights]

    terms = []
    for name, weight in lo.terms.items():
        if not _is_active(weight):
            continue
        per_step = list(weight) if isinstance(weight, (list, tuple)) else [weight] * n_steps
        terms.append(WeightedLossTerm(name, _term(cfg, data, name), [w * r for w, r in zip(per_step, relative)]))
    return terms
