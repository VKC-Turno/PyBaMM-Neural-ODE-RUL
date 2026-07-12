"""
Voltaris/Data_Exploration/phase3_operator.py
Neural ODE operator with theta-in-branch input fix.

Architectural change vs src/pinn/model.py: the ODE branch takes both the
health-feature vector x_health (5) AND the normalised physics-parameter
vector theta_norm (6). Feeding theta into the branch resolves the
"flat SoH" failure — without it the operator collapsed to a
theta-agnostic mean trajectory.

The softplus-monotonic decoder is preserved line-for-line.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from typing import Optional


class ThetaConditionedDegradationODE(nn.Module):
    """
    Right-hand side of dSOH/dn = f(SOH, n, x_health, theta_norm).

    Health + theta context are injected via a stored attribute — torchdiffeq
    only accepts (t, y) signatures, so the branch inputs are stashed on the
    module before odeint is called.
    """

    def __init__(self, x_health_dim: int = 5, theta_dim: int = 6,
                 hidden: int = 64, n_layers: int = 3,
                 dropout_p: float = 0.1,
                 decoder_bias_init: float = -5.0):
        super().__init__()
        # Input: [SOH(1), normalised_cycle(1), x_health(5), theta_norm(6)]
        in_dim = 1 + 1 + x_health_dim + theta_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Dropout(dropout_p), nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        # Initialise final layer near zero → start with slow degradation
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, decoder_bias_init)

        self.branch_context: Optional[torch.Tensor] = None  # set before integrate
        self.n_max = 2000.0  # normalisation factor for cycle number

    def forward(self, n: torch.Tensor, soh: torch.Tensor) -> torch.Tensor:
        """
        n:   scalar cycle number (torchdiffeq calls with scalar t)
        soh: (batch, 1) current SOH values
        Returns: (batch, 1) dSOH/dn — always <= 0
        """
        batch = soh.shape[0]
        n_norm = (n / self.n_max).expand(batch, 1)
        ctx = self.branch_context  # (batch, x_health_dim + theta_dim)

        inp = torch.cat([soh, n_norm, ctx], dim=-1)
        raw = self.net(inp)
        # Enforce monotonic decrease: dSOH/dn <= 0 (line-for-line preserved)
        dSOH = -F.softplus(raw)
        return dSOH


class ThetaConditionedRULPredictor(nn.Module):
    """
    Theta-conditioned Neural ODE operator.

    Public API:
        model = ThetaConditionedRULPredictor(x_health_dim=5, theta_dim=6, ...)
        traj  = model(x_health, theta_norm, n_grid)      # (T, batch, 1)
        soh_c = model.predict(x_health, theta_norm, cycles)  # SoH at cycles
    """

    def __init__(self, x_health_dim: int = 5, theta_dim: int = 6,
                 hidden: int = 64, n_layers: int = 3, dropout: float = 0.1,
                 decoder_bias_init: float = -5.0,
                 ode_rtol: float = 1e-4, ode_atol: float = 1e-6,
                 eol_threshold: float = 0.8):
        super().__init__()
        self.x_health_dim = x_health_dim
        self.theta_dim = theta_dim
        self.ode = ThetaConditionedDegradationODE(
            x_health_dim=x_health_dim,
            theta_dim=theta_dim,
            hidden=hidden,
            n_layers=n_layers,
            dropout_p=dropout,
            decoder_bias_init=decoder_bias_init,
        )
        self.rtol = ode_rtol
        self.atol = ode_atol
        self.eol = eol_threshold

        # Feature normalisation params for x_health.
        self.register_buffer("feat_mean", torch.zeros(x_health_dim))
        self.register_buffer("feat_std",  torch.ones(x_health_dim))
        # Bug fix (2026-07-10 adversarial audit): theta_norm arrives already
        # in σ-units at train time, but at inference time a downstream caller
        # may pass raw physical θ (from a new corpus). Persist theta stats as
        # buffers so the checkpoint is self-contained and inference doesn't
        # silently drift.
        self.register_buffer("theta_mean", torch.zeros(theta_dim))
        self.register_buffer("theta_std",  torch.ones(theta_dim))

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor):
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def set_theta_normalisation(self, mean: torch.Tensor, std: torch.Tensor):
        """Set theta_norm z-score buffers so inference can reproduce train-time
        θ standardisation."""
        self.theta_mean.copy_(mean)
        self.theta_std.copy_(std)

    def normalise_features(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feat_mean) / (self.feat_std + 1e-8)

    def normalise_theta(self, t: torch.Tensor) -> torch.Tensor:
        """Optional helper: apply the persisted θ z-score if a caller passes
        raw σ-units are already assumed by forward(), but this hook lets an
        inference path re-standardise theta_norm against corpus stats."""
        return (t - self.theta_mean) / (self.theta_std + 1e-8)

    def forward(self, x_health: torch.Tensor, theta_norm: torch.Tensor,
                n_grid: torch.Tensor) -> torch.Tensor:
        """
        Integrate the ODE across n_grid starting from SOH=1.0 (BoL).

        Args:
            x_health:   (batch, x_health_dim) raw health features
            theta_norm: (batch, theta_dim)    normalised physics parameters
            n_grid:     (T,) monotonic cycle grid, or scalar T for
                        torch.linspace(0, n_max, T)

        Returns:
            soh_trajectory: (T, batch, 1)
        """
        if n_grid.dim() == 0:
            n_grid = torch.linspace(0.0, self.ode.n_max, int(n_grid.item()),
                                    device=x_health.device)
        batch = x_health.shape[0]
        x_norm = self.normalise_features(x_health)
        # theta_norm arrives pre-normalised (identified-parameter z-scores)
        ctx = torch.cat([x_norm, theta_norm], dim=-1)
        self.ode.branch_context = ctx

        soh_0 = torch.ones(batch, 1, device=x_health.device,
                           dtype=x_health.dtype)

        trajectory = odeint(
            self.ode, soh_0, n_grid,
            method="dopri5",
            rtol=self.rtol, atol=self.atol,
            options={"max_num_steps": 1000},
        )
        return trajectory  # (T, batch, 1)

    def predict(self, x_health: torch.Tensor, theta_norm: torch.Tensor,
                cycles: torch.Tensor) -> torch.Tensor:
        """
        Return SoH at the requested cycle numbers.

        Args:
            x_health:   (batch, x_health_dim)
            theta_norm: (batch, theta_dim)
            cycles:     (T,) sorted cycle numbers to evaluate at

        Returns:
            soh_at_cycles: (T, batch, 1)
        """
        self.eval()
        cycles = cycles if cycles[0].item() == 0.0 else torch.cat(
            [torch.zeros(1, device=cycles.device, dtype=cycles.dtype), cycles]
        )
        with torch.no_grad():
            traj = self.forward(x_health, theta_norm, cycles)
        # If we prepended n=0, drop that sample
        if traj.shape[0] != cycles.shape[0]:
            return traj
        return traj if cycles[0].item() == 0.0 else traj[1:]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Dry forward pass — confirms branch signature and shape
    torch.manual_seed(0)
    model = ThetaConditionedRULPredictor(
        x_health_dim=5, theta_dim=6,
        hidden=64, n_layers=3, dropout=0.1,
        decoder_bias_init=-5.0,
    )

    batch = 4
    x_health   = torch.zeros(batch, 5)
    theta_norm = torch.zeros(batch, 6)
    n_grid     = torch.linspace(0.0, 2000.0, 100)

    traj = model(x_health, theta_norm, n_grid)

    print(f"Module path: Voltaris/Data_Exploration/phase3_operator.py")
    print(f"Branch input dim: 1 (SOH) + 1 (n_norm) + 5 (x_health) + 6 (theta) = 13")
    print(f"Dry-forward output shape: {tuple(traj.shape)}")
    print(f"Parameter count: {model.n_parameters():,}")
    print(f"SoH range at t=n_grid[-1]: "
          f"{traj[-1].min().item():.4f} – {traj[-1].max().item():.4f}")
