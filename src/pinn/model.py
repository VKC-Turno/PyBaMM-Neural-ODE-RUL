"""
src/pinn/model.py — Neural ODE for LFP degradation / RUL prediction.

Architecture: continuous-time ODE where dSOH/dn = f_theta(SOH, n, x_health)
Integrator: dopri5 (adaptive Runge-Kutta 4/5) via torchdiffeq
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from typing import Optional
import yaml
from pathlib import Path


class DegradationODE(nn.Module):
    """
    Parameterises the right-hand side of dSOH/dn = f(SOH, n, x_health).

    Forces output to be negative (SOH can only decrease) via -softplus.
    Health context is injected via a stored attribute — avoids the
    torchdiffeq interface which only accepts (t, y) signatures.
    """

    def __init__(self, health_dim: int = 5, hidden: int = 64,
                 n_layers: int = 3, dropout_p: float = 0.1):
        super().__init__()
        # Input: [SOH(1), normalised_cycle(1), health_features(health_dim)]
        in_dim = 1 + 1 + health_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Dropout(dropout_p), nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        # Initialise final layer near zero → start with slow degradation
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -5.0)  # small initial rate

        self.health_context: Optional[torch.Tensor] = None  # set before integrate
        self.n_max = 2000.0  # normalisation factor for cycle number

    def forward(self, n: torch.Tensor, soh: torch.Tensor) -> torch.Tensor:
        """
        n:   scalar cycle number (torchdiffeq calls with scalar t)
        soh: (batch, 1) current SOH values
        Returns: (batch, 1) dSOH/dn — always <= 0
        """
        batch = soh.shape[0]
        n_norm = (n / self.n_max).expand(batch, 1)
        h = self.health_context  # (batch, health_dim)

        inp = torch.cat([soh, n_norm, h], dim=-1)
        raw = self.net(inp)
        # Enforce monotonic decrease: dSOH/dn <= 0
        dSOH = -F.softplus(raw)
        return dSOH


class RULPredictor(nn.Module):
    """
    Full RUL prediction model.

    Usage:
        model = RULPredictor.from_config("configs/pinn_config.yaml")
        trajectory = model(soh_0, n_eval, x_health)
        rul = model.predict_rul(soh_now, cycle_now, x_health)
    """

    def __init__(self, health_dim: int = 5, hidden: int = 64,
                 n_layers: int = 3, dropout_p: float = 0.1,
                 eol_threshold: float = 0.8,
                 ode_rtol: float = 1e-4, ode_atol: float = 1e-6):
        super().__init__()
        self.ode = DegradationODE(health_dim, hidden, n_layers, dropout_p)
        self.eol = eol_threshold
        self.rtol = ode_rtol
        self.atol = ode_atol

        # Feature normalisation params (set via set_normalisation())
        self.register_buffer("feat_mean", torch.zeros(health_dim))
        self.register_buffer("feat_std",  torch.ones(health_dim))

    @classmethod
    def from_config(cls, config_path: str) -> "RULPredictor":
        cfg = yaml.safe_load(Path(config_path).read_text())
        m = cfg["model"]
        eol = cfg["eol"]["soh_threshold"]
        return cls(
            health_dim=m["health_feature_dim"],
            hidden=m["hidden_size"],
            n_layers=m["n_hidden_layers"],
            dropout_p=m["dropout_p"],
            eol_threshold=eol,
            ode_rtol=m["ode_rtol"],
            ode_atol=m["ode_atol"],
        )

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor):
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def normalise_features(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feat_mean) / (self.feat_std + 1e-8)

    def forward(self, soh_0: torch.Tensor, n_eval: torch.Tensor,
                x_health: torch.Tensor) -> torch.Tensor:
        """
        Integrate ODE from n_eval[0] to n_eval[-1].

        Args:
            soh_0:    (batch, 1) initial SOH at n_eval[0]
            n_eval:   (T,) cycle numbers to evaluate at
            x_health: (batch, health_dim) health features (raw, unnormalised)

        Returns:
            trajectory: (T, batch, 1) predicted SOH at each n in n_eval
        """
        x_norm = self.normalise_features(x_health)
        self.ode.health_context = x_norm

        trajectory = odeint(
            self.ode, soh_0, n_eval,
            method="dopri5",
            rtol=self.rtol, atol=self.atol,
            options={"max_num_steps": 1000},
        )
        return trajectory  # (T, batch, 1)

    def predict_rul(self, soh_now: float, cycle_now: float,
                    x_health: torch.Tensor,
                    max_future_cycles: int = 2000,
                    n_points: int = 500) -> dict:
        """
        Predict RUL from current state.

        Returns dict with rul_cycles, trajectory arrays, n_eol estimate.
        """
        self.eval()
        with torch.no_grad():
            soh_0 = torch.tensor([[soh_now]], dtype=torch.float32)
            n_eval = torch.linspace(cycle_now,
                                    cycle_now + max_future_cycles,
                                    n_points)
            traj = self(soh_0, n_eval, x_health.unsqueeze(0))  # (T,1,1)
            traj = traj.squeeze().numpy()                        # (T,)

        below = (traj < self.eol).nonzero()[0] if (traj < self.eol).any() else None
        if below is None or len(below) == 0:
            n_eol = cycle_now + max_future_cycles
        else:
            n_eol = float(n_eval[below[0]])

        return {
            "rul_cycles":   max(0.0, n_eol - cycle_now),
            "n_eol":        n_eol,
            "soh_trajectory": traj,
            "n_trajectory":   n_eval.numpy(),
        }

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    model = RULPredictor(health_dim=5, hidden=64)
    print(f"Model parameters: {model.n_parameters():,}")

    soh_0    = torch.tensor([[0.95]])
    n_eval   = torch.linspace(0, 500, 50)
    x_health = torch.zeros(1, 5)

    traj = model(soh_0, n_eval, x_health)
    print(f"Trajectory shape: {traj.shape}")   # (50, 1, 1)
    print(f"SOH range: {traj.min():.4f} – {traj.max():.4f}")

    result = model.predict_rul(0.95, 100, torch.zeros(5))
    print(f"RUL estimate: {result['rul_cycles']:.0f} cycles")
