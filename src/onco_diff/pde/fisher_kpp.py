"""Fisher-Kolmogorov-Petrovsky-Piskunov (Fisher-KPP) Anisotropic PDE Solver."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional, List, Callable
import numpy as np
from onco_diff.pde.anisotropy import DTITensorField


@dataclass
class SimulationHistory:
    """Time-series of glioma cell distributions and total tumor burden."""
    times_days: np.ndarray
    tumor_density_snapshots: List[np.ndarray]
    total_cell_burdens: np.ndarray

    @property
    def final_density(self) -> np.ndarray:
        return self.tumor_density_snapshots[-1]


class GliomaReactionDiffusionSolver:
    """Solves anisotropic reaction-diffusion equations for glioma cell invasion."""

    def __init__(self, tensor_field: DTITensorField):
        self.field = tensor_field
        self.shape = tensor_field.shape
        self.ndim = tensor_field.ndim
        self.dx = tensor_field.voxel_size_mm
        self.props = tensor_field.props

    def _compute_flux_divergence(self, c: np.ndarray) -> np.ndarray:
        """
        Compute div(D * grad(c)) via finite differences with zero-flux boundary conditions.
        """
        ndim = self.ndim
        grads = np.gradient(c, self.dx)  # [dc/dx0, dc/dx1, ...]

        # Flux vector J_i = sum_j D_ij * grad_j(c)
        flux = np.zeros((ndim, *self.shape), dtype=np.float64)
        for i in range(ndim):
            for j in range(ndim):
                flux[i] += self.field.d_tensor[i, j] * grads[j]

        # Divergence: div(J) = sum_i d(J_i)/dx_i
        div_flux = np.zeros(self.shape, dtype=np.float64)
        for i in range(ndim):
            div_flux += np.gradient(flux[i], self.dx, axis=i)

        # Enforce zero-flux on CSF / non-brain voxels
        brain_mask = (self.field.tissue_map > 0)
        div_flux[~brain_mask] = 0.0

        return div_flux

    def step(
        self,
        c: np.ndarray,
        dt_days: float,
        external_kill_rate: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Single explicit time integration step with logistic proliferation and therapy kill.
        """
        # Diffusion term
        diffusion = self._compute_flux_divergence(c)

        # Logistic proliferation: rho * c * (1 - c / K)
        k = self.props.carrying_capacity
        rho = self.props.proliferation_rate_per_day
        proliferation = rho * c * np.maximum(0.0, 1.0 - c / k)

        # External cytotoxic kill (radiotherapy / chemotherapy)
        kill = external_kill_rate if external_kill_rate is not None else 0.0

        # Update
        c_new = c + dt_days * (diffusion + proliferation - kill)

        # Biological bounds [0, K]
        c_new = np.clip(c_new, 0.0, k)
        # Zero out CSF / non-brain
        c_new[self.field.tissue_map == 0] = 0.0

        return c_new

    def simulate(
        self,
        initial_density: np.ndarray,
        total_days: float,
        dt_days: float = 0.1,
        snapshot_interval_days: float = 5.0,
        intervention_callback: Optional[Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray | None]]] = None,
    ) -> SimulationHistory:
        """
        Run forward simulation over specified time horizon.
        
        Args:
            initial_density: Initial normalized tumor cell distribution (shape matching field).
            total_days: Total simulation duration in days.
            dt_days: Time step (must satisfy CFL stability limit: dt <= dx^2 / (2 * ndim * Dmax)).
            snapshot_interval_days: Time between saved snapshots.
            intervention_callback: Optional function (t, c) -> (c_after_discrete_event, kill_rate_field)
                                   used to model surgical resection, radiotherapy fractions, etc.
        """
        # Check CFL condition
        d_max = self.props.diff_white_matter_mm2_day
        cfl_limit = (self.dx ** 2) / (2.0 * self.ndim * max(1e-6, d_max))
        if dt_days > cfl_limit:
            dt_days = cfl_limit * 0.8  # Enforce numerical stability

        n_steps = int(np.ceil(total_days / dt_days))
        curr_t = 0.0
        c = np.copy(initial_density)

        times_list: List[float] = [curr_t]
        snapshots: List[np.ndarray] = [np.copy(c)]
        burdens_list: List[float] = [float(np.sum(c))]

        next_snapshot_t = snapshot_interval_days

        for step_idx in range(n_steps):
            kill_rate = None
            if intervention_callback is not None:
                c, kill_rate = intervention_callback(curr_t, c)

            c = self.step(c, dt_days, external_kill_rate=kill_rate)
            curr_t += dt_days

            if curr_t >= next_snapshot_t or step_idx == n_steps - 1:
                times_list.append(curr_t)
                snapshots.append(np.copy(c))
                burdens_list.append(float(np.sum(c)))
                next_snapshot_t += snapshot_interval_days

        return SimulationHistory(
            times_days=np.array(times_list),
            tumor_density_snapshots=snapshots,
            total_cell_burdens=np.array(burdens_list),
        )
