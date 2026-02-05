from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HUTParams:
    """Numerical and physical parameters for a local HUT stage simulation."""

    nx: int = 96
    ny: int = 96
    dx: float = 0.2
    dt: float = 0.05
    steps: int = 240
    c_u: float = 1.1
    gamma_u: float = 0.08
    kappa_u: float = 0.02
    alpha_u: float = 0.7
    c_t: float = 0.25
    gamma_t: float = 0.18
    beta_t: float = 0.5
    z_sat_u: float = 4.0
    z_sat_t: float = 2.0
    lambda_min: float = 0.04
    lambda_max: float = 3.0
    stamp_sigma_cutoff: float = 3.5
    particle_damping: float = 0.6
    particle_coupling: float = 0.7
    particle_noise: float = 0.05
    move_particles: bool = False
    seed: int = 7


@dataclass
class Particle:
    """HUT state object with anisotropic covariance and phase structure."""

    mu: np.ndarray  # shape (2,)
    sigma: np.ndarray  # shape (2,2)
    phase: float
    phase_grad: np.ndarray  # shape (2,)
    vel: np.ndarray


@dataclass
class Stage:
    """HUT stage fields on a local lattice."""

    U: np.ndarray
    U_prev: np.ndarray
    T: np.ndarray


def clamp_covariance(sigma: np.ndarray, *, lambda_min: float, lambda_max: float) -> np.ndarray:
    sym = 0.5 * (sigma + sigma.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, lambda_min, lambda_max)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def certainty(sigma: np.ndarray, *, eps: float = 1e-12) -> float:
    """Canonical HUT certainty, exact invariant definition."""

    det_sigma = max(float(np.linalg.det(sigma)), eps)
    return 1.0 / np.sqrt(det_sigma)


def heat(sigma: np.ndarray) -> float:
    """Canonical HUT heat, exact invariant definition."""

    return 1.0 / certainty(sigma)


def sat(z: np.ndarray, *, z_sat: float) -> np.ndarray:
    return z_sat * np.tanh(z / max(z_sat, 1e-12))


def laplacian_periodic(field: np.ndarray, *, dx: float) -> np.ndarray:
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (dx * dx)


def gradient_periodic(field: np.ndarray, *, dx: float) -> tuple[np.ndarray, np.ndarray]:
    gx = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * dx)
    gy = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * dx)
    return gx, gy


def world_grid(params: HUTParams) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(params.nx) - params.nx / 2.0) * params.dx
    ys = (np.arange(params.ny) - params.ny / 2.0) * params.dx
    return np.meshgrid(xs, ys, indexing="ij")


def deposit_sources_local(
    particles: list[Particle],
    params: HUTParams,
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Local deposition only: S_C = Σ C_i A_i, S_phi = Σ curl(A_i^2 ∇phi_i)."""

    S_C = np.zeros_like(X)
    jx_total = np.zeros_like(X)
    jy_total = np.zeros_like(X)

    for p in particles:
        sigma = clamp_covariance(p.sigma, lambda_min=params.lambda_min, lambda_max=params.lambda_max)
        inv_sigma = np.linalg.inv(sigma)
        eigvals = np.linalg.eigvalsh(sigma)
        max_std = float(np.sqrt(np.max(eigvals)))
        half_width = max(2, int(np.ceil(params.stamp_sigma_cutoff * max_std / params.dx)))

        cx = int(np.argmin(np.abs(X[:, 0] - p.mu[0])))
        cy = int(np.argmin(np.abs(Y[0, :] - p.mu[1])))

        i0, i1 = max(0, cx - half_width), min(params.nx, cx + half_width + 1)
        j0, j1 = max(0, cy - half_width), min(params.ny, cy + half_width + 1)

        dx_local = X[i0:i1, j0:j1] - p.mu[0]
        dy_local = Y[i0:i1, j0:j1] - p.mu[1]
        quad = (
            inv_sigma[0, 0] * dx_local * dx_local
            + 2.0 * inv_sigma[0, 1] * dx_local * dy_local
            + inv_sigma[1, 1] * dy_local * dy_local
        )
        A = np.exp(-0.5 * quad)

        C = certainty(sigma)
        S_C[i0:i1, j0:j1] += C * A

        A2 = A * A
        jx_total[i0:i1, j0:j1] += A2 * p.phase_grad[0]
        jy_total[i0:i1, j0:j1] += A2 * p.phase_grad[1]

    d_jy_dx = (np.roll(jy_total, -1, axis=0) - np.roll(jy_total, 1, axis=0)) / (2.0 * params.dx)
    d_jx_dy = (np.roll(jx_total, -1, axis=1) - np.roll(jx_total, 1, axis=1)) / (2.0 * params.dx)
    S_phi = d_jy_dx - d_jx_dy
    return S_C, S_phi


def evolve_stage(stage: Stage, S_C: np.ndarray, S_phi: np.ndarray, params: HUTParams) -> Stage:
    """Neighbor-only finite-difference updates with inertia/memory for U."""

    lap_U = laplacian_periodic(stage.U, dx=params.dx)
    source_u = sat(S_C, z_sat=params.z_sat_u)
    U_next = (
        (2.0 - params.gamma_u * params.dt) * stage.U
        - (1.0 - params.gamma_u * params.dt) * stage.U_prev
        + (params.c_u * params.dt) ** 2 * lap_U
        + (params.dt * params.dt) * params.alpha_u * source_u
        - (params.dt * params.dt) * params.kappa_u * stage.U
    )

    lap_T = laplacian_periodic(stage.T, dx=params.dx)
    source_t = sat(S_phi, z_sat=params.z_sat_t)
    T_next = stage.T + params.dt * (params.c_t * params.c_t * lap_T - params.gamma_t * stage.T + params.beta_t * source_t)

    return Stage(U=U_next, U_prev=stage.U.copy(), T=T_next)


def sample_grad_at(field: np.ndarray, pos: np.ndarray, params: HUTParams) -> np.ndarray:
    gx, gy = gradient_periodic(field, dx=params.dx)
    ix = int(np.clip(np.round(pos[0] / params.dx + params.nx / 2.0), 0, params.nx - 1))
    iy = int(np.clip(np.round(pos[1] / params.dx + params.ny / 2.0), 0, params.ny - 1))
    return np.array([gx[ix, iy], gy[ix, iy]])


def update_particles_local(particles: list[Particle], stage: Stage, params: HUTParams, rng: np.random.Generator) -> None:
    if not params.move_particles:
        return

    for p in particles:
        grad_u = sample_grad_at(stage.U, p.mu, params)
        noise = rng.normal(size=2) * params.particle_noise * np.sqrt(heat(p.sigma))
        acc = -params.particle_coupling * grad_u - params.particle_damping * p.vel + noise
        p.vel = p.vel + params.dt * acc
        p.mu = p.mu + params.dt * p.vel


def radio_scenario(params: HUTParams) -> list[Particle]:
    """Single rotating anisotropic source with constant det(Sigma)."""

    base = np.diag([0.16, 1.0])
    return [
        Particle(
            mu=np.array([0.0, 0.0]),
            sigma=base.copy(),
            phase=0.0,
            phase_grad=np.array([2.5, 0.0]),
            vel=np.zeros(2),
        )
    ]


def rotate_covariance(base_sigma: np.ndarray, angle: float) -> np.ndarray:
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return R @ base_sigma @ R.T


def run_simulation(params: HUTParams) -> dict[str, float]:
    rng = np.random.default_rng(params.seed)
    X, Y = world_grid(params)

    stage = Stage(
        U=np.zeros((params.nx, params.ny), dtype=float),
        U_prev=np.zeros((params.nx, params.ny), dtype=float),
        T=np.zeros((params.nx, params.ny), dtype=float),
    )
    particles = radio_scenario(params)
    base_sigma = particles[0].sigma.copy()

    det_series: list[float] = []
    certainty_series: list[float] = []

    for step in range(params.steps):
        angle = 2.0 * np.pi * step / max(params.steps, 1)
        particles[0].sigma = clamp_covariance(
            rotate_covariance(base_sigma, angle),
            lambda_min=params.lambda_min,
            lambda_max=params.lambda_max,
        )
        particles[0].phase = angle
        particles[0].phase_grad = np.array([2.5 * np.cos(angle), 2.5 * np.sin(angle)])

        det_series.append(float(np.linalg.det(particles[0].sigma)))
        certainty_series.append(certainty(particles[0].sigma))

        S_C, S_phi = deposit_sources_local(particles, params, X, Y)
        stage = evolve_stage(stage, S_C, S_phi, params)
        update_particles_local(particles, stage, params, rng)

    return {
        "det_min": float(np.min(det_series)),
        "det_max": float(np.max(det_series)),
        "C_min": float(np.min(certainty_series)),
        "C_max": float(np.max(certainty_series)),
        "U_rms": float(np.sqrt(np.mean(stage.U * stage.U))),
        "T_rms": float(np.sqrt(np.mean(stage.T * stage.T))),
    }


def demo() -> None:
    params = HUTParams()
    metrics = run_simulation(params)
    print("HUT radio scenario complete")
    print(f"det(Sigma) range: [{metrics['det_min']:.8f}, {metrics['det_max']:.8f}]")
    print(f"certainty C range: [{metrics['C_min']:.8f}, {metrics['C_max']:.8f}]")
    print(f"stage U_rms: {metrics['U_rms']:.8f}")
    print(f"stage T_rms: {metrics['T_rms']:.8f}")


if __name__ == "__main__":
    demo()
