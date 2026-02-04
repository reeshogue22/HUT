from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HUTParams:
    """Parameters for the HUT lattice + kernel-field model."""
    dims: int = 4
    planck_length: float = 1.0
    gamma: float = 0.35
    kappa: float = 1.0
    sigma_floor: float = 0.05
    sigma_ceiling: float = 5.0
    dark_pressure_coeff: float = 0.25
    certainty_coeff: float = 1.0
    newton_coeff: float = 1.0
    softening: float = 1e-3


@dataclass
class StatisticalObject:
    """HUT statistical object: (mu, sigma, theta, k)."""
    mu: np.ndarray
    sigma: float
    theta: float
    k: np.ndarray


def planck_lattice(indices: np.ndarray, *, planck_length: float) -> np.ndarray:
    """Embed integer lattice indices into physical space."""
    return planck_length * indices


def certainty(sigma: float) -> float:
    return 1.0 / (sigma * sigma)


def mass_observed(sigma: float, *, kappa: float, planck_length: float, hbar: float = 1.0) -> float:
    """Hogue--mass relation: M_obs = kappa * hbar / (ell_P * sigma)."""
    return kappa * hbar / (planck_length * sigma)


def gaussian_kernel(x: np.ndarray, obj: StatisticalObject) -> np.ndarray:
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * obj.sigma) ** x.shape[-1]
    exponent = -np.sum((x - obj.mu) ** 2, axis=-1) / (2.0 * obj.sigma ** 2)
    phase = obj.theta + np.dot(x, obj.k)
    return norm * np.exp(exponent + 1j * phase)


def total_field(x: np.ndarray, objects: Iterable[StatisticalObject]) -> np.ndarray:
    kernels = [gaussian_kernel(x, obj) for obj in objects]
    return np.sum(kernels, axis=0)


def potential_certainty(sigma: float, *, coeff: float) -> float:
    return -coeff / sigma


def potential_dark_pressure(sigma: float, *, coeff: float) -> float:
    return coeff * sigma * sigma


def sink_flow_sigma(sigma: float, params: HUTParams, *, dt: float) -> float:
    """Minimal sink-flow update for sigma."""
    dV = params.certainty_coeff / (sigma * sigma)
    dP = 2.0 * params.dark_pressure_coeff * sigma
    dsigma = -params.gamma * (dV + dP)
    sigma_next = sigma + dt * dsigma
    return float(np.clip(sigma_next, params.sigma_floor, params.sigma_ceiling))


def newton_potential(mu: np.ndarray, objects: Iterable[StatisticalObject], params: HUTParams) -> float:
    """Effective Newtonian potential from observed masses (softened)."""
    potential = 0.0
    for obj in objects:
        if np.allclose(obj.mu, mu):
            continue
        r = np.linalg.norm(mu - obj.mu)
        m = mass_observed(obj.sigma, kappa=params.kappa, planck_length=params.planck_length)
        potential -= params.newton_coeff * m / np.sqrt(r * r + params.softening ** 2)
    return potential


def newton_force(mu: np.ndarray, objects: Iterable[StatisticalObject], params: HUTParams) -> np.ndarray:
    """Negative gradient of the Newtonian potential (pairwise form)."""
    force = np.zeros(params.dims)
    for obj in objects:
        if np.allclose(obj.mu, mu):
            continue
        r_vec = mu - obj.mu
        r2 = np.dot(r_vec, r_vec) + params.softening ** 2
        r = np.sqrt(r2)
        m = mass_observed(obj.sigma, kappa=params.kappa, planck_length=params.planck_length)
        force -= params.newton_coeff * m * r_vec / (r2 * r)
    return force


def phase_connection(theta: np.ndarray, k_vec: np.ndarray) -> np.ndarray:
    """Discrete connection A_mu = Delta_mu theta + k_mu on a lattice."""
    grads = np.stack(np.gradient(theta), axis=0)
    return grads + k_vec[:, None, None, None, None]


def phase_curvature(A_mu: np.ndarray) -> np.ndarray:
    """Curvature F_{mu,nu} = Delta_mu A_nu - Delta_nu A_mu."""
    dims = A_mu.shape[0]
    F = np.zeros((dims, dims) + A_mu.shape[1:])
    for mu in range(dims):
        for nu in range(dims):
            if mu == nu:
                continue
            dmu_Anu = np.gradient(A_mu[nu], axis=mu)
            dnu_Amu = np.gradient(A_mu[mu], axis=nu)
            F[mu, nu] = dmu_Anu - dnu_Amu
    return F


def phase_vorticity_flow(theta: np.ndarray, k_vec: np.ndarray) -> np.ndarray:
    """Compute a proxy for phase vorticity on a lattice (sum of curvature norms)."""
    A_mu = phase_connection(theta, k_vec)
    F = phase_curvature(A_mu)
    return np.sqrt(np.sum(F * F, axis=(0, 1)))


def update_objects(objects: list[StatisticalObject], params: HUTParams, *, dt: float) -> list[StatisticalObject]:
    """Evolve (mu, sigma, theta) with sink-flow and Newton + phase flow."""
    next_objects: list[StatisticalObject] = []
    for obj in objects:
        sigma_next = sink_flow_sigma(obj.sigma, params, dt=dt)
        mu_next = obj.mu + dt * newton_force(obj.mu, objects, params)
        theta_next = obj.theta + dt * 0.1 * certainty(sigma_next)
        next_objects.append(StatisticalObject(mu=mu_next, sigma=sigma_next, theta=theta_next, k=obj.k))
    return next_objects


def demo() -> None:
    params = HUTParams()
    objects = [
        StatisticalObject(mu=np.array([0.0, 0.0, 0.0, 0.0]), sigma=0.6, theta=0.1, k=np.zeros(4)),
        StatisticalObject(mu=np.array([1.5, 0.5, 0.0, -0.2]), sigma=1.2, theta=0.4, k=np.zeros(4)),
    ]

    for _ in range(10):
        objects = update_objects(objects, params, dt=0.05)

    lattice = np.stack(np.meshgrid(*[np.linspace(-2, 2, 8)] * params.dims, indexing="ij"), axis=-1)
    field = total_field(lattice, objects)
    print("Field magnitude (sample):", float(np.mean(np.abs(field))))


if __name__ == "__main__":
    demo()
