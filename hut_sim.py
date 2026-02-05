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
    lambda_min: float = 5e-2
    lambda_max: float = 5.0
    dark_pressure_coeff: float = 0.25
    certainty_coeff: float = 1.0
    newton_coeff: float = 1.0
    softening: float = 1e-3


@dataclass
class StatisticalObject:
    """HUT statistical object: (mu, Sigma, theta, k)."""

    mu: np.ndarray
    sigma: np.ndarray
    theta: float
    k: np.ndarray


def planck_lattice(indices: np.ndarray, *, planck_length: float) -> np.ndarray:
    """Embed integer lattice indices into physical space."""

    return planck_length * indices


def clamp_covariance(sigma: np.ndarray, *, lambda_min: float, lambda_max: float) -> np.ndarray:
    """Clamp covariance eigenvalues and reconstruct a symmetric PSD matrix."""

    sym = 0.5 * (sigma + sigma.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, lambda_min, lambda_max)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def certainty(sigma: np.ndarray, *, eps: float = 1e-12) -> float:
    """Rotation-invariant certainty: C = 1/sqrt(det(Sigma))."""

    det_sigma = max(float(np.linalg.det(sigma)), eps)
    return 1.0 / np.sqrt(det_sigma)


def heat(sigma: np.ndarray, *, eps: float = 1e-12) -> float:
    """Canonical heat definition: H = 1/C."""

    return 1.0 / certainty(sigma, eps=eps)


def mass_observed(sigma: np.ndarray, *, kappa: float, planck_length: float, hbar: float = 1.0) -> float:
    """Certainty-weighted observed mass scale with finite covariance clamping."""

    return kappa * hbar * certainty(sigma) / max(planck_length, 1e-12)


def gaussian_kernel(x: np.ndarray, obj: StatisticalObject, *, eps: float = 1e-12) -> np.ndarray:
    dims = x.shape[-1]
    cov = obj.sigma
    det_cov = max(float(np.linalg.det(cov)), eps)
    inv_cov = np.linalg.inv(cov + eps * np.eye(dims))
    diff = x - obj.mu
    exponent = -0.5 * np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    norm = 1.0 / np.sqrt(((2.0 * np.pi) ** dims) * det_cov)
    phase = obj.theta + np.tensordot(x, obj.k, axes=([-1], [0]))
    return norm * np.exp(exponent + 1j * phase)


def super_gaussian_kernel(x: np.ndarray, obj: StatisticalObject, *, power: int = 4) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(obj.sigma)
    transformed = np.tensordot(x - obj.mu, eigvecs, axes=([-1], [0]))
    scaled = transformed / np.maximum(np.sqrt(eigvals), 1e-12)
    exponent = -0.5 * np.sum(np.abs(scaled) ** power, axis=-1)
    phase = obj.theta + np.tensordot(x, obj.k, axes=([-1], [0]))
    return np.exp(exponent + 1j * phase)


def total_field(x: np.ndarray, objects: Iterable[StatisticalObject], *, power: int = 2) -> np.ndarray:
    kernels = [gaussian_kernel(x, obj) if power == 2 else super_gaussian_kernel(x, obj, power=power) for obj in objects]
    return np.sum(kernels, axis=0)


def potential_certainty(sigma: np.ndarray, *, coeff: float) -> float:
    return -coeff * certainty(sigma)


def potential_dark_pressure(sigma: np.ndarray, *, coeff: float) -> float:
    return coeff * heat(sigma)


def sink_flow_sigma(sigma: np.ndarray, params: HUTParams, *, dt: float) -> np.ndarray:
    """Stable covariance update with finite certainty response and eigenvalue clamping."""

    C = certainty(sigma)
    H = heat(sigma)
    drive = params.certainty_coeff * C - params.dark_pressure_coeff * H
    sigma_next = sigma + dt * params.gamma * drive * np.eye(params.dims)
    return clamp_covariance(sigma_next, lambda_min=params.lambda_min, lambda_max=params.lambda_max)


def newton_potential(mu: np.ndarray, objects: Iterable[StatisticalObject], params: HUTParams) -> float:
    """Effective Newtonian-like potential from certainty-weighted masses (softened)."""

    potential = 0.0
    for obj in objects:
        if np.allclose(obj.mu, mu):
            continue
        r = np.linalg.norm(mu - obj.mu)
        m = mass_observed(obj.sigma, kappa=params.kappa, planck_length=params.planck_length)
        potential -= params.newton_coeff * m / np.sqrt(r * r + params.softening**2)
    return potential


def newton_force(mu: np.ndarray, objects: Iterable[StatisticalObject], params: HUTParams) -> np.ndarray:
    """Negative gradient of the certainty-weighted potential (pairwise form)."""

    force = np.zeros(params.dims)
    for obj in objects:
        if np.allclose(obj.mu, mu):
            continue
        r_vec = mu - obj.mu
        r2 = np.dot(r_vec, r_vec) + params.softening**2
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
    """Evolve (mu, Sigma, theta) with local sink-flow and certainty-weighted forces."""

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
        StatisticalObject(mu=np.array([0.0, 0.0, 0.0, 0.0]), sigma=np.diag([0.6, 0.7, 0.8, 0.9]), theta=0.1, k=np.zeros(4)),
        StatisticalObject(mu=np.array([1.5, 0.5, 0.0, -0.2]), sigma=np.diag([1.2, 1.0, 0.9, 0.8]), theta=0.4, k=np.zeros(4)),
    ]

    for _ in range(10):
        objects = update_objects(objects, params, dt=0.05)

    lattice = np.stack(np.meshgrid(*[np.linspace(-2, 2, 8)] * params.dims, indexing="ij"), axis=-1)
    field = total_field(lattice, objects, power=2)
    print("Field magnitude (sample):", float(np.mean(np.abs(field))))


if __name__ == "__main__":
    demo()
