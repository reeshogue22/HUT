from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""Unified Simple Field Theory: particle-universe simulation (assumption-closed)

You asked for a single, self-contained "particle universe" simulation that turns
intuition into explicit, runnable dynamics.

This file therefore defines a concrete closure (explicit assumptions) consistent
with your themes:
  - Certainty / information density: C_i = 1/sigma_i^2
  - Heat = inverse certainty: H_i = 1/C_i = sigma_i^2
  - Gravity from certainty gradients (delta-C), modulated by buoyancy:
        high certainty (small sigma) -> strong attraction
        low certainty (large sigma)  -> weak or buoyant (repulsive) coupling
  - Torsion as spin-driven twist: spin differences induce transverse forces
  - Two-sheet fabric: a backsheet (antimatter) is modeled as a time-reversed
    copy of the dynamics (backwards in time in a simplified sense)
  - 7D kinematics: 6 spatial dimensions + 1 spin dimension
    (a single observer sees a 4D projection of the full state)
  - Dark-energy-like behavior: a horizon boundary acts as an entropy sink and
    generates an outward drift ("terminal velocity flow")

Everything below is a simplified model (not GR, not Einstein--Cartan equations). It is
structured so each term is editable and labeled, keeping the closure compact
and explicit rather than relying on long, opaque Lagrangians.

Outputs
-------
A single run produces a 2x2 "universe dashboard":
  (1) particle positions colored by log10(sigma) (2D projection)
  (2) radial speed vs radius (Hubble-like trend if horizon term dominates)
  (3) tangential speed vs radius (rotation-curve-like diagnostic)
  (4) sigma histogram (uncertainty distribution)
"""


@dataclass(frozen=True)
class UniverseParams:
    n_particles: int = 240
    dims: int = 6
    horizon_radius: float = 10.0
    dt: float = 2.0e-3
    certainty_gravity: float = 0.08
    sigma_crit: float = 1.2
    eps: float = 0.06
    torsion_strength: float = 0.35
    torsion_r0: float = 1.2
    sigma_bh: float = 4.0e-4
    r_annihilate: float = 0.35
    back_coupling: float = 0.6
    r_sing: float = 1.0
    sigma_sing: float = 1.5e-4
    horizon_v0: float = 0.05
    horizon_alpha: float = 1.0
    sigma_floor: float = 1e-3
    sigma_ceiling: float = 6.0
    cool_rate: float = 0.9
    heat_rate: float = 0.25
    mu_sigma: float = 1.4
    sigma_stddev: float = 0.6


def apply_universe_style() -> None:
    """A consistent matplotlib look (pure-matplotlib, no external styles)."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "axes.facecolor": "#0b1020",
        "figure.facecolor": "#0b1020",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#e2e8f0",
        "axes.titlecolor": "#e2e8f0",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "grid.color": "#334155",
        "grid.alpha": 0.35,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.frameon": True,
        "legend.facecolor": "#0b1020",
        "legend.edgecolor": "#334155",
        "lines.linewidth": 2.0,
    })
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=["#22c55e", "#60a5fa", "#f59e0b", "#f43f5e", "#a78bfa", "#14b8a6"]
    )


def certainty(sigma: np.ndarray, sigma_floor: float = 1e-6) -> np.ndarray:
    sig = np.maximum(sigma, sigma_floor)
    return 1.0 / (sig * sig)

def heat(sigma: np.ndarray, sigma_floor: float = 1e-6) -> np.ndarray:
    """Heat proxy as inverse certainty: H = 1 / C."""
    return 1.0 / certainty(sigma, sigma_floor=sigma_floor)


def buoyancy_factor(sigma: np.ndarray, sigma_crit: float) -> np.ndarray:
    """Coupling sign/strength: positive attracts, negative is buoyant/repulsive."""
    return 1.0 - sigma / sigma_crit


def softnorm(x: np.ndarray, eps: float) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=-1) + eps * eps)


def pairwise_forces(x: np.ndarray,
                    v: np.ndarray,
                    sigma: np.ndarray,
                    spin: np.ndarray,
                    layer: np.ndarray,
                    *,
                    Gc: float,
                    sigma_crit: float,
                    eps: float,
                    torsion_strength: float,
                    torsion_r0: float,
                    back_coupling: float,
                    sigma_bh: float = 3.5e-4,
                    bh_grav_gain: float = 7.0,
                    bh_spin_strength: float = 0.65,
                    bh_spin_r0: float = 0.9) -> np.ndarray:
    """Compute net acceleration from certainty-gradient gravity + spin torsion.

    Added feature: a two-sided fabric (matter vs ``antimatter backsheet'').

    - layer=0: matter sheet
    - layer=1: antimatter (back) sheet

    Coupling rule (simplified closure):
      - same-layer interactions: attract/repel according to buoyancy (as before)
      - cross-layer interactions: sign-flipped and reduced by `back_coupling`
        (models "right on the back" with an opposite geometric response)

    Added feature: singularities / ``black holes''.

    If a particle cools below `sigma_bh`, it behaves like a singular core that:
      (i) attracts more strongly, and
      (ii) induces a transverse ``frame-dragging'' twist on nearby trajectories.

    Complexity: O(N^2) (intended for N~100--400).
    """
    N = x.shape[0]
    a = np.zeros_like(x)

    C = certainty(sigma)
    b = buoyancy_factor(sigma, sigma_crit=sigma_crit)
    is_bh = sigma < float(sigma_bh)

    for i in range(N):
        dx = x[i] - x  # (N,2)
        r = softnorm(dx, eps=eps)  # (N,)
        invr3 = 1.0 / (r * r * r)

        # remove self
        invr3[i] = 0.0

        # Sheet coupling: +1 for same sheet, -back_coupling for opposite sheet
        sheet = np.where(layer == layer[i], 1.0, -float(back_coupling))

        # If j is a BH-like core, it pulls harder.
        bh_gain = 1.0 + float(bh_grav_gain) * is_bh.astype(float)

        # gravity from certainty gradients (delta-C), modulated by buoyancy
        dC = C[i] - C
        w = Gc * dC * (b[i] * b) * sheet * bh_gain  # (N,)

        # attraction/repulsion along dx
        a[i] += -np.sum((w * invr3)[:, None] * dx, axis=0)

        # torsion: perpendicular to dx in the x-y plane, driven by spin differences
        dspin = spin[i] - spin
        perp = np.stack([-dx[:, 1], dx[:, 0]], axis=1)
        core = 1.0 / (1.0 + (r / torsion_r0) ** 2)
        twist = np.sum((np.sin(dspin) * core * sheet / (r + eps))[:, None] * perp, axis=0)
        a[i, :2] += torsion_strength * twist

        # BH frame-dragging: a local transverse ``twist'' sourced by BH cores.
        bh_core = 1.0 / (1.0 + (r / bh_spin_r0) ** 2)
        bh_twist = np.sum(((is_bh.astype(float) * bh_core) * sheet / (r + eps))[:, None] * perp, axis=0)
        a[i, :2] += bh_spin_strength * bh_twist

    # mild velocity damping to stabilize long runs
    a += -0.02 * v
    return a


def horizon_drift(x: np.ndarray, *, R: float, v0: float, alpha: float, eps: float = 1e-6) -> np.ndarray:
    """Outward terminal-velocity drift toward a horizon boundary at r=R."""
    r = softnorm(x, eps=eps)
    rhat = np.where(r[:, None] > 0, x / r[:, None], 0.0)
    y = np.clip(r / R, 0.0, 1.0 - eps)
    # drift speed rises near boundary
    speed = v0 * (y / (1.0 - y) ** alpha)
    return speed[:, None] * rhat


def update_sigma(sigma: np.ndarray,
                 *,
                 dt: float,
                 cool_rate: float,
                 heat_rate: float,
                 sigma_floor: float,
                 sigma_ceiling: float,
                 local_energy: np.ndarray) -> np.ndarray:
    """Simplified thermodynamics:

    - cooling (removing uncertainty): sigma decays toward sigma_floor
    - heating from motion/collisions: sigma increases with local_energy

    This lets regions "crystallize" if cooling dominates.
    """
    ds_cool = -cool_rate * (sigma - sigma_floor)
    ds_heat = heat_rate * local_energy
    sigma_next = sigma + dt * (ds_cool + ds_heat)
    return np.clip(sigma_next, sigma_floor, sigma_ceiling)

def universe_diagnostics(x: np.ndarray,
                         v: np.ndarray,
                         sigma: np.ndarray,
                         *,
                         sigma_floor: float) -> dict[str, float]:
    kinetic = float(np.mean(np.sum(v * v, axis=1)))
    mean_certainty = float(np.mean(certainty(sigma, sigma_floor=sigma_floor)))
    mean_heat = float(np.mean(heat(sigma, sigma_floor=sigma_floor)))
    rms_radius = float(np.sqrt(np.mean(np.sum(x * x, axis=1))))
    return {
        "mean_kinetic": kinetic,
        "mean_certainty": mean_certainty,
        "mean_heat": mean_heat,
        "rms_radius": rms_radius,
    }


def finalize_figure(fig: plt.Figure, *, output_path: str | None, show: bool) -> None:
    if output_path:
        fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)


def dashboard(x: np.ndarray, v: np.ndarray, sigma: np.ndarray, layer: np.ndarray, *, R: float, step: int) -> plt.Figure:
    x_proj = x[:, :2]
    v_proj = v[:, :2]
    r = softnorm(x_proj, eps=1e-6)
    rhat = np.where(r[:, None] > 0, x_proj / r[:, None], 0.0)

    vr = np.sum(v_proj * rhat, axis=1)
    vt = v_proj[:, 0] * (-rhat[:, 1]) + v_proj[:, 1] * (rhat[:, 0])

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # (1) positions: two sheets (front/back)
    m = layer == 0
    a = layer == 1
    sc_m = ax[0, 0].scatter(x_proj[m, 0], x_proj[m, 1], c=np.log10(sigma[m]), s=12, cmap="viridis", alpha=0.9, label="matter")
    ax[0, 0].scatter(x_proj[a, 0], x_proj[a, 1], c=np.log10(sigma[a]), s=10, cmap="viridis", alpha=0.5, marker="x", label="antimatter (back)")

    ax[0, 0].add_patch(plt.Circle((0, 0), R, fill=False, ec="#94a3b8", lw=1.5, alpha=0.7))
    ax[0, 0].set_aspect("equal", "box")
    ax[0, 0].set_title(f"Particle universe (step {step})")
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_ylabel("y")
    ax[0, 0].legend(loc="upper right")

    cb = fig.colorbar(sc_m, ax=ax[0, 0])
    cb.set_label(r"$\log_{10}(\sigma)$")

    # (2) radial speed vs radius
    ax[0, 1].scatter(r, vr, s=8, alpha=0.7)
    ax[0, 1].set_title("Radial speed vs radius")
    ax[0, 1].set_xlabel("r")
    ax[0, 1].set_ylabel(r"$v_r$")

    # (3) tangential speed vs radius
    ax[1, 0].scatter(r, np.abs(vt), s=8, alpha=0.7)
    ax[1, 0].set_title("Tangential speed vs radius")
    ax[1, 0].set_xlabel("r")
    ax[1, 0].set_ylabel(r"$|v_t|$")

    # (4) sigma distribution
    ax[1, 1].hist(np.log10(sigma), bins=30, color="#60a5fa", alpha=0.8)
    ax[1, 1].set_title("Uncertainty distribution")
    ax[1, 1].set_xlabel(r"$\log_{10}(\sigma)$")
    ax[1, 1].set_ylabel("count")

    plt.tight_layout()
    return fig


def main():
    apply_universe_style()

    # -----------------
    # Universe parameters (model units)
    # -----------------
    params = UniverseParams()
    N = params.n_particles
    dims = params.dims
    R = params.horizon_radius  # "horizon" radius (simulation boundary)
    dt = params.dt

    # Render mode
    animate = os.environ.get("ANIMATE", "1").lower() not in {"0", "false", "no"}
    show_plots = os.environ.get("SHOW_PLOTS", "1").lower() not in {"0", "false", "no"}
    output_path = os.environ.get("UNIVERSE_OUTPUT")

    # If animate=True
    n_frames = int(os.environ.get("N_FRAMES", "360"))
    substeps_per_frame = int(os.environ.get("SUBSTEPS_PER_FRAME", "12"))
    interval_ms = int(os.environ.get("FRAME_INTERVAL_MS", "30"))
    print_diagnostics = os.environ.get("PRINT_DIAGNOSTICS", "0").lower() in {"1", "true", "yes"}

    # If animate=False
    n_steps = int(os.environ.get("N_STEPS", "6000"))

    # Certainty gravity
    Gc = params.certainty_gravity
    sigma_crit = params.sigma_crit
    eps = params.eps

    # Phase/torsion
    torsion_strength = params.torsion_strength
    torsion_r0 = params.torsion_r0

    # Singularity / black-hole proxy
    sigma_bh = params.sigma_bh
    r_annihilate = params.r_annihilate

    # Two equal matter sheets (no time-reversal/annihilation); reduced cross-sheet coupling
    back_coupling = params.back_coupling

    # One pinned singularity per sheet
    r_sing = params.r_sing
    sigma_sing = params.sigma_sing

    # Horizon sink / drain flow
    horizon_v0 = params.horizon_v0
    horizon_alpha = params.horizon_alpha

    # Thermodynamics
    sigma_floor = params.sigma_floor
    sigma_ceiling = params.sigma_ceiling
    cool_rate = params.cool_rate
    heat_rate = params.heat_rate
    mu_sigma = params.mu_sigma
    sigma_stddev = params.sigma_stddev

    rng = np.random.default_rng(4)

    # -----------------
    # Initial conditions (two sheets + time-reversal + twin annihilation)
    # -----------------
    if N % 2 != 0:
        raise ValueError("N must be even to build matter/antimatter twins.")
    half = N // 2

    # Deterministic 50/50 sheets: first half is matter, second half is antimatter
    layer = np.zeros(N, dtype=int)
    layer[half:] = 1

    # Twin pairing: i <-> i+half
    twin = np.empty(N, dtype=int)
    twin[:half] = np.arange(half, N)
    twin[half:] = np.arange(0, half)

    # Sample matter positions and copy to antimatter with tiny jitter
    ang = rng.uniform(0, 2 * np.pi, half)
    rad = R * np.sqrt(rng.uniform(0, 0.75, half))
    x_m = np.zeros((half, dims))
    x_m[:, 0] = rad * np.cos(ang)
    x_m[:, 1] = rad * np.sin(ang)
    x_m[:, 2:] = 0.2 * rng.normal(size=(half, dims - 2))
    x = np.zeros((N, dims))
    x[:half] = x_m
    x[half:] = x_m + 0.03 * rng.normal(size=(half, dims))

    v_m = 0.05 * rng.normal(size=(half, dims))
    swirl = np.stack([-x_m[:, 1], x_m[:, 0]], axis=1) / (softnorm(x_m[:, :2], eps=1e-6)[:, None] + 1e-6)
    v_m[:, :2] += 0.03 * swirl
    v = np.zeros((N, dims))
    v[:half] = v_m
    v[half:] = -v_m

    sigma_m = rng.normal(mu_sigma, sigma_stddev, half)
    sigma_m *= 1.0 + 1.5 * (softnorm(x_m, eps=1e-6) / R)
    sigma_m = np.clip(sigma_m, sigma_floor, sigma_ceiling)
    sigma = np.zeros(N)
    sigma[:half] = sigma_m
    sigma[half:] = sigma_m

    spin_m = rng.uniform(-np.pi, np.pi, half) + 0.2 * softnorm(x_m, eps=1e-6)
    spin = np.zeros(N)
    spin[:half] = spin_m
    spin[half:] = spin_m + np.pi
    spin = (spin + np.pi) % (2 * np.pi) - np.pi

    # Pinned singularities: index 0 on sheet-0 and its twin (index half) on sheet-1
    sing_idx0 = 0
    sing_idx1 = half
    sing_idx = np.array([sing_idx0, sing_idx1], dtype=int)

    sing_pos = np.zeros((2, dims))
    sing_pos[0, 0] = -r_sing
    sing_pos[1, 0] = r_sing

    x[sing_idx] = sing_pos
    v[sing_idx] = 0.0
    sigma[sing_idx] = sigma_sing

    # Everything happens "between" the singularities: a soft midplane confiner toward x=0
    midplane_k = 0.06

    def evolve_one_step(x_: np.ndarray, v_: np.ndarray, sigma_: np.ndarray, spin_: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Antimatter sheet runs backwards in time (simplified): flip the sign of dt.
        tdir = np.where(layer == 0, 1.0, -1.0)  # +1 matter, -1 antimatter

        # Drift is time-reversed on the backsheet
        v_drift_ = horizon_drift(x_, R=R, v0=horizon_v0, alpha=horizon_alpha) * tdir[:, None]

        a_ = pairwise_forces(
            x_,
            v_ - v_drift_,
            sigma_,
            spin_,
            layer,
            Gc=Gc,
            sigma_crit=sigma_crit,
            eps=eps,
            torsion_strength=torsion_strength,
            torsion_r0=torsion_r0,
            back_coupling=back_coupling,
            sigma_bh=sigma_bh,
        )

        # Midplane confiner (x -> 0)
        midplane = np.zeros_like(x_)
        midplane[:, 0] = -midplane_k * x_[:, 0]
        a_ = a_ + midplane

        v_ = v_ + (tdir[:, None] * dt) * a_
        x_ = x_ + (tdir[:, None] * dt) * (v_ + v_drift_)

        r_ = softnorm(x_, eps=1e-6)
        outside_ = r_ > R
        if np.any(outside_):
            rhat_ = x_[outside_] / r_[outside_][:, None]
            x_[outside_] = R * rhat_
            vr_ = np.sum(v_[outside_] * rhat_, axis=1)
            v_[outside_] = v_[outside_] - 2.0 * vr_[:, None] * rhat_

        # Thermodynamics
        local_energy_ = np.sum(v_ * v_, axis=1)
        sigma_ = update_sigma(
            sigma_,
            dt=dt,
            cool_rate=cool_rate,
            heat_rate=heat_rate,
            sigma_floor=sigma_floor,
            sigma_ceiling=sigma_ceiling,
            local_energy=local_energy_,
        )

        # Twin annihilation near singularities: radiates locally as max-uncertainty light
        is_bh = sigma_ < sigma_bh
        if np.any(is_bh):
            xb = x_[is_bh]
            d = softnorm(x_[:, None, :] - xb[None, :, :], eps=1e-6)
            near = np.min(d, axis=1) < r_annihilate
            near = near & (~is_bh)

            if np.any(near):
                crossed = np.flatnonzero(near)
                kill = np.unique(np.concatenate([crossed, twin[crossed]]))

                core_idx = np.argmin(d[crossed], axis=1)
                core_pos = xb[core_idx]

                jitter = 0.06 * rng.normal(size=(crossed.size, dims))
                x_[crossed] = core_pos + jitter
                x_[twin[crossed]] = x_[crossed] + 0.01 * rng.normal(size=(crossed.size, dims))

                v_[kill] = 0.08 * rng.normal(size=(kill.size, dims))
                sigma_[kill] = sigma_ceiling
                spin_[kill] = rng.uniform(-np.pi, np.pi, kill.size)

                v_[twin[crossed]] = -v_[crossed]
                sigma_[twin[crossed]] = sigma_[crossed]
                spin_[twin[crossed]] = (spin_[crossed] + np.pi + 2 * np.pi) % (2 * np.pi) - np.pi

                rkill = softnorm(x_[kill], eps=1e-6)
                out2 = rkill > R
                if np.any(out2):
                    x_[kill[out2]] = (R * 0.98) * x_[kill[out2]] / rkill[out2][:, None]

        # Spin evolution: opposite on the backsheet
        spin_ = (spin_ + (tdir * dt) * (0.5 + 0.15 * certainty(sigma_)))
        spin_ = (spin_ + np.pi) % (2 * np.pi) - np.pi

        # Re-pin singularities (stay frozen/high-certainty)
        x_[sing_idx] = sing_pos
        v_[sing_idx] = 0.0
        sigma_[sing_idx] = sigma_sing

        return x_, v_, sigma_, spin_, v_drift_

    if not animate:
        v_drift = np.zeros_like(x)
        for _ in range(n_steps):
            x, v, sigma, spin, v_drift = evolve_one_step(x, v, sigma, spin)
        if print_diagnostics:
            metrics = universe_diagnostics(x, v + v_drift, sigma, sigma_floor=sigma_floor)
            print("Diagnostics:", metrics)
        fig = dashboard(x, v + v_drift, sigma, layer, R=R, step=n_steps)
        finalize_figure(fig, output_path=output_path, show=show_plots)
        return

    # -----------------
    # Animation dashboard
    # -----------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Precompute first derived quantities
    v_drift0 = horizon_drift(x, R=R, v0=horizon_v0, alpha=horizon_alpha)
    x_proj0 = x[:, :2]
    v_proj0 = (v + v_drift0)[:, :2]
    r0 = softnorm(x_proj0, eps=1e-6)
    rhat0 = np.where(r0[:, None] > 0, x_proj0 / r0[:, None], 0.0)
    vr0 = np.sum(v_proj0 * rhat0, axis=1)
    vt0 = v_proj0[:, 0] * (-rhat0[:, 1]) + v_proj0[:, 1] * (rhat0[:, 0])

    # (1) positions
    m0 = layer == 0
    a0 = layer == 1
    sc_m = ax[0, 0].scatter(x_proj0[m0, 0], x_proj0[m0, 1], c=np.log10(sigma[m0]), s=12, cmap="viridis",
                           vmin=np.log10(sigma_floor), vmax=np.log10(sigma_ceiling), alpha=0.9, label="matter")
    sc_a = ax[0, 0].scatter(x_proj0[a0, 0], x_proj0[a0, 1], c=np.log10(sigma[a0]), s=10, cmap="viridis",
                           vmin=np.log10(sigma_floor), vmax=np.log10(sigma_ceiling), alpha=0.5, marker="x", label="antimatter (back)")
    ax[0, 0].add_patch(plt.Circle((0, 0), R, fill=False, ec="#94a3b8", lw=1.5, alpha=0.7))
    ax[0, 0].set_aspect("equal", "box")
    ax[0, 0].set_title("Particle universe")
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_ylabel("y")
    ax[0, 0].legend(loc="upper right")
    cb = fig.colorbar(sc_m, ax=ax[0, 0])
    cb.set_label(r"$\log_{10}(\sigma)$")

    # (2) radial speed vs radius
    scat_vr = ax[0, 1].scatter(r0, vr0, s=8, alpha=0.7)
    ax[0, 1].set_title("Radial speed vs radius")
    ax[0, 1].set_xlabel("r")
    ax[0, 1].set_ylabel(r"$v_r$")

    # (3) tangential speed vs radius
    scat_vt = ax[1, 0].scatter(r0, np.abs(vt0), s=8, alpha=0.7)
    ax[1, 0].set_title("Tangential speed vs radius")
    ax[1, 0].set_xlabel("r")
    ax[1, 0].set_ylabel(r"$|v_t|$")

    # (4) sigma distribution
    ax[1, 1].set_title("Uncertainty distribution")
    ax[1, 1].set_xlabel(r"$\log_{10}(\sigma)$")
    ax[1, 1].set_ylabel("count")
    hist_bins = np.linspace(np.log10(sigma_floor), np.log10(sigma_ceiling), 32)
    ax[1, 1].hist(np.log10(sigma), bins=hist_bins, color="#60a5fa", alpha=0.8)

    title = fig.suptitle("", color="#e2e8f0")

    def update(frame_idx: int):
        nonlocal x, v, sigma, spin

        v_drift = np.zeros_like(x)
        for _ in range(substeps_per_frame):
            x, v, sigma, spin, v_drift = evolve_one_step(x, v, sigma, spin)

        # Update scatter positions + colors
        m = layer == 0
        a = layer == 1
        sc_m.set_offsets(x[m, :2])
        sc_m.set_array(np.log10(sigma[m]))
        sc_a.set_offsets(x[a, :2])
        sc_a.set_array(np.log10(sigma[a]))

        # Update derived velocity diagnostics
        x_proj = x[:, :2]
        v_tot = (v + v_drift)[:, :2]
        r = softnorm(x_proj, eps=1e-6)
        rhat = np.where(r[:, None] > 0, x_proj / r[:, None], 0.0)
        vr = np.sum(v_tot * rhat, axis=1)
        vt = v_tot[:, 0] * (-rhat[:, 1]) + v_tot[:, 1] * (rhat[:, 0])

        scat_vr.set_offsets(np.column_stack([r, vr]))
        scat_vt.set_offsets(np.column_stack([r, np.abs(vt)]))

        # Update histogram by clearing and redrawing (simple + robust)
        ax[1, 1].cla()
        ax[1, 1].set_title("Uncertainty distribution")
        ax[1, 1].set_xlabel(r"$\log_{10}(\sigma)$")
        ax[1, 1].set_ylabel("count")
        ax[1, 1].hist(np.log10(sigma), bins=hist_bins, color="#60a5fa", alpha=0.8)

        if print_diagnostics and frame_idx % max(1, n_frames // 6) == 0:
            metrics = universe_diagnostics(x, v_tot, sigma, sigma_floor=sigma_floor)
            print(f"Diagnostics @ frame {frame_idx}:", metrics)
        title.set_text(f"Unified Simple Field Theory (frame {frame_idx}/{n_frames})")
        return (sc_m, sc_a, scat_vr, scat_vt, title)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)

    # Keep a reference to the animation object so it isn't garbage-collected.
    # (If it's collected, matplotlib may display only a static final frame.)
    _ = ani

    plt.tight_layout()
    finalize_figure(fig, output_path=output_path, show=show_plots)


if __name__ == "__main__":
    main()
