"""
pde/dataset_builder.py
======================

Module autonome de construction de dataset pour l'étude :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce module permet de générer directement des datasets de trajectoires sans
supposer que les autres modules du projet sont déjà installés comme package.
Il contient donc une version autoportée des briques nécessaires :

- génération de conditions initiales,
- résolution numérique simple de Burgers,
- calcul des observables physiques de référence,
- assemblage et sauvegarde des données.

Pourquoi une version autoportée ?
---------------------------------
Dans le cadre d'un prototypage rapide ou d'un échange de fichiers, il est
pratique d'avoir un fichier unique capable de fonctionner tout seul. Cela ne
remplace pas l'architecture modulaire du projet, mais cela permet de démarrer
sans friction.

Contenu principal
-----------------
Ce module fournit :

1. une fonction pour générer une trajectoire complète avec métadonnées,
2. une fonction pour générer un split entier (train / val / test),
3. des fonctions simples de sauvegarde / chargement,
4. un résumé statistique du dataset,
5. des sanity checks de base.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np


# =============================================================================
# A. UTILITAIRES GENERAUX
# =============================================================================

def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Retourne un générateur numpy, en crée un si nécessaire."""
    return rng if rng is not None else np.random.default_rng()


def _validate_grid(x: np.ndarray) -> np.ndarray:
    """Vérifie que `x` est une grille 1D non vide."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` doit être un tableau 1D.")
    if x.size < 2:
        raise ValueError("`x` doit contenir au moins deux points.")
    return x


def _domain_length_from_grid(x: np.ndarray) -> float:
    """Retourne la longueur du domaine pour une grille uniforme endpoint=False."""
    x = _validate_grid(x)
    dx = x[1] - x[0]
    return float(x.size * dx)


def normalize_l2(u: np.ndarray, x: np.ndarray, target_norm: float = 1.0) -> np.ndarray:
    """Renormalise un champ pour lui imposer une norme L2 discrète cible."""
    x = _validate_grid(x)
    u = np.asarray(u, dtype=float)
    if u.shape != x.shape:
        raise ValueError("`u` et `x` doivent avoir la même forme.")

    dx = x[1] - x[0]
    current_norm = np.sqrt(np.sum(u ** 2) * dx)
    if current_norm < 1.0e-14:
        return u.copy()
    return (target_norm / current_norm) * u


# =============================================================================
# B. CONDITIONS INITIALES (VERSION AUTO-PORTEE)
# =============================================================================

def random_fourier_ic(
    x: np.ndarray,
    max_mode: int = 5,
    amplitude_scale: float = 1.0,
    zero_mean: bool = True,
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Condition initiale = combinaison aléatoire de modes de Fourier."""
    x = _validate_grid(x)
    rng = _get_rng(rng)
    if max_mode < 1:
        raise ValueError("`max_mode` doit être >= 1.")

    L = _domain_length_from_grid(x)
    u0 = np.zeros_like(x, dtype=float)
    for k in range(1, max_mode + 1):
        a_k = amplitude_scale * rng.uniform(-1.0, 1.0)
        b_k = amplitude_scale * rng.uniform(-1.0, 1.0)
        phase = 2.0 * np.pi * k * x / L
        u0 += a_k * np.sin(phase) + b_k * np.cos(phase)

    if not zero_mean:
        u0 += amplitude_scale * rng.uniform(-1.0, 1.0)

    if normalize:
        u0 = normalize_l2(u0, x, target_norm=target_norm)
    return u0


def _periodic_distance(x: np.ndarray, center: float, L: float) -> np.ndarray:
    """Distance périodique minimale entre x et un centre."""
    raw = np.abs(x - center)
    return np.minimum(raw, L - raw)


def gaussian_bumps_ic(
    x: np.ndarray,
    n_bumps: int = 2,
    amplitude_range: Sequence[float] = (-1.0, 1.0),
    width_range: Sequence[float] = (0.1, 0.4),
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Condition initiale = somme de bosses gaussiennes périodisées."""
    x = _validate_grid(x)
    rng = _get_rng(rng)
    if n_bumps < 1:
        raise ValueError("`n_bumps` doit être >= 1.")

    amin, amax = float(amplitude_range[0]), float(amplitude_range[1])
    wmin, wmax = float(width_range[0]), float(width_range[1])
    if wmin <= 0.0 or wmax <= 0.0:
        raise ValueError("Les largeurs doivent être strictement positives.")

    L = _domain_length_from_grid(x)
    u0 = np.zeros_like(x, dtype=float)
    for _ in range(n_bumps):
        center = rng.uniform(x[0], x[0] + L)
        amplitude = rng.uniform(amin, amax)
        sigma = rng.uniform(wmin, wmax) * L
        d = _periodic_distance(x, center, L)
        u0 += amplitude * np.exp(-(d ** 2) / (sigma ** 2))

    if normalize:
        u0 = normalize_l2(u0, x, target_norm=target_norm)
    return u0


def _enforce_hermitian_symmetry(coeffs: np.ndarray) -> np.ndarray:
    """Impose une symétrie hermitienne pour obtenir une ifft réelle."""
    coeffs = np.asarray(coeffs, dtype=complex).copy()
    N = coeffs.size
    coeffs[0] = coeffs[0].real + 0j
    if N % 2 == 0:
        coeffs[N // 2] = coeffs[N // 2].real + 0j
    for k in range(1, N // 2 + (0 if N % 2 == 0 else 1)):
        coeffs[-k] = np.conjugate(coeffs[k])
    return coeffs


def random_smooth_ic(
    x: np.ndarray,
    spectral_decay: float = 2.0,
    amplitude_scale: float = 1.0,
    zero_mean: bool = True,
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Condition initiale = champ aléatoire lisse construit spectralement."""
    x = _validate_grid(x)
    rng = _get_rng(rng)
    Nx = x.size

    modes = np.fft.fftfreq(Nx, d=1.0 / Nx)
    coeffs = rng.normal(size=Nx) + 1j * rng.normal(size=Nx)
    coeffs *= amplitude_scale * (1.0 + np.abs(modes)) ** (-spectral_decay)

    if zero_mean:
        coeffs[0] = 0.0
    coeffs = _enforce_hermitian_symmetry(coeffs)
    u0 = np.fft.ifft(coeffs).real

    if normalize:
        u0 = normalize_l2(u0, x, target_norm=target_norm)
    return u0


def sample_initial_condition(
    x: np.ndarray,
    ic_type: str = "fourier",
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> np.ndarray:
    """Interface unique pour générer une condition initiale."""
    x = _validate_grid(x)
    rng = _get_rng(rng)
    ic_type = ic_type.lower()

    # On filtre les kwargs selon la famille choisie, pour éviter les erreurs
    # quand on mutualise un même dictionnaire de paramètres.
    if ic_type == "fourier":
        allowed = {"max_mode", "amplitude_scale", "zero_mean"}
        family_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return random_fourier_ic(
            x,
            normalize=normalize,
            target_norm=target_norm,
            rng=rng,
            **family_kwargs,
        )

    if ic_type == "gaussian_bumps":
        allowed = {"n_bumps", "amplitude_range", "width_range"}
        family_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return gaussian_bumps_ic(
            x,
            normalize=normalize,
            target_norm=target_norm,
            rng=rng,
            **family_kwargs,
        )

    if ic_type == "random_smooth":
        allowed = {"spectral_decay", "amplitude_scale", "zero_mean"}
        family_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return random_smooth_ic(
            x,
            normalize=normalize,
            target_norm=target_norm,
            rng=rng,
            **family_kwargs,
        )

    raise ValueError(
        f"Type de condition initiale inconnu : {ic_type!r}. "
        "Choisir parmi {'fourier', 'gaussian_bumps', 'random_smooth'}."
    )


def sample_initial_condition_from_list(
    x: np.ndarray,
    ic_types: Sequence[str],
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> tuple[np.ndarray, str]:
    """Tire aléatoirement un type de CI parmi une liste puis génère le champ."""
    if len(ic_types) == 0:
        raise ValueError("La séquence `ic_types` ne doit pas être vide.")
    rng = _get_rng(rng)
    chosen_type = str(rng.choice(list(ic_types)))
    u0 = sample_initial_condition(
        x,
        ic_type=chosen_type,
        normalize=normalize,
        target_norm=target_norm,
        rng=rng,
        **kwargs,
    )
    return u0, chosen_type


# =============================================================================
# C. SOLVEUR BURGERS (VERSION AUTO-PORTEE)
# =============================================================================

def build_spatial_grid(L: float, Nx: int) -> tuple[np.ndarray, float]:
    """Construit une grille uniforme périodique sur [0, L)."""
    if L <= 0.0:
        raise ValueError("`L` doit être strictement positif.")
    if Nx < 4:
        raise ValueError("`Nx` doit être >= 4.")
    x = np.linspace(0.0, L, Nx, endpoint=False)
    dx = x[1] - x[0]
    return x, float(dx)


def compute_first_derivative_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    """Approxime u_x par différences finies centrées périodiques."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("`u` doit être 1D.")
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def compute_second_derivative_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    """Approxime u_xx par différences finies centrées périodiques."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("`u` doit être 1D.")
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx ** 2)


def burgers_rhs(u: np.ndarray, dx: float, nu: float) -> np.ndarray:
    """Second membre de Burgers : u_t = -u u_x + nu u_xx."""
    ux = compute_first_derivative_periodic(u, dx)
    uxx = compute_second_derivative_periodic(u, dx)
    return -u * ux + nu * uxx


def step_euler_explicit(u: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """Un pas en temps par Euler explicite."""
    return u + dt * burgers_rhs(u, dx, nu)


def step_rk4(u: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """Un pas en temps par Runge-Kutta d'ordre 4."""
    k1 = burgers_rhs(u, dx, nu)
    k2 = burgers_rhs(u + 0.5 * dt * k1, dx, nu)
    k3 = burgers_rhs(u + 0.5 * dt * k2, dx, nu)
    k4 = burgers_rhs(u + dt * k3, dx, nu)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def estimate_stability_indicators(u: np.ndarray, dt: float, dx: float, nu: float) -> Dict[str, float]:
    """Calcule quelques indicateurs simples de stabilité / raideur."""
    u = np.asarray(u, dtype=float)
    max_abs_u = float(np.max(np.abs(u)))
    advective_cfl = float(dt * max_abs_u / dx) if dx > 0.0 else np.inf
    diffusive_number = float(nu * dt / (dx ** 2)) if dx > 0.0 else np.inf
    return {
        "max_abs_u": max_abs_u,
        "advective_cfl": advective_cfl,
        "diffusive_number": diffusive_number,
    }


def solve_burgers(
    u0: np.ndarray,
    L: float,
    T: float,
    dt: float,
    nu: float,
    save_every: int = 1,
    time_scheme: str = "rk4",
    check_nan: bool = True,
    return_full_grid: bool = True,
) -> Dict[str, Any]:
    """Résout numériquement Burgers sur [0, L] périodique."""
    u0 = np.asarray(u0, dtype=float)
    if u0.ndim != 1:
        raise ValueError("`u0` doit être 1D.")
    if T <= 0.0 or dt <= 0.0:
        raise ValueError("`T` et `dt` doivent être strictement positifs.")
    if save_every < 1:
        raise ValueError("`save_every` doit être >= 1.")
    if nu < 0.0:
        raise ValueError("`nu` doit être positive ou nulle.")

    Nx = u0.size
    x, dx = build_spatial_grid(L, Nx)

    n_steps = int(np.round(T / dt))
    if n_steps < 1:
        raise ValueError("Le nombre de pas de temps doit être >= 1.")

    scheme = time_scheme.lower()
    if scheme == "euler":
        step_function = step_euler_explicit
    elif scheme == "rk4":
        step_function = step_rk4
    else:
        raise ValueError("`time_scheme` doit être 'euler' ou 'rk4'.")

    n_saved = n_steps // save_every + 1
    states = np.empty((n_saved, Nx), dtype=float)
    times = np.empty(n_saved, dtype=float)

    u = u0.copy()
    states[0] = u
    times[0] = 0.0
    save_id = 1
    stability_indicators_initial = estimate_stability_indicators(u, dt, dx, nu)

    for n in range(1, n_steps + 1):
        u = step_function(u, dt, dx, nu)

        if check_nan and (not np.all(np.isfinite(u))):
            raise FloatingPointError(
                f"Des valeurs non finies ont été détectées au pas de temps {n}."
            )

        if n % save_every == 0:
            states[save_id] = u
            times[save_id] = n * dt
            save_id += 1

    result: Dict[str, Any] = {
        "times": times,
        "states": states,
        "stability_indicators_initial": stability_indicators_initial,
    }
    if return_full_grid:
        result["x"] = x
        result["dx"] = dx
    return result


# =============================================================================
# D. OBSERVABLES (VERSION AUTO-PORTEE)
# =============================================================================

def compute_energy(u: np.ndarray, dx: float) -> float:
    """Énergie L2 discrète : 1/2 * somme(u^2) * dx."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("`u` doit être 1D.")
    return 0.5 * float(np.sum(u ** 2) * dx)


def compute_energy_trajectory(states: np.ndarray, dx: float) -> np.ndarray:
    """Énergie pour tous les snapshots d'une trajectoire."""
    states = np.asarray(states, dtype=float)
    if states.ndim != 2:
        raise ValueError("`states` doit être 2D de forme (n_times, Nx).")
    return 0.5 * np.sum(states ** 2, axis=1) * dx


def compute_spatial_gradient(u: np.ndarray, dx: float) -> np.ndarray:
    """Gradient spatial périodique par différences finies centrées."""
    return compute_first_derivative_periodic(u, dx)


def compute_dissipation(u: np.ndarray, dx: float) -> float:
    """Dissipation : somme(|u_x|^2) * dx."""
    ux = compute_spatial_gradient(u, dx)
    return float(np.sum(ux ** 2) * dx)


def compute_dissipation_trajectory(states: np.ndarray, dx: float) -> np.ndarray:
    """Dissipation pour tous les snapshots d'une trajectoire."""
    states = np.asarray(states, dtype=float)
    if states.ndim != 2:
        raise ValueError("`states` doit être 2D de forme (n_times, Nx).")
    out = np.empty(states.shape[0], dtype=float)
    for n in range(states.shape[0]):
        out[n] = compute_dissipation(states[n], dx)
    return out


def compute_monotonicity_violations(values: np.ndarray) -> np.ndarray:
    """Violations positives d'une monotonie décroissante."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("`values` doit être 1D.")
    diffs = values[1:] - values[:-1]
    return np.maximum(0.0, diffs)


def compute_monotonicity_metrics(values: np.ndarray, atol: float = 0.0) -> Dict[str, Any]:
    """Métriques simples de monotonie décroissante."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("`values` doit être 1D et contenir au moins deux valeurs.")

    diffs = values[1:] - values[:-1]
    positive_violations = np.maximum(0.0, diffs - atol)
    return {
        "n_steps": int(values.size - 1),
        "monotone_fraction": float(np.mean(diffs <= atol)),
        "mean_positive_violation": float(np.mean(positive_violations)),
        "max_positive_violation": float(np.max(positive_violations)),
        "is_nonincreasing": bool(np.all(diffs <= atol)),
    }


def compute_reference_observables(states: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    """Calcule les observables physiques de référence pour une trajectoire."""
    return {
        "energy": compute_energy_trajectory(states, dx),
        "dissipation": compute_dissipation_trajectory(states, dx),
    }


# =============================================================================
# E. CONSTRUCTION DU DATASET
# =============================================================================

def save_dataset_npz(path: str | Path, dataset: Dict[str, Any]) -> None:
    """Sauvegarde un dataset dans un fichier .npz compressé."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in dataset.items():
        serializable[key] = np.asarray(value) if isinstance(value, list) else value
    np.savez_compressed(path, **serializable)


def load_dataset_npz(path: str | Path) -> Dict[str, Any]:
    """Recharge un dataset sauvegardé au format .npz."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_ic_type_mapping(ic_types: Sequence[str]) -> Dict[str, int]:
    """Construit un dictionnaire nom -> identifiant entier pour les familles de CI."""
    if len(ic_types) == 0:
        raise ValueError("La séquence `ic_types` ne doit pas être vide.")
    return {name: idx for idx, name in enumerate(ic_types)}


def generate_single_trajectory(
    x: np.ndarray,
    L: float,
    T: float,
    dt: float,
    nu: float,
    save_every: int,
    time_scheme: str,
    ic_type: Optional[str] = None,
    allowed_ic_types: Optional[Sequence[str]] = None,
    normalize_ic: bool = False,
    target_ic_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    ic_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Génère une trajectoire unique complète avec observables et métadonnées."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` doit être un tableau 1D.")
    if ic_kwargs is None:
        ic_kwargs = {}
    rng = _get_rng(rng)

    # 1) Condition initiale
    if ic_type is not None:
        chosen_ic_type = str(ic_type)
        u0 = sample_initial_condition(
            x,
            ic_type=chosen_ic_type,
            normalize=normalize_ic,
            target_norm=target_ic_norm,
            rng=rng,
            **ic_kwargs,
        )
    else:
        if allowed_ic_types is None or len(allowed_ic_types) == 0:
            raise ValueError("Si `ic_type` est None, fournir `allowed_ic_types` non vide.")
        u0, chosen_ic_type = sample_initial_condition_from_list(
            x,
            ic_types=allowed_ic_types,
            normalize=normalize_ic,
            target_norm=target_ic_norm,
            rng=rng,
            **ic_kwargs,
        )

    # 2) Résolution de Burgers
    solution = solve_burgers(
        u0=u0,
        L=L,
        T=T,
        dt=dt,
        nu=nu,
        save_every=save_every,
        time_scheme=time_scheme,
        check_nan=True,
        return_full_grid=True,
    )

    states = solution["states"]
    times = solution["times"]
    dx = float(solution["dx"])

    # 3) Observables de référence
    observables = compute_reference_observables(states, dx)
    energy_monotonicity = compute_monotonicity_metrics(observables["energy"], atol=1.0e-12)

    return {
        "u0": u0,
        "states": states,
        "times": times,
        "x": solution["x"],
        "dx": dx,
        "nu": float(nu),
        "ic_type": chosen_ic_type,
        "energy": observables["energy"],
        "dissipation": observables["dissipation"],
        "stability_indicators_initial": solution["stability_indicators_initial"],
        "energy_monotonicity": energy_monotonicity,
    }


def generate_dataset_split(
    n_trajectories: int,
    x: np.ndarray,
    L: float,
    T: float,
    dt: float,
    nu: float,
    save_every: int,
    time_scheme: str,
    ic_types: Sequence[str],
    normalize_ic: bool = False,
    target_ic_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    ic_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Génère un split complet du dataset (train / val / test)."""
    if n_trajectories < 1:
        raise ValueError("`n_trajectories` doit être >= 1.")

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` doit être un tableau 1D.")

    rng = _get_rng(rng)
    if ic_kwargs is None:
        ic_kwargs = {}

    ic_type_to_id = build_ic_type_mapping(ic_types)

    # Première trajectoire pour inférer les dimensions de sortie
    first = generate_single_trajectory(
        x=x,
        L=L,
        T=T,
        dt=dt,
        nu=nu,
        save_every=save_every,
        time_scheme=time_scheme,
        ic_type=None,
        allowed_ic_types=ic_types,
        normalize_ic=normalize_ic,
        target_ic_norm=target_ic_norm,
        rng=rng,
        ic_kwargs=ic_kwargs,
    )

    states0 = first["states"]
    times = first["times"]
    x_out = first["x"]
    dx = float(first["dx"])
    n_times, Nx = states0.shape

    states = np.empty((n_trajectories, n_times, Nx), dtype=float)
    energy = np.empty((n_trajectories, n_times), dtype=float)
    dissipation = np.empty((n_trajectories, n_times), dtype=float)
    u0_all = np.empty((n_trajectories, Nx), dtype=float)
    nu_values = np.empty(n_trajectories, dtype=float)
    ic_type_ids = np.empty(n_trajectories, dtype=int)

    # Diagnostics simples par trajectoire
    initial_max_abs_u = np.empty(n_trajectories, dtype=float)
    initial_advective_cfl = np.empty(n_trajectories, dtype=float)
    initial_diffusive_number = np.empty(n_trajectories, dtype=float)
    energy_monotone_fraction = np.empty(n_trajectories, dtype=float)
    energy_mean_positive_violation = np.empty(n_trajectories, dtype=float)

    # Remplissage avec la première trajectoire
    states[0] = first["states"]
    energy[0] = first["energy"]
    dissipation[0] = first["dissipation"]
    u0_all[0] = first["u0"]
    nu_values[0] = first["nu"]
    ic_type_ids[0] = ic_type_to_id[first["ic_type"]]

    initial_max_abs_u[0] = first["stability_indicators_initial"]["max_abs_u"]
    initial_advective_cfl[0] = first["stability_indicators_initial"]["advective_cfl"]
    initial_diffusive_number[0] = first["stability_indicators_initial"]["diffusive_number"]
    energy_monotone_fraction[0] = first["energy_monotonicity"]["monotone_fraction"]
    energy_mean_positive_violation[0] = first["energy_monotonicity"]["mean_positive_violation"]

    if verbose:
        print(f"[dataset_builder] Trajectoire 1 / {n_trajectories} générée.")

    for i in range(1, n_trajectories):
        traj = generate_single_trajectory(
            x=x,
            L=L,
            T=T,
            dt=dt,
            nu=nu,
            save_every=save_every,
            time_scheme=time_scheme,
            ic_type=None,
            allowed_ic_types=ic_types,
            normalize_ic=normalize_ic,
            target_ic_norm=target_ic_norm,
            rng=rng,
            ic_kwargs=ic_kwargs,
        )

        states[i] = traj["states"]
        energy[i] = traj["energy"]
        dissipation[i] = traj["dissipation"]
        u0_all[i] = traj["u0"]
        nu_values[i] = traj["nu"]
        ic_type_ids[i] = ic_type_to_id[traj["ic_type"]]

        initial_max_abs_u[i] = traj["stability_indicators_initial"]["max_abs_u"]
        initial_advective_cfl[i] = traj["stability_indicators_initial"]["advective_cfl"]
        initial_diffusive_number[i] = traj["stability_indicators_initial"]["diffusive_number"]
        energy_monotone_fraction[i] = traj["energy_monotonicity"]["monotone_fraction"]
        energy_mean_positive_violation[i] = traj["energy_monotonicity"]["mean_positive_violation"]

        if verbose and ((i + 1) % max(1, n_trajectories // 10) == 0 or (i + 1) == n_trajectories):
            print(f"[dataset_builder] Trajectoire {i + 1} / {n_trajectories} générée.")

    dataset = {
        "states": states,
        "times": times,
        "energy": energy,
        "dissipation": dissipation,
        "u0": u0_all,
        "x": x_out,
        "dx": np.array(dx, dtype=float),
        "nu_values": nu_values,
        "ic_type_ids": ic_type_ids,
        "ic_type_names": np.asarray(list(ic_type_to_id.keys())),
        "initial_max_abs_u": initial_max_abs_u,
        "initial_advective_cfl": initial_advective_cfl,
        "initial_diffusive_number": initial_diffusive_number,
        "energy_monotone_fraction": energy_monotone_fraction,
        "energy_mean_positive_violation": energy_mean_positive_violation,
    }
    return dataset


def build_dataset_from_config(
    config: Any,
    split: str,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Path]:
    """Génère un split du dataset à partir de l'objet de configuration global."""
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("`split` doit être 'train', 'val' ou 'test'.")

    if split == "train":
        n_trajectories = int(config.dataset.n_train)
        filename = config.dataset.train_filename
    elif split == "val":
        n_trajectories = int(config.dataset.n_val)
        filename = config.dataset.val_filename
    else:
        n_trajectories = int(config.dataset.n_test)
        filename = config.dataset.test_filename

    L = float(config.pde.L)
    Nx = int(config.pde.Nx)
    x = np.linspace(0.0, L, Nx, endpoint=False)

    dataset = generate_dataset_split(
        n_trajectories=n_trajectories,
        x=x,
        L=L,
        T=float(config.pde.T),
        dt=float(config.pde.dt),
        nu=float(config.pde.nu),
        save_every=int(config.pde.save_every),
        time_scheme=str(config.pde.time_scheme),
        ic_types=list(config.dataset.ic_types),
        normalize_ic=True,
        target_ic_norm=1.0,
        rng=rng,
        ic_kwargs={
            "max_mode": int(config.dataset.max_fourier_mode),
            "amplitude_scale": float(config.dataset.fourier_amplitude_scale),
            "n_bumps": int(config.dataset.n_gaussian_bumps),
            "spectral_decay": float(config.dataset.spectral_decay),
        },
        verbose=verbose,
    )

    output_path = Path(config.paths.data_dir) / filename
    return dataset, output_path


def summarize_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Retourne un résumé simple et lisible du dataset généré."""
    states = np.asarray(dataset["states"], dtype=float)
    energy = np.asarray(dataset["energy"], dtype=float)
    dissipation = np.asarray(dataset["dissipation"], dtype=float)
    ic_type_ids = np.asarray(dataset["ic_type_ids"], dtype=int)
    ic_type_names = np.asarray(dataset["ic_type_names"])

    if states.ndim != 3:
        raise ValueError("`dataset['states']` doit être de dimension 3 : (n_traj, n_times, Nx).")

    n_traj, n_times, Nx = states.shape
    unique_ids, counts = np.unique(ic_type_ids, return_counts=True)
    ic_distribution = {
        str(ic_type_names[idx]): int(count)
        for idx, count in zip(unique_ids, counts)
    }

    return {
        "n_trajectories": int(n_traj),
        "n_times": int(n_times),
        "Nx": int(Nx),
        "global_min_state": float(np.min(states)),
        "global_max_state": float(np.max(states)),
        "mean_initial_energy": float(np.mean(energy[:, 0])),
        "mean_final_energy": float(np.mean(energy[:, -1])),
        "mean_dissipation": float(np.mean(dissipation)),
        "ic_distribution": ic_distribution,
        "mean_energy_monotone_fraction": float(np.mean(dataset["energy_monotone_fraction"])),
        "mean_energy_positive_violation": float(np.mean(dataset["energy_mean_positive_violation"])),
    }


def basic_dataset_sanity_checks(dataset: Dict[str, Any]) -> Dict[str, bool]:
    """Effectue quelques tests très simples de cohérence structurelle."""
    states = np.asarray(dataset["states"])
    energy = np.asarray(dataset["energy"])
    dissipation = np.asarray(dataset["dissipation"])
    times = np.asarray(dataset["times"])
    x = np.asarray(dataset["x"])

    return {
        "states_is_finite": bool(np.all(np.isfinite(states))),
        "energy_is_finite": bool(np.all(np.isfinite(energy))),
        "dissipation_is_finite": bool(np.all(np.isfinite(dissipation))),
        "times_is_sorted": bool(np.all(np.diff(times) > 0.0)),
        "x_is_sorted": bool(np.all(np.diff(x) > 0.0)),
        "energy_nonnegative": bool(np.all(energy >= -1.0e-14)),
        "dissipation_nonnegative": bool(np.all(dissipation >= -1.0e-14)),
    }


def build_and_save_split_from_config(
    config: Any,
    split: str,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Pipeline pratique : génère un split, le sauvegarde et retourne le dataset."""
    dataset, output_path = build_dataset_from_config(
        config=config,
        split=split,
        rng=rng,
        verbose=verbose,
    )
    save_dataset_npz(output_path, dataset)
    if verbose:
        print(f"[dataset_builder] Split '{split}' sauvegardé dans : {output_path}")
    return dataset


# =============================================================================
# F. BLOC DE TEST MINIMAL
# =============================================================================

if __name__ == "__main__":
    L = 2.0 * np.pi
    Nx = 64
    T = 0.5
    dt = 1.0e-3
    nu = 0.05
    save_every = 20
    x = np.linspace(0.0, L, Nx, endpoint=False)
    rng = np.random.default_rng(123)

    dataset = generate_dataset_split(
        n_trajectories=5,
        x=x,
        L=L,
        T=T,
        dt=dt,
        nu=nu,
        save_every=save_every,
        time_scheme="rk4",
        ic_types=["fourier", "gaussian_bumps", "random_smooth"],
        normalize_ic=False,
        target_ic_norm=1.0,
        rng=rng,
        ic_kwargs={
            "max_mode": 4,
            "amplitude_scale": 1.0,
            "n_bumps": 2,
            "spectral_decay": 2.0,
        },
        verbose=True,
    )

    print("\nRésumé du dataset de test :")
    summary = summarize_dataset(dataset)
    for key, value in summary.items():
        print(f"- {key}: {value}")

    print("\nSanity checks :")
    checks = basic_dataset_sanity_checks(dataset)
    for key, value in checks.items():
        print(f"- {key}: {value}")
