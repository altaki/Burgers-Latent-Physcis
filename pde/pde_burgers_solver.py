"""
pde/burgers_solver.py
=====================

Solveur numérique simple, lisible et commenté pour l'équation de Burgers
visqueuse 1D sur un domaine périodique :

    u_t + u u_x = nu * u_xx

Objectif
--------
Ce module fournit une première implémentation robuste et pédagogique pour :

1. simuler des trajectoires de l'équation de Burgers visqueuse,
2. générer des snapshots utilisables pour le dataset,
3. vérifier qualitativement la dissipation d'énergie,
4. servir de base claire pour de futures améliorations numériques.

Choix numériques de cette première version
------------------------------------------
- Domaine 1D périodique.
- Dérivées spatiales par différences finies centrées.
- Schémas temporels disponibles :
    * Euler explicite
    * Runge-Kutta d'ordre 4 (RK4)
- Code volontairement simple et lisible, plutôt que très optimisé.

Remarque importante
-------------------
Pour des viscosités très faibles ou des conditions initiales très raides,
un schéma explicite peut devenir délicat numériquement. Pour la première
version du projet, cela reste cependant un bon compromis entre simplicité
et contrôle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np


# -----------------------------------------------------------------------------
# 1. Utilitaires de validation
# -----------------------------------------------------------------------------

def _validate_state(u: np.ndarray) -> np.ndarray:
    """
    Vérifie qu'un état spatial est un tableau 1D.

    Paramètres
    ----------
    u : np.ndarray
        Snapshot spatial supposé de forme (Nx,).

    Retour
    ------
    np.ndarray
        Tableau converti en float.
    """
    u = np.asarray(u, dtype=float)

    if u.ndim != 1:
        raise ValueError("`u` doit être un tableau 1D de forme (Nx,).")

    return u


# -----------------------------------------------------------------------------
# 2. Grille spatiale
# -----------------------------------------------------------------------------

def build_spatial_grid(L: float, Nx: int) -> tuple[np.ndarray, float]:
    """
    Construit la grille spatiale 1D uniforme sur [0, L) avec convention périodique.

    Paramètres
    ----------
    L : float
        Longueur du domaine spatial.
    Nx : int
        Nombre de points en espace.

    Retour
    ------
    tuple[np.ndarray, float]
        - x  : grille spatiale de forme (Nx,)
        - dx : pas d'espace
    """
    if L <= 0.0:
        raise ValueError("`L` doit être strictement positif.")
    if Nx < 4:
        raise ValueError("`Nx` doit être au moins égal à 4.")

    x = np.linspace(0.0, L, Nx, endpoint=False)
    dx = x[1] - x[0]
    return x, float(dx)


# -----------------------------------------------------------------------------
# 3. Dérivées spatiales périodiques
# -----------------------------------------------------------------------------

def compute_first_derivative_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Approxime la dérivée première u_x par différences finies centrées
    avec conditions périodiques.

    Formule :
        u_x(x_j) ≈ [u_{j+1} - u_{j-1}] / (2 dx)

    Paramètres
    ----------
    u : np.ndarray
        Tableau 1D de forme (Nx,).
    dx : float
        Pas d'espace.

    Retour
    ------
    np.ndarray
        Approximation de u_x.
    """
    u = _validate_state(u)
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)



def compute_second_derivative_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Approxime la dérivée seconde u_xx par différences finies centrées
    avec conditions périodiques.

    Formule :
        u_xx(x_j) ≈ [u_{j+1} - 2u_j + u_{j-1}] / dx^2

    Paramètres
    ----------
    u : np.ndarray
        Tableau 1D de forme (Nx,).
    dx : float
        Pas d'espace.

    Retour
    ------
    np.ndarray
        Approximation de u_xx.
    """
    u = _validate_state(u)
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx ** 2)


# -----------------------------------------------------------------------------
# 4. Second membre de Burgers
# -----------------------------------------------------------------------------

def burgers_rhs(u: np.ndarray, dx: float, nu: float) -> np.ndarray:
    """
    Calcule le second membre de l'équation de Burgers visqueuse :

        u_t = - u u_x + nu * u_xx

    Paramètres
    ----------
    u : np.ndarray
        État spatial courant de forme (Nx,).
    dx : float
        Pas d'espace.
    nu : float
        Viscosité.

    Retour
    ------
    np.ndarray
        Tableau 1D représentant du/dt.
    """
    u = _validate_state(u)

    if nu < 0.0:
        raise ValueError("La viscosité `nu` doit être positive ou nulle.")

    ux = compute_first_derivative_periodic(u, dx)
    uxx = compute_second_derivative_periodic(u, dx)

    return -u * ux + nu * uxx


# -----------------------------------------------------------------------------
# 5. Schémas en temps
# -----------------------------------------------------------------------------

def step_euler_explicit(u: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """
    Effectue un pas de temps par Euler explicite.

    Paramètres
    ----------
    u : np.ndarray
        État courant.
    dt : float
        Pas de temps.
    dx : float
        Pas d'espace.
    nu : float
        Viscosité.

    Retour
    ------
    np.ndarray
        Nouvel état après un pas de temps.
    """
    return u + dt * burgers_rhs(u, dx, nu)



def step_rk4(u: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """
    Effectue un pas de temps avec le schéma de Runge-Kutta d'ordre 4.

    Paramètres
    ----------
    u : np.ndarray
        État courant.
    dt : float
        Pas de temps.
    dx : float
        Pas d'espace.
    nu : float
        Viscosité.

    Retour
    ------
    np.ndarray
        Nouvel état après un pas de temps.
    """
    k1 = burgers_rhs(u, dx, nu)
    k2 = burgers_rhs(u + 0.5 * dt * k1, dx, nu)
    k3 = burgers_rhs(u + 0.5 * dt * k2, dx, nu)
    k4 = burgers_rhs(u + dt * k3, dx, nu)

    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# -----------------------------------------------------------------------------
# 6. Choix du pas de temps
# -----------------------------------------------------------------------------

def estimate_stability_indicators(u: np.ndarray, dt: float, dx: float, nu: float) -> Dict[str, float]:
    """
    Calcule quelques indicateurs simples de stabilité / raideur.

    Attention
    ---------
    Ces quantités ne sont pas des garanties strictes de stabilité, mais elles
    permettent de détecter des situations potentiellement problématiques.

    Indicateurs retournés
    ---------------------
    - max_abs_u : max |u|
    - advective_cfl : dt * max|u| / dx
    - diffusive_number : nu * dt / dx^2

    Paramètres
    ----------
    u : np.ndarray
        État courant.
    dt : float
        Pas de temps.
    dx : float
        Pas d'espace.
    nu : float
        Viscosité.

    Retour
    ------
    Dict[str, float]
        Dictionnaire d'indicateurs.
    """
    u = _validate_state(u)
    max_abs_u = float(np.max(np.abs(u)))

    # Si u est identiquement nul, le CFL advectif vaut 0.
    advective_cfl = float(dt * max_abs_u / dx) if dx > 0.0 else np.inf
    diffusive_number = float(nu * dt / (dx ** 2)) if dx > 0.0 else np.inf

    return {
        "max_abs_u": max_abs_u,
        "advective_cfl": advective_cfl,
        "diffusive_number": diffusive_number,
    }


# -----------------------------------------------------------------------------
# 7. Solveur principal
# -----------------------------------------------------------------------------

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
    """
    Résout numériquement l'équation de Burgers visqueuse sur [0, L] périodique.

    Paramètres
    ----------
    u0 : np.ndarray
        Condition initiale de forme (Nx,).
    L : float
        Longueur du domaine spatial.
    T : float
        Temps final de simulation.
    dt : float
        Pas de temps.
    nu : float
        Viscosité.
    save_every : int, optionnel
        Sauvegarde un snapshot tous les `save_every` pas de temps.
    time_scheme : str, optionnel
        Schéma en temps :
        - "euler"
        - "rk4"
    check_nan : bool, optionnel
        Si True, vérifie à chaque pas si des NaN / inf apparaissent.
    return_full_grid : bool, optionnel
        Si True, renvoie aussi la grille spatiale `x` et `dx`.

    Retour
    ------
    Dict[str, Any]
        Dictionnaire contenant au minimum :
        - "times"  : tableau des temps sauvegardés
        - "states" : tableau des états sauvegardés (n_saved, Nx)
        et éventuellement :
        - "x"      : grille spatiale
        - "dx"     : pas d'espace
        - "stability_indicators_initial" : indicateurs initiaux simples
    """
    u0 = _validate_state(u0)

    if T <= 0.0:
        raise ValueError("`T` doit être strictement positif.")
    if dt <= 0.0:
        raise ValueError("`dt` doit être strictement positif.")
    if save_every < 1:
        raise ValueError("`save_every` doit être >= 1.")
    if nu < 0.0:
        raise ValueError("`nu` doit être positive ou nulle.")

    Nx = u0.size
    x, dx = build_spatial_grid(L, Nx)

    n_steps = int(np.round(T / dt))
    if n_steps < 1:
        raise ValueError("Le nombre de pas de temps doit être >= 1.")

    # Pour éviter des incohérences flagrantes si T / dt n'est pas entier.
    implied_T = n_steps * dt
    if not np.isclose(implied_T, T, rtol=1e-10, atol=1e-12):
        # On choisit ici d'avertir par simple commentaire de code :
        # la simulation sera faite jusqu'au temps n_steps * dt.
        # Dans un projet plus avancé, on pourrait ajuster le dernier pas.
        T = implied_T

    # Choix du schéma temporel.
    scheme = time_scheme.lower()
    if scheme == "euler":
        step_function = step_euler_explicit
    elif scheme == "rk4":
        step_function = step_rk4
    else:
        raise ValueError("`time_scheme` doit être 'euler' ou 'rk4'.")

    # Nombre de snapshots sauvegardés, en incluant l'état initial.
    n_saved = n_steps // save_every + 1

    states = np.empty((n_saved, Nx), dtype=float)
    times = np.empty(n_saved, dtype=float)

    # Initialisation.
    u = u0.copy()
    states[0] = u
    times[0] = 0.0
    save_id = 1

    stability_indicators_initial = estimate_stability_indicators(u, dt, dx, nu)

    # Boucle en temps.
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


# -----------------------------------------------------------------------------
# 8. Interface pratique utilisant une configuration
# -----------------------------------------------------------------------------

def solve_burgers_from_config(u0: np.ndarray, config: Any) -> Dict[str, Any]:
    """
    Interface pratique pour lancer la résolution à partir d'un objet de
    configuration possédant au minimum les attributs suivants :

        config.pde.L
        config.pde.T
        config.pde.dt
        config.pde.nu
        config.pde.save_every
        config.pde.time_scheme

    Paramètres
    ----------
    u0 : np.ndarray
        Condition initiale.
    config : Any
        Objet de configuration compatible (par exemple l'objet ProjectConfig).

    Retour
    ------
    Dict[str, Any]
        Résultat renvoyé par `solve_burgers`.
    """
    return solve_burgers(
        u0=u0,
        L=float(config.pde.L),
        T=float(config.pde.T),
        dt=float(config.pde.dt),
        nu=float(config.pde.nu),
        save_every=int(config.pde.save_every),
        time_scheme=str(config.pde.time_scheme),
        check_nan=True,
        return_full_grid=True,
    )


# -----------------------------------------------------------------------------
# 9. Petit outil de diagnostic
# -----------------------------------------------------------------------------

def summarize_solution(states: np.ndarray) -> Dict[str, float]:
    """
    Retourne quelques statistiques simples sur une trajectoire de solution.

    Paramètres
    ----------
    states : np.ndarray
        Tableau de forme (n_times, Nx).

    Retour
    ------
    Dict[str, float]
        Quelques statistiques élémentaires.
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2:
        raise ValueError("`states` doit être un tableau 2D de forme (n_times, Nx).")

    return {
        "min_value": float(np.min(states)),
        "max_value": float(np.max(states)),
        "mean_value": float(np.mean(states)),
        "std_value": float(np.std(states)),
    }


# -----------------------------------------------------------------------------
# 10. Bloc de test minimal
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Test autonome minimal sans dépendre du reste du projet.
    Nx = 128
    L = 2.0 * np.pi
    T = 1.0
    dt = 1.0e-3
    nu = 0.05
    save_every = 20

    x, dx = build_spatial_grid(L, Nx)

    # Condition initiale simple : combinaison de deux modes.
    u0 = np.sin(x) + 0.5 * np.sin(2.0 * x)

    result = solve_burgers(
        u0=u0,
        L=L,
        T=T,
        dt=dt,
        nu=nu,
        save_every=save_every,
        time_scheme="rk4",
        check_nan=True,
        return_full_grid=True,
    )

    states = result["states"]
    times = result["times"]

    print("Test du solveur Burgers")
    print(f"- Nombre de snapshots sauvegardés : {states.shape[0]}")
    print(f"- Nombre de points spatiaux       : {states.shape[1]}")
    print(f"- Temps final effectif            : {times[-1]:.6f}")
    print(f"- Min global                      : {states.min():.6f}")
    print(f"- Max global                      : {states.max():.6f}")
    print("- Indicateurs initiaux            :", result["stability_indicators_initial"])
