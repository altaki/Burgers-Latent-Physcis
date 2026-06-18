"""
pde/observables.py
==================

Fonctions utilitaires pour calculer les observables physiques associées
à l'équation de Burgers visqueuse 1D.

Objectif
--------
Ce module fournit des fonctions simples, lisibles et abondamment commentées
pour calculer des quantités de référence sur les snapshots d'une solution
numérique de l'équation de Burgers. Ces quantités serviront à :

1. vérifier qualitativement la cohérence physique des simulations,
2. construire les labels de référence pour certaines expériences supervisées,
3. comparer les variables latentes apprises par les modèles de machine learning
   à des observables physiques connues.

Observables implémentés dans cette première version
---------------------------------------------------
- énergie L2 : E(u) = 1/2 * ∫ u(x)^2 dx
- gradient spatial : u_x
- dissipation associée : D(u) = ∫ |u_x(x)|^2 dx
- métriques simples de monotonie pour une suite temporelle

Convention numérique
--------------------
- Domaine 1D périodique.
- Les intégrales sont approchées par une somme de Riemann uniforme.
- Les dérivées spatiales sont calculées par différences finies centrées
  avec extension périodique via numpy.roll.

Exemple d'utilisation
---------------------
>>> import numpy as np
>>> from pde.observables import compute_energy, compute_energy_trajectory
>>> x = np.linspace(0.0, 2*np.pi, 128, endpoint=False)
>>> dx = x[1] - x[0]
>>> u = np.sin(x)
>>> E = compute_energy(u, dx)
>>> print(E)

Style de code
-------------
Le code est volontairement simple et pédagogique afin de servir de base
solide à la suite du projet.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np


# -----------------------------------------------------------------------------
# 1. Fonctions de base : validations et helpers
# -----------------------------------------------------------------------------

def _validate_1d_state(u: np.ndarray) -> np.ndarray:
    """
    Vérifie qu'un état spatial u est bien un tableau 1D.

    Paramètres
    ----------
    u : np.ndarray
        Tableau représentant un snapshot spatial de la solution.

    Retour
    ------
    np.ndarray
        Le tableau converti en float si nécessaire.

    Erreurs
    -------
    ValueError
        Si `u` n'est pas un tableau unidimensionnel.
    """
    u = np.asarray(u, dtype=float)

    if u.ndim != 1:
        raise ValueError(
            "L'état spatial `u` doit être un tableau 1D de forme (Nx,)."
        )

    return u


def _validate_2d_trajectory(states: np.ndarray) -> np.ndarray:
    """
    Vérifie qu'une trajectoire de snapshots est bien un tableau 2D.

    Paramètres
    ----------
    states : np.ndarray
        Tableau de forme (n_times, Nx).

    Retour
    ------
    np.ndarray
        Tableau converti en float si nécessaire.
    """
    states = np.asarray(states, dtype=float)

    if states.ndim != 2:
        raise ValueError(
            "`states` doit être un tableau 2D de forme (n_times, Nx)."
        )

    return states


# -----------------------------------------------------------------------------
# 2. Énergie L2
# -----------------------------------------------------------------------------

def compute_energy(u: np.ndarray, dx: float) -> float:
    """
    Calcule l'énergie L2 discrète d'un snapshot u.

    Définition continue :
        E(u) = 1/2 * ∫ u(x)^2 dx

    Approximation discrète :
        E(u) ≈ 1/2 * Σ_j u_j^2 * dx

    Paramètres
    ----------
    u : np.ndarray
        Snapshot spatial de forme (Nx,).
    dx : float
        Pas d'espace uniforme.

    Retour
    ------
    float
        Énergie discrète du snapshot.
    """
    u = _validate_1d_state(u)
    return 0.5 * float(np.sum(u ** 2) * dx)



def compute_energy_trajectory(states: np.ndarray, dx: float) -> np.ndarray:
    """
    Calcule l'énergie pour tous les snapshots d'une trajectoire.

    Paramètres
    ----------
    states : np.ndarray
        Tableau de forme (n_times, Nx).
    dx : float
        Pas d'espace uniforme.

    Retour
    ------
    np.ndarray
        Tableau 1D de longueur n_times contenant l'énergie à chaque instant.
    """
    states = _validate_2d_trajectory(states)
    return 0.5 * np.sum(states ** 2, axis=1) * dx


# -----------------------------------------------------------------------------
# 3. Gradient spatial et dissipation
# -----------------------------------------------------------------------------

def compute_spatial_gradient(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Calcule le gradient spatial u_x par différences finies centrées
    avec conditions périodiques.

    Formule utilisée :
        u_x(x_j) ≈ [u_{j+1} - u_{j-1}] / (2 dx)

    Le décalage périodique est réalisé avec numpy.roll.

    Paramètres
    ----------
    u : np.ndarray
        Snapshot spatial de forme (Nx,).
    dx : float
        Pas d'espace uniforme.

    Retour
    ------
    np.ndarray
        Tableau de forme (Nx,) contenant une approximation de u_x.
    """
    u = _validate_1d_state(u)
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)



def compute_second_spatial_derivative(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Calcule la dérivée seconde spatiale u_xx par différences finies centrées
    avec conditions périodiques.

    Formule utilisée :
        u_xx(x_j) ≈ [u_{j+1} - 2u_j + u_{j-1}] / dx^2

    Cette fonction est fournie ici car elle peut être utile pour des analyses
    complémentaires, même si elle n'est pas directement requise pour le calcul
    de l'énergie.

    Paramètres
    ----------
    u : np.ndarray
        Snapshot spatial de forme (Nx,).
    dx : float
        Pas d'espace uniforme.

    Retour
    ------
    np.ndarray
        Tableau de forme (Nx,) contenant une approximation de u_xx.
    """
    u = _validate_1d_state(u)
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx ** 2)



def compute_dissipation(u: np.ndarray, dx: float) -> float:
    """
    Calcule la dissipation associée au gradient spatial :

        D(u) = ∫ |u_x(x)|^2 dx

    Approximation discrète :
        D(u) ≈ Σ_j |(u_x)_j|^2 * dx

    Remarque
    --------
    Dans l'équation de Burgers visqueuse, l'évolution de l'énergie vérifie
    formellement :

        dE/dt = - nu * D(u)

    sous conditions périodiques ou conditions adaptées.

    Paramètres
    ----------
    u : np.ndarray
        Snapshot spatial de forme (Nx,).
    dx : float
        Pas d'espace uniforme.

    Retour
    ------
    float
        Valeur discrète de la dissipation.
    """
    ux = compute_spatial_gradient(u, dx)
    return float(np.sum(ux ** 2) * dx)



def compute_dissipation_trajectory(states: np.ndarray, dx: float) -> np.ndarray:
    """
    Calcule la dissipation pour tous les snapshots d'une trajectoire.

    Paramètres
    ----------
    states : np.ndarray
        Tableau de forme (n_times, Nx).
    dx : float
        Pas d'espace uniforme.

    Retour
    ------
    np.ndarray
        Tableau 1D de longueur n_times contenant la dissipation à chaque instant.
    """
    states = _validate_2d_trajectory(states)
    dissipations = np.empty(states.shape[0], dtype=float)

    for n in range(states.shape[0]):
        dissipations[n] = compute_dissipation(states[n], dx)

    return dissipations


# -----------------------------------------------------------------------------
# 4. Suites temporelles : monotonie et violations
# -----------------------------------------------------------------------------

def compute_monotonicity_violations(values: np.ndarray) -> np.ndarray:
    """
    Calcule les violations positives de monotonie décroissante dans une suite.

    Si la suite idéale est décroissante, on considère que la quantité :
        values[n+1] - values[n]
    ne doit pas être positive.

    On définit donc les violations par :
        violation[n] = max(0, values[n+1] - values[n])

    Paramètres
    ----------
    values : np.ndarray
        Suite 1D de valeurs au cours du temps.

    Retour
    ------
    np.ndarray
        Tableau 1D de taille len(values)-1 contenant les violations positives.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("`values` doit être un tableau 1D.")

    diffs = values[1:] - values[:-1]
    return np.maximum(0.0, diffs)



def compute_monotonicity_metrics(values: np.ndarray, atol: float = 0.0) -> Dict[str, Any]:
    """
    Calcule plusieurs métriques simples caractérisant la monotonie décroissante
    d'une suite temporelle.

    Paramètres
    ----------
    values : np.ndarray
        Suite 1D de valeurs.
    atol : float, optionnel
        Tolérance absolue admise pour considérer qu'une augmentation est nulle.
        Ceci peut être utile pour absorber de très petites erreurs numériques.

    Retour
    ------
    Dict[str, Any]
        Dictionnaire contenant :
        - n_steps : nombre total de transitions temporelles
        - monotone_fraction : proportion de transitions non croissantes
        - mean_positive_violation : moyenne des violations positives
        - max_positive_violation : maximum des violations positives
        - is_nonincreasing : booléen indiquant si toute la suite est non croissante
          à la tolérance `atol` près
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("`values` doit être un tableau 1D.")

    if values.size < 2:
        raise ValueError("La suite doit contenir au moins deux valeurs.")

    diffs = values[1:] - values[:-1]
    positive_violations = np.maximum(0.0, diffs - atol)

    monotone_fraction = float(np.mean(diffs <= atol))
    mean_positive_violation = float(np.mean(positive_violations))
    max_positive_violation = float(np.max(positive_violations))
    is_nonincreasing = bool(np.all(diffs <= atol))

    return {
        "n_steps": int(values.size - 1),
        "monotone_fraction": monotone_fraction,
        "mean_positive_violation": mean_positive_violation,
        "max_positive_violation": max_positive_violation,
        "is_nonincreasing": is_nonincreasing,
    }


# -----------------------------------------------------------------------------
# 5. Fonction de synthèse pour une trajectoire complète
# -----------------------------------------------------------------------------

def compute_reference_observables(states: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    """
    Calcule les observables physiques de référence pour une trajectoire entière.

    Paramètres
    ----------
    states : np.ndarray
        Tableau de forme (n_times, Nx) contenant les snapshots d'une trajectoire.
    dx : float
        Pas d'espace uniforme.

    Retour
    ------
    Dict[str, np.ndarray]
        Dictionnaire contenant au minimum :
        - "energy" : énergie au cours du temps
        - "dissipation" : dissipation au cours du temps

    Remarque
    --------
    Cette fonction est pratique pour centraliser les calculs lors de la création
    du dataset.
    """
    states = _validate_2d_trajectory(states)

    energy = compute_energy_trajectory(states, dx)
    dissipation = compute_dissipation_trajectory(states, dx)

    return {
        "energy": energy,
        "dissipation": dissipation,
    }


# -----------------------------------------------------------------------------
# 6. Bloc de test minimal
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Petit test simple sur une fonction sinusoïdale.
    Nx = 128
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, Nx, endpoint=False)
    dx = x[1] - x[0]

    u = np.sin(x)

    E = compute_energy(u, dx)
    D = compute_dissipation(u, dx)
    ux = compute_spatial_gradient(u, dx)
    uxx = compute_second_spatial_derivative(u, dx)

    print("Test simple sur u(x) = sin(x)")
    print(f"Énergie E(u)          = {E:.6f}")
    print(f"Dissipation D(u)      = {D:.6f}")
    print(f"Norme max de u_x      = {np.max(np.abs(ux)):.6f}")
    print(f"Norme max de u_xx     = {np.max(np.abs(uxx)):.6f}")

    # Test de monotonie sur une suite décroissante avec légère violation.
    vals = np.array([1.0, 0.8, 0.7, 0.72, 0.6])
    metrics = compute_monotonicity_metrics(vals)
    print("\nMétriques de monotonie :")
    for key, value in metrics.items():
        print(f"- {key}: {value}")
