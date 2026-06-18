"""
pde/initial_conditions.py
=========================

Fonctions utilitaires pour générer des conditions initiales (CI) pour
l'équation de Burgers visqueuse 1D sur un domaine périodique.

Objectif
--------
Ce module fournit plusieurs familles de conditions initiales afin de :

1. produire un dataset varié pour l'apprentissage,
2. tester la robustesse des modèles à différentes structures spatiales,
3. conserver un code simple, lisible et facilement extensible.

Familles implémentées dans cette première version
-------------------------------------------------
- combinaisons aléatoires de modes de Fourier,
- sommes de bosses gaussiennes périodisées,
- champs aléatoires lisses construits spectralement.

Convention
----------
- Le domaine est 1D périodique, typiquement [0, L].
- Le maillage spatial est donné explicitement par le vecteur `x`.
- Les fonctions renvoient des tableaux 1D numpy de même taille que `x`.

Philosophie de code
-------------------
- une fonction = une responsabilité claire,
- commentaires explicatifs,
- paramètres explicites,
- comportement simple et reproductible si un générateur aléatoire est fourni.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# 1. Utilitaires généraux
# -----------------------------------------------------------------------------

def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """
    Retourne un générateur aléatoire numpy.

    Paramètres
    ----------
    rng : np.random.Generator ou None
        Si un générateur est fourni, il est renvoyé tel quel.
        Sinon, un nouveau générateur est créé via `np.random.default_rng()`.

    Retour
    ------
    np.random.Generator
        Générateur utilisable pour produire des tirages reproductibles.
    """
    return rng if rng is not None else np.random.default_rng()



def _validate_grid(x: np.ndarray) -> np.ndarray:
    """
    Vérifie que le maillage spatial `x` est un tableau 1D non vide.

    Paramètres
    ----------
    x : np.ndarray
        Grille spatiale 1D.

    Retour
    ------
    np.ndarray
        Tableau converti en float.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Le maillage `x` doit être un tableau 1D.")
    if x.size < 2:
        raise ValueError("Le maillage `x` doit contenir au moins deux points.")

    return x



def _domain_length_from_grid(x: np.ndarray) -> float:
    """
    Calcule une estimation de la longueur du domaine à partir d'une grille
    régulièrement espacée, en supposant une convention de type endpoint=False.

    Si x = [0, dx, 2dx, ..., (N-1)dx], alors la longueur du domaine est N*dx.

    Paramètres
    ----------
    x : np.ndarray
        Grille spatiale 1D régulière.

    Retour
    ------
    float
        Longueur estimée du domaine.
    """
    x = _validate_grid(x)
    dx = x[1] - x[0]
    return float(x.size * dx)



def normalize_l2(u: np.ndarray, x: np.ndarray, target_norm: float = 1.0) -> np.ndarray:
    """
    Renormalise une condition initiale pour lui imposer une norme L2 discrète cible.

    Définition discrète utilisée :
        ||u||_L2 ≈ sqrt(Σ_j u_j^2 dx)

    Paramètres
    ----------
    u : np.ndarray
        Champ spatial de forme (Nx,).
    x : np.ndarray
        Grille spatiale 1D.
    target_norm : float
        Norme L2 cible.

    Retour
    ------
    np.ndarray
        Champ renormalisé.
    """
    x = _validate_grid(x)
    u = np.asarray(u, dtype=float)

    if u.shape != x.shape:
        raise ValueError("`u` et `x` doivent avoir la même forme.")

    dx = x[1] - x[0]
    current_norm = np.sqrt(np.sum(u ** 2) * dx)

    if current_norm < 1.0e-14:
        # Si le champ est (quasi) nul, on le renvoie tel quel pour éviter une division instable.
        return u.copy()

    return (target_norm / current_norm) * u


# -----------------------------------------------------------------------------
# 2. Famille A : combinaisons aléatoires de modes de Fourier
# -----------------------------------------------------------------------------

def random_fourier_ic(
    x: np.ndarray,
    max_mode: int = 5,
    amplitude_scale: float = 1.0,
    zero_mean: bool = True,
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Génère une condition initiale comme combinaison aléatoire de modes de Fourier.

    Forme utilisée :
        u0(x) = Σ_{k=1}^{K} [a_k sin(2πk x / L) + b_k cos(2πk x / L)]

    où K = max_mode et les coefficients a_k, b_k sont tirés aléatoirement.

    Paramètres
    ----------
    x : np.ndarray
        Grille spatiale 1D.
    max_mode : int
        Nombre maximal de modes de Fourier utilisés.
    amplitude_scale : float
        Échelle des coefficients aléatoires.
    zero_mean : bool
        Si True, on n'ajoute pas de mode constant.
    normalize : bool
        Si True, renormalise le champ à une norme L2 cible.
    target_norm : float
        Norme L2 cible si `normalize=True`.
    rng : np.random.Generator ou None
        Générateur aléatoire optionnel.

    Retour
    ------
    np.ndarray
        Condition initiale de forme (Nx,).
    """
    x = _validate_grid(x)
    rng = _get_rng(rng)

    if max_mode < 1:
        raise ValueError("`max_mode` doit être >= 1.")

    L = _domain_length_from_grid(x)
    u0 = np.zeros_like(x, dtype=float)

    # Coefficients aléatoires des modes sin/cos.
    for k in range(1, max_mode + 1):
        a_k = amplitude_scale * rng.uniform(-1.0, 1.0)
        b_k = amplitude_scale * rng.uniform(-1.0, 1.0)
        phase = 2.0 * np.pi * k * x / L
        u0 += a_k * np.sin(phase) + b_k * np.cos(phase)

    if not zero_mean:
        c0 = amplitude_scale * rng.uniform(-1.0, 1.0)
        u0 += c0

    if normalize:
        u0 = normalize_l2(u0, x, target_norm=target_norm)

    return u0


# -----------------------------------------------------------------------------
# 3. Famille B : bosses gaussiennes périodisées
# -----------------------------------------------------------------------------

def _periodic_distance(x: np.ndarray, center: float, L: float) -> np.ndarray:
    """
    Calcule la distance périodique minimale entre chaque point de `x` et un centre.

    Paramètres
    ----------
    x : np.ndarray
        Grille spatiale 1D.
    center : float
        Position du centre sur le domaine périodique.
    L : float
        Longueur du domaine.

    Retour
    ------
    np.ndarray
        Distance périodique minimale à `center`.
    """
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
    """
    Génère une condition initiale comme somme de bosses gaussiennes périodisées.

    Chaque bosse est de la forme :
        A_m * exp( - d(x, x_m)^2 / sigma_m^2 )
    où d est la distance périodique minimale sur le domaine.

    Paramètres
    ----------
    x : np.ndarray
        Grille spatiale 1D.
    n_bumps : int
        Nombre de bosses gaussiennes.
    amplitude_range : Sequence[float]
        Intervalle [amin, amax] pour les amplitudes.
    width_range : Sequence[float]
        Intervalle [wmin, wmax] pour les largeurs sigma.
        Les largeurs sont interprétées relativement à la longueur du domaine.
    normalize : bool
        Si True, renormalise le champ à une norme L2 cible.
    target_norm : float
        Norme L2 cible si `normalize=True`.
    rng : np.random.Generator ou None
        Générateur aléatoire optionnel.

    Retour
    ------
    np.ndarray
        Condition initiale de forme (Nx,).
    """
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


# -----------------------------------------------------------------------------
# 4. Famille C : champ aléatoire lisse construit spectralement
# -----------------------------------------------------------------------------

def random_smooth_ic(
    x: np.ndarray,
    spectral_decay: float = 2.0,
    amplitude_scale: float = 1.0,
    zero_mean: bool = True,
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Génère un champ aléatoire lisse à partir d'un spectre de Fourier amorti.

    Idée
    ----
    On tire des coefficients complexes aléatoires en fréquence, puis on applique
    un amortissement spectral du type (1 + |k|)^(-spectral_decay), ce qui rend
    le champ d'autant plus lisse que `spectral_decay` est grand.

    Paramètres
    ----------
    x : np.ndarray
        Grille spatiale 1D.
    spectral_decay : float
        Exposant d'amortissement spectral.
    amplitude_scale : float
        Facteur global d'échelle des amplitudes.
    zero_mean : bool
        Si True, la composante moyenne est annulée.
    normalize : bool
        Si True, renormalise le champ à une norme L2 cible.
    target_norm : float
        Norme L2 cible si `normalize=True`.
    rng : np.random.Generator ou None
        Générateur aléatoire optionnel.

    Retour
    ------
    np.ndarray
        Condition initiale de forme (Nx,).
    """
    x = _validate_grid(x)
    rng = _get_rng(rng)

    Nx = x.size

    # Fréquences entières compatibles avec la FFT discrète.
    modes = np.fft.fftfreq(Nx, d=1.0 / Nx)

    # Coefficients complexes gaussiens centrés.
    real_part = rng.normal(loc=0.0, scale=1.0, size=Nx)
    imag_part = rng.normal(loc=0.0, scale=1.0, size=Nx)
    coeffs = real_part + 1j * imag_part

    # Amortissement spectral : plus l'exposant est grand, plus le champ est lisse.
    decay = (1.0 + np.abs(modes)) ** (-spectral_decay)
    coeffs *= amplitude_scale * decay

    # Gestion de la moyenne.
    if zero_mean:
        coeffs[0] = 0.0

    # Pour obtenir un champ réel via ifft, on impose une symétrie hermitienne.
    coeffs = _enforce_hermitian_symmetry(coeffs)

    u0 = np.fft.ifft(coeffs).real

    if normalize:
        u0 = normalize_l2(u0, x, target_norm=target_norm)

    return u0



def _enforce_hermitian_symmetry(coeffs: np.ndarray) -> np.ndarray:
    """
    Imposera une symétrie hermitienne à un spectre de Fourier discret afin que
    la transformée inverse soit réelle.

    Paramètres
    ----------
    coeffs : np.ndarray
        Tableau complexe 1D de coefficients spectraux.

    Retour
    ------
    np.ndarray
        Tableau complexe symétrisé.
    """
    coeffs = np.asarray(coeffs, dtype=complex).copy()
    N = coeffs.size

    # k = 0 doit être réel.
    coeffs[0] = coeffs[0].real + 0j

    # Si N est pair, le mode de Nyquist doit également être réel.
    if N % 2 == 0:
        coeffs[N // 2] = coeffs[N // 2].real + 0j

    # Impose coeffs[-k] = conjugate(coeffs[k]).
    for k in range(1, N // 2 + (0 if N % 2 == 0 else 1)):
        coeffs[-k] = np.conjugate(coeffs[k])

    return coeffs


# -----------------------------------------------------------------------------
# 5. Interface unique de tirage d'une condition initiale
# -----------------------------------------------------------------------------

def sample_initial_condition(
    x: np.ndarray,
    ic_type: str = "fourier",
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> np.ndarray:
    """
    Interface unique pour générer une condition initiale parmi plusieurs familles.

    Paramètres
    ----------
    x : np.ndarray
        Grille spatiale 1D.
    ic_type : str
        Type de condition initiale. Possibilités actuellement supportées :
        - "fourier"
        - "gaussian_bumps"
        - "random_smooth"
    normalize : bool
        Si True, renormalise la condition initiale générée.
    target_norm : float
        Norme L2 cible si `normalize=True`.
    rng : np.random.Generator ou None
        Générateur aléatoire optionnel.
    **kwargs : dict
        Paramètres supplémentaires transmis à la fonction spécialisée.

    Retour
    ------
    np.ndarray
        Condition initiale de forme (Nx,).
    """
    x = _validate_grid(x)
    rng = _get_rng(rng)

    ic_type = ic_type.lower()

    if ic_type == "fourier":
        return random_fourier_ic(
            x,
            normalize=normalize,
            target_norm=target_norm,
            rng=rng,
            **kwargs,
        )

    if ic_type == "gaussian_bumps":
        return gaussian_bumps_ic(
            x,
            normalize=normalize,
            target_norm=target_norm,
            rng=rng,
            **kwargs,
        )

    if ic_type == "random_smooth":
        return random_smooth_ic(
            x,
            normalize=normalize,
            target_norm=target_norm,
            rng=rng,
            **kwargs,
        )

    raise ValueError(
        f"Type de condition initiale inconnu : {ic_type!r}. "
        "Choisir parmi {'fourier', 'gaussian_bumps', 'random_smooth'}."
    )


# -----------------------------------------------------------------------------
# 6. Petit utilitaire : tirage aléatoire d'un type parmi une liste
# -----------------------------------------------------------------------------

def sample_initial_condition_from_list(
    x: np.ndarray,
    ic_types: Sequence[str],
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> tuple[np.ndarray, str]:
    """
    Tire aléatoirement un type de condition initiale parmi une liste,
    puis génère le champ correspondant.

    Paramètres
    ----------
    x : np.ndarray
        Grille spatiale 1D.
    ic_types : Sequence[str]
        Liste / séquence des types autorisés.
    normalize : bool
        Si True, renormalise la condition initiale générée.
    target_norm : float
        Norme L2 cible si `normalize=True`.
    rng : np.random.Generator ou None
        Générateur aléatoire optionnel.
    **kwargs : dict
        Paramètres supplémentaires transmis à la fonction de génération.

    Retour
    ------
    tuple[np.ndarray, str]
        Le champ généré, puis le type de CI effectivement tiré.
    """
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


# -----------------------------------------------------------------------------
# 7. Bloc de test minimal
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    Nx = 128
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, Nx, endpoint=False)
    rng = np.random.default_rng(42)

    print("Test de génération des conditions initiales")

    u_fourier = random_fourier_ic(x, max_mode=5, normalize=True, rng=rng)
    print(f"- Fourier        : min={u_fourier.min(): .4f}, max={u_fourier.max(): .4f}")

    u_gauss = gaussian_bumps_ic(x, n_bumps=3, normalize=True, rng=rng)
    print(f"- Gaussian bumps : min={u_gauss.min(): .4f}, max={u_gauss.max(): .4f}")

    u_smooth = random_smooth_ic(x, spectral_decay=2.5, normalize=True, rng=rng)
    print(f"- Random smooth  : min={u_smooth.min(): .4f}, max={u_smooth.max(): .4f}")

    u_any, ic_name = sample_initial_condition_from_list(
        x,
        ic_types=["fourier", "gaussian_bumps", "random_smooth"],
        normalize=True,
        rng=rng,
    )
    print(f"- Tirage mixte   : type={ic_name}, min={u_any.min(): .4f}, max={u_any.max(): .4f}")
