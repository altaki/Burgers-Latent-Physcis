"""
utils/plotting.py
=================

Fonctions de visualisation pour le projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce module fournit des fonctions simples, lisibles et réutilisables pour produire
les figures principales du projet :

1. snapshots et cartes spatio-temporelles des trajectoires,
2. courbes d'énergie et de dissipation,
3. comparaison entre une observable latente z(t) et l'énergie E(t),
4. nuages de points z vs E,
5. histogrammes de violations de monotonie,
6. courbes d'entraînement à partir des fichiers JSON d'historique.

Philosophie
-----------
- code volontairement simple et pédagogique,
- figures prêtes à être intégrées dans un README ou un rapport,
- labels en français pour cohérence avec l'usage actuel du projet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. Chargement simple des données et rapports JSON
# =============================================================================

def load_npz_dataset(path: str | Path) -> Dict[str, Any]:
    """Charge un dataset `.npz` et renvoie un dictionnaire de tableaux numpy."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}



def load_json(path: str | Path) -> Dict[str, Any]:
    """Charge un fichier JSON et renvoie un dictionnaire Python."""
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)



def ensure_dir(path: str | Path) -> Path:
    """Crée un dossier si nécessaire et renvoie le Path correspondant."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# 2. Utilitaires de conversion / reconstruction
# =============================================================================

def flatten_scalar_array(a: np.ndarray) -> np.ndarray:
    """Aplati une observable scalaire de forme (n,) ou (n,1) vers (n,)."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    raise ValueError(f"Observable scalaire attendue de forme (n,) ou (n,1), obtenu {a.shape}")



def reconstruct_scalar_trajectories(
    scalar_values: np.ndarray,
    trajectory_index: np.ndarray,
    time_index: np.ndarray,
    n_trajectories: Optional[int] = None,
    n_times: Optional[int] = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Reconstruit un tableau 2D (n_traj, n_times) à partir de valeurs plates.
    """
    scalar_values = flatten_scalar_array(np.asarray(scalar_values, dtype=float))
    trajectory_index = np.asarray(trajectory_index, dtype=int)
    time_index = np.asarray(time_index, dtype=int)

    if not (scalar_values.shape[0] == trajectory_index.shape[0] == time_index.shape[0]):
        raise ValueError("Les vecteurs scalar_values, trajectory_index et time_index doivent avoir la même longueur.")

    if n_trajectories is None:
        n_trajectories = int(np.max(trajectory_index)) + 1
    if n_times is None:
        n_times = int(np.max(time_index)) + 1

    out = np.full((n_trajectories, n_times), fill_value, dtype=float)
    out[trajectory_index, time_index] = scalar_values
    return out



def compute_monotonicity_violations(values: np.ndarray, atol: float = 0.0) -> np.ndarray:
    """Retourne les violations positives de monotonie décroissante."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("`values` doit être 1D.")
    diffs = values[1:] - values[:-1] - float(atol)
    return np.maximum(0.0, diffs)


# =============================================================================
# 3. Figures PDE / données de référence
# =============================================================================

def plot_snapshot(
    x: np.ndarray,
    u: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """Trace un snapshot spatial unique u(x)."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, u)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def plot_trajectory_heatmap(
    x: np.ndarray,
    times: np.ndarray,
    states: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """
    Trace une carte spatio-temporelle d'une trajectoire u(t,x).

    Paramètres
    ----------
    states : np.ndarray
        Tableau de forme (n_times, Nx).
    """
    x = np.asarray(x, dtype=float)
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)

    if states.ndim != 2:
        raise ValueError("`states` doit être de forme (n_times, Nx).")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(
        states,
        aspect='auto',
        origin='lower',
        extent=[x[0], x[-1], times[0], times[-1]],
    )
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='u(t,x)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def plot_energy_dissipation(
    times: np.ndarray,
    energy: np.ndarray,
    dissipation: Optional[np.ndarray],
    title: str,
    output_path: str | Path,
) -> None:
    """Trace les courbes énergie / dissipation d'une trajectoire."""
    times = np.asarray(times, dtype=float)
    energy = np.asarray(energy, dtype=float)

    if dissipation is not None:
        dissipation = np.asarray(dissipation, dtype=float)

    if dissipation is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(times, energy, label='Énergie')
        ax.set_xlabel('t')
        ax.set_ylabel('Valeur')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(times, energy, label='Énergie')
        axes[0].set_ylabel('Énergie')
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(times, dissipation, label='Dissipation')
        axes[1].set_xlabel('t')
        axes[1].set_ylabel('Dissipation')
        axes[1].grid(True)
        axes[1].legend()
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =============================================================================
# 4. Figures ML : observable latente vs énergie
# =============================================================================

def plot_scalar_vs_energy_scatter(
    scalar: np.ndarray,
    energy: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """Trace un nuage de points entre z et E."""
    z = flatten_scalar_array(np.asarray(scalar, dtype=float))
    e = flatten_scalar_array(np.asarray(energy, dtype=float))

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(z, e, s=8, alpha=0.5)
    ax.set_xlabel('Observable latente z')
    ax.set_ylabel('Énergie E')
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def plot_scalar_and_energy_trajectory(
    times: np.ndarray,
    scalar: np.ndarray,
    energy: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """
    Trace z(t) et E(t) sur une même trajectoire, avec normalisation affine simple
    pour faciliter la comparaison visuelle.
    """
    times = np.asarray(times, dtype=float)
    z = flatten_scalar_array(np.asarray(scalar, dtype=float))
    e = flatten_scalar_array(np.asarray(energy, dtype=float))

    # Normalisation [0,1] pour comparaison visuelle uniquement.
    def normalize01(v: np.ndarray) -> np.ndarray:
        vmin, vmax = np.min(v), np.max(v)
        if np.isclose(vmax, vmin):
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)

    z01 = normalize01(z)
    e01 = normalize01(e)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, z01, label='z(t) normalisé')
    ax.plot(times, e01, label='E(t) normalisée')
    ax.set_xlabel('t')
    ax.set_ylabel('Valeur normalisée')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def plot_monotonicity_histogram(
    scalar_trajectories: np.ndarray,
    title: str,
    output_path: str | Path,
    atol: float = 0.0,
) -> None:
    """
    Trace l'histogramme des violations positives de monotonie sur toutes les trajectoires.
    """
    scalar_trajectories = np.asarray(scalar_trajectories, dtype=float)
    if scalar_trajectories.ndim != 2:
        raise ValueError("`scalar_trajectories` doit être de forme (n_traj, n_times).")

    violations = []
    for i in range(scalar_trajectories.shape[0]):
        violations.append(compute_monotonicity_violations(scalar_trajectories[i], atol=atol))
    violations = np.concatenate(violations, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(violations, bins=40)
    ax.set_xlabel('Violation positive de monotonie')
    ax.set_ylabel('Fréquence')
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =============================================================================
# 5. Courbes d'entraînement à partir des fichiers JSON
# =============================================================================

def plot_training_history(
    history_json_path: str | Path,
    title: str,
    output_path: str | Path,
) -> None:
    """
    Trace automatiquement les séries présentes dans un historique JSON.

    Le JSON doit contenir un dictionnaire {nom_metric: liste_de_valeurs}.
    """
    history = load_json(history_json_path)

    # On conserve uniquement les entrées de type liste numérique.
    series = {
        k: v for k, v in history.items()
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float))
    }
    if len(series) == 0:
        raise ValueError(f"Aucune série numérique trouvée dans {history_json_path}.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in series.items():
        ax.plot(np.arange(1, len(values) + 1), values, label=name)

    ax.set_xlabel('Époque')
    ax.set_ylabel('Valeur')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =============================================================================
# 6. Fonctions haut niveau prêtes pour le README
# =============================================================================

def generate_reference_dataset_figures(
    dataset_path: str | Path,
    output_dir: str | Path,
    trajectory_index: int = 0,
) -> Dict[str, str]:
    """
    Génère les figures de référence PDE à partir d'un dataset `.npz`.

    Retour
    ------
    Dict[str, str]
        Chemins des figures produites.
    """
    dataset = load_npz_dataset(dataset_path)
    output_dir = ensure_dir(output_dir)

    x = np.asarray(dataset['x'], dtype=float)
    times = np.asarray(dataset['times'], dtype=float)
    states = np.asarray(dataset['states'], dtype=float)
    energy = np.asarray(dataset['energy'], dtype=float)
    dissipation = np.asarray(dataset['dissipation'], dtype=float)

    n_traj = states.shape[0]
    idx = int(np.clip(trajectory_index, 0, n_traj - 1))

    out = {}
    path1 = Path(output_dir) / f'trajectory_heatmap_traj{idx}.png'
    plot_trajectory_heatmap(x, times, states[idx], 'Carte spatio-temporelle d\'une trajectoire', path1)
    out['trajectory_heatmap'] = str(path1)

    path2 = Path(output_dir) / f'energy_dissipation_traj{idx}.png'
    plot_energy_dissipation(times, energy[idx], dissipation[idx], 'Énergie et dissipation', path2)
    out['energy_dissipation'] = str(path2)

    path3 = Path(output_dir) / f'initial_snapshot_traj{idx}.png'
    plot_snapshot(x, states[idx, 0], 'Condition initiale', path3)
    out['initial_snapshot'] = str(path3)

    path4 = Path(output_dir) / f'final_snapshot_traj{idx}.png'
    plot_snapshot(x, states[idx, -1], 'Snapshot final', path4)
    out['final_snapshot'] = str(path4)

    return out



def generate_latent_evaluation_figures(
    extracted: Dict[str, Any],
    output_dir: str | Path,
    prefix: str = 'latent',
    trajectory_index: Optional[int] = None,
) -> Dict[str, str]:
    """
    Génère les figures principales pour un modèle latent à partir du dictionnaire
    `extracted` renvoyé par `ml/evaluate.py`.

    Paramètres
    ----------
    extracted : Dict[str, Any]
        Dictionnaire contenant au minimum `energy`, `trajectory_index`, `time_index`.
        Si `scalar` est présent, on génère les figures z vs E.
    """
    output_dir = ensure_dir(output_dir)
    out: Dict[str, str] = {}

    if 'scalar' not in extracted:
        return out

    scalar = extracted['scalar']
    energy = extracted['energy']
    traj_idx = np.asarray(extracted['trajectory_index'], dtype=int)
    time_idx = np.asarray(extracted['time_index'], dtype=int)

    # Figure globale scatter z vs E
    path1 = Path(output_dir) / f'{prefix}_scatter_z_vs_energy.png'
    plot_scalar_vs_energy_scatter(
        scalar, energy,
        'Nuage de points : observable latente vs énergie',
        path1,
    )
    out['scatter_z_vs_energy'] = str(path1)

    # Reconstruction trajectoire par trajectoire
    scalar_traj = reconstruct_scalar_trajectories(scalar, traj_idx, time_idx)
    energy_traj = reconstruct_scalar_trajectories(energy, traj_idx, time_idx)
    n_traj = scalar_traj.shape[0]

    idx = 0 if trajectory_index is None else int(np.clip(trajectory_index, 0, n_traj - 1))
    times = np.arange(scalar_traj.shape[1], dtype=float)

    path2 = Path(output_dir) / f'{prefix}_trajectory_z_vs_energy_traj{idx}.png'
    plot_scalar_and_energy_trajectory(
        times=times,
        scalar=scalar_traj[idx],
        energy=energy_traj[idx],
        title='Comparaison z(t) / E(t) sur une trajectoire',
        output_path=path2,
    )
    out['trajectory_z_vs_energy'] = str(path2)

    path3 = Path(output_dir) / f'{prefix}_monotonicity_histogram.png'
    plot_monotonicity_histogram(
        scalar_trajectories=scalar_traj,
        title='Histogramme des violations de monotonie de z(t)',
        output_path=path3,
    )
    out['monotonicity_histogram'] = str(path3)

    return out


# =============================================================================
# 7. Bloc de test minimal
# =============================================================================

if __name__ == '__main__':
    # Test minimal purement synthétique.
    x = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    times = np.linspace(0.0, 1.0, 50)

    states = np.zeros((50, 64))
    for i, t in enumerate(times):
        states[i] = np.exp(-t) * np.sin(x)

    energy = np.exp(-times)
    dissipation = 0.5 * np.exp(-times)

    ensure_dir('figures_test')
    plot_snapshot(x, states[0], 'Test snapshot', 'figures_test/test_snapshot.png')
    plot_trajectory_heatmap(x, times, states, 'Test heatmap', 'figures_test/test_heatmap.png')
    plot_energy_dissipation(times, energy, dissipation, 'Test énergie/dissipation', 'figures_test/test_energy.png')
    plot_scalar_vs_energy_scatter(energy, energy + 0.1, 'Test scatter', 'figures_test/test_scatter.png')
    plot_scalar_and_energy_trajectory(times, energy, energy + 0.1, 'Test z(t) vs E(t)', 'figures_test/test_z_energy.png')

    scalar_traj = np.vstack([np.exp(-times), np.exp(-times) + 0.05 * np.sin(5 * times)])
    plot_monotonicity_histogram(scalar_traj, 'Test monotonie', 'figures_test/test_mono.png')
    print('Tests rapides de utils/plotting.py terminés.')
