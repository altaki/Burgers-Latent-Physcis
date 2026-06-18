"""
ml/evaluate.py
==============

Outils d'évaluation pour le projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce module fournit des fonctions simples, lisibles et directement utilisables pour :

1. charger un dataset et un modèle entraîné,
2. extraire les latents et/ou observables scalaires prédits,
3. comparer une variable apprise à une grandeur physique de référence
   (par exemple l'énergie),
4. mesurer la monotonie d'une observable scalaire au cours du temps,
5. produire des métriques quantitatives prêtes à être analysées dans un
   protocole scientifique.

Cas d'usage typiques
--------------------
- après un entraînement supervisé : comparer `energy_hat` à l'énergie vraie,
- après un entraînement latent : comparer `z_t` à l'énergie E(t),
- évaluer si une observable apprise est monotone,
- calibrer une variable latente par transformation affine simple,
- récupérer les sorties du modèle pour produire ensuite des figures.

Philosophie
-----------
- code simple, explicite et bien commenté,
- séparations claires entre chargement, extraction, métriques et évaluation,
- robustesse aux imports dans un prototype encore en construction,
- pas de dépendances inutiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.util
import json

import numpy as np
import torch
from torch.utils.data import DataLoader


# =============================================================================
# 1. Chargement flexible des modules du projet
# =============================================================================

def _load_module_from_candidates(module_name: str, candidates: List[str]):
    """
    Charge dynamiquement un module depuis une liste de chemins candidats.
    """
    try:
        return __import__(module_name, fromlist=['*'])
    except ModuleNotFoundError:
        pass

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(
        f"Impossible de charger le module {module_name!r}. Candidats testés : {candidates}"
    )



def load_project_symbols():
    """
    Charge les symboles utiles pour l'évaluation.

    Retour
    ------
    tuple
        Contient notamment :
        - get_default_config
        - datasets module symbols
        - models module symbols
        - losses / helpers si besoin
    """
    try:
        from config import get_default_config  # type: ignore
    except ModuleNotFoundError:
        config_module = _load_module_from_candidates('config', ['config.py'])
        get_default_config = config_module.get_default_config

    datasets_module = _load_module_from_candidates(
        'ml_datasets', ['ml_datasets.py', 'ml/datasets.py']
    )
    models_module = _load_module_from_candidates(
        'ml_models', ['ml_models.py', 'ml/models.py']
    )

    return (
        get_default_config,
        datasets_module.load_dataset_npz,
        datasets_module.describe_dataset,
        datasets_module.BurgersSnapshotDataset,
        datasets_module.BurgersPairDataset,
        datasets_module.BurgersTrajectoryDataset,
        models_module.EnergyRegressor,
        models_module.LatentAutoencoder,
        models_module.LatentDynamicsModel,
    )


# =============================================================================
# 2. Utilitaires généraux
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    """Crée un dossier si nécessaire et renvoie le Path correspondant."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Sauvegarde un dictionnaire dans un fichier JSON lisible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def _to_numpy(x: Any) -> np.ndarray:
    """Convertit une entrée torch ou numpy en tableau numpy CPU."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)



def _flatten_scalar_array(a: np.ndarray) -> np.ndarray:
    """
    Aplati une observable scalaire batchée vers une forme (n,).
    Accepte (n,) ou (n,1).
    """
    a = np.asarray(a)
    if a.ndim == 1:
        return a
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    raise ValueError(f"Observable scalaire attendue de forme (n,) ou (n,1), obtenu {a.shape}")


# =============================================================================
# 3. Métriques statistiques simples
# =============================================================================

def compute_regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Calcule quelques métriques standard de régression.

    Retour
    ------
    Dict[str, float]
        - mse
        - rmse
        - mae
        - max_abs_error
    """
    pred = _flatten_scalar_array(np.asarray(pred, dtype=float))
    target = _flatten_scalar_array(np.asarray(target, dtype=float))

    if pred.shape != target.shape:
        raise ValueError(f"Formes incompatibles : {pred.shape} vs {target.shape}")

    error = pred - target
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    max_abs_error = float(np.max(np.abs(error)))

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_abs_error': max_abs_error,
    }



def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcule la corrélation linéaire de Pearson entre deux vecteurs 1D.
    Si l'un des deux vecteurs est constant, renvoie 0.0.
    """
    x = _flatten_scalar_array(np.asarray(x, dtype=float))
    y = _flatten_scalar_array(np.asarray(y, dtype=float))
    if x.shape != y.shape:
        raise ValueError(f"Formes incompatibles : {x.shape} vs {y.shape}")

    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < 1.0e-14 or y_std < 1.0e-14:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])



def rankdata_average(a: np.ndarray) -> np.ndarray:
    """
    Calcule un ranking moyen en gérant les ex aequo (version numpy simple).

    Retourne des rangs commençant à 1.
    """
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind='mergesort')
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))

    a_sorted = a[sorter]
    ranks_sorted = np.zeros(len(a), dtype=float)

    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a_sorted[j] == a_sorted[i]:
            j += 1
        # rang moyen sur [i, j)
        avg_rank = 0.5 * ((i + 1) + j)
        ranks_sorted[i:j] = avg_rank
        i = j

    return ranks_sorted[inv]



def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcule la corrélation de Spearman entre deux vecteurs 1D.
    """
    x = _flatten_scalar_array(np.asarray(x, dtype=float))
    y = _flatten_scalar_array(np.asarray(y, dtype=float))
    if x.shape != y.shape:
        raise ValueError(f"Formes incompatibles : {x.shape} vs {y.shape}")

    rx = rankdata_average(x)
    ry = rankdata_average(y)
    return pearson_correlation(rx, ry)



def fit_affine_calibration(source: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    """
    Ajuste une calibration affine :

        target ≈ a * source + b

    au sens des moindres carrés.

    Retour
    ------
    Dict[str, Any]
        - a
        - b
        - calibrated
        - regression_metrics
    """
    source = _flatten_scalar_array(np.asarray(source, dtype=float))
    target = _flatten_scalar_array(np.asarray(target, dtype=float))
    if source.shape != target.shape:
        raise ValueError(f"Formes incompatibles : {source.shape} vs {target.shape}")

    A = np.column_stack([source, np.ones_like(source)])
    coeffs, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    calibrated = a * source + b
    metrics = compute_regression_metrics(calibrated, target)

    return {
        'a': a,
        'b': b,
        'calibrated': calibrated,
        'regression_metrics': metrics,
    }


# =============================================================================
# 4. Monotonie des trajectoires d'observable
# =============================================================================

def compute_monotonicity_violations_numpy(values: np.ndarray, atol: float = 0.0) -> np.ndarray:
    """
    Calcule les violations positives d'une monotonie décroissante :

        max(0, values[n+1] - values[n] - atol)
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError('`values` doit être 1D.')
    diffs = values[1:] - values[:-1] - float(atol)
    return np.maximum(0.0, diffs)



def compute_monotonicity_metrics(values: np.ndarray, atol: float = 0.0) -> Dict[str, float | bool]:
    """
    Calcule des métriques simples de monotonie décroissante pour une suite 1D.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError('`values` doit être 1D et contenir au moins deux valeurs.')

    violations = compute_monotonicity_violations_numpy(values, atol=atol)
    diffs = values[1:] - values[:-1]

    return {
        'n_steps': int(values.size - 1),
        'monotone_fraction': float(np.mean(diffs <= atol)),
        'mean_positive_violation': float(np.mean(violations)),
        'max_positive_violation': float(np.max(violations)),
        'is_nonincreasing': bool(np.all(diffs <= atol)),
    }



def compute_trajectorywise_monotonicity(z_trajectories: np.ndarray, atol: float = 0.0) -> Dict[str, Any]:
    """
    Calcule les métriques de monotonie trajectoire par trajectoire puis les moyenne.

    Paramètres
    ----------
    z_trajectories : np.ndarray
        Tableau de forme (n_traj, n_times).

    Retour
    ------
    Dict[str, Any]
        - per_trajectory : liste de métriques
        - mean_monotone_fraction
        - mean_positive_violation
        - max_positive_violation_overall
    """
    z_trajectories = np.asarray(z_trajectories, dtype=float)
    if z_trajectories.ndim != 2:
        raise ValueError('`z_trajectories` doit être 2D de forme (n_traj, n_times).')

    per_trajectory = []
    for i in range(z_trajectories.shape[0]):
        per_trajectory.append(compute_monotonicity_metrics(z_trajectories[i], atol=atol))

    mean_monotone_fraction = float(np.mean([m['monotone_fraction'] for m in per_trajectory]))
    mean_positive_violation = float(np.mean([m['mean_positive_violation'] for m in per_trajectory]))
    max_positive_violation_overall = float(np.max([m['max_positive_violation'] for m in per_trajectory]))

    return {
        'per_trajectory': per_trajectory,
        'mean_monotone_fraction': mean_monotone_fraction,
        'mean_positive_violation': mean_positive_violation,
        'max_positive_violation_overall': max_positive_violation_overall,
    }


# =============================================================================
# 5. Extraction des sorties de modèles supervisés
# =============================================================================

@torch.no_grad()
def extract_supervised_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Extrait les prédictions d'un modèle supervisé de type `EnergyRegressor`.

    Retour
    ------
    Dict[str, np.ndarray]
        Contient notamment :
        - pred_energy
        - target_energy
        - latent
        - time
        - trajectory_index
        - time_index
    """
    model.eval()

    pred_energy_list = []
    target_energy_list = []
    latent_list = []
    time_list = []
    trajectory_index_list = []
    time_index_list = []

    for batch in dataloader:
        x = batch['u'].to(device)
        outputs = model(x)

        pred_energy_list.append(_to_numpy(outputs['energy_hat']))
        target_energy_list.append(np.asarray(batch['energy'], dtype=float).reshape(-1, 1))
        latent_list.append(_to_numpy(outputs['latent']))
        time_list.append(np.asarray(batch['time'], dtype=float).reshape(-1, 1))
        trajectory_index_list.append(np.asarray(batch['trajectory_index'], dtype=int).reshape(-1, 1))
        time_index_list.append(np.asarray(batch['time_index'], dtype=int).reshape(-1, 1))

    return {
        'pred_energy': np.vstack(pred_energy_list),
        'target_energy': np.vstack(target_energy_list),
        'latent': np.vstack(latent_list),
        'time': np.vstack(time_list)[:, 0],
        'trajectory_index': np.vstack(trajectory_index_list)[:, 0],
        'time_index': np.vstack(time_index_list)[:, 0],
    }



def evaluate_supervised_predictions(pred_energy: np.ndarray, target_energy: np.ndarray) -> Dict[str, Any]:
    """
    Évalue quantitativement des prédictions supervisées d'énergie.
    """
    pred = _flatten_scalar_array(pred_energy)
    target = _flatten_scalar_array(target_energy)

    metrics = compute_regression_metrics(pred, target)
    metrics['pearson_correlation'] = pearson_correlation(pred, target)
    metrics['spearman_correlation'] = spearman_correlation(pred, target)
    return metrics


# =============================================================================
# 6. Extraction des sorties de modèles autoencodeur latent
# =============================================================================

@torch.no_grad()
def extract_autoencoder_outputs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Extrait les sorties d'un `LatentAutoencoder` sur un DataLoader de snapshots.

    Retour
    ------
    Dict[str, np.ndarray]
        - reconstruction
        - target
        - latent
        - scalar (si disponible)
        - energy
        - dissipation
        - time
        - trajectory_index
        - time_index
    """
    model.eval()

    recon_list = []
    target_list = []
    latent_list = []
    scalar_list = []
    energy_list = []
    dissipation_list = []
    time_list = []
    traj_list = []
    time_index_list = []

    has_scalar = None

    for batch in dataloader:
        x = batch['u'].to(device)
        outputs = model(x)

        recon_list.append(_to_numpy(outputs['reconstruction']))
        target_list.append(_to_numpy(x))
        latent_list.append(_to_numpy(outputs['latent']))
        energy_list.append(np.asarray(batch['energy'], dtype=float).reshape(-1, 1))
        dissipation_list.append(np.asarray(batch['dissipation'], dtype=float).reshape(-1, 1))
        time_list.append(np.asarray(batch['time'], dtype=float).reshape(-1, 1))
        traj_list.append(np.asarray(batch['trajectory_index'], dtype=int).reshape(-1, 1))
        time_index_list.append(np.asarray(batch['time_index'], dtype=int).reshape(-1, 1))

        current_has_scalar = 'scalar' in outputs
        if has_scalar is None:
            has_scalar = current_has_scalar
        if current_has_scalar:
            scalar_list.append(_to_numpy(outputs['scalar']))

    out = {
        'reconstruction': np.vstack(recon_list),
        'target': np.vstack(target_list),
        'latent': np.vstack(latent_list),
        'energy': np.vstack(energy_list),
        'dissipation': np.vstack(dissipation_list),
        'time': np.vstack(time_list)[:, 0],
        'trajectory_index': np.vstack(traj_list)[:, 0],
        'time_index': np.vstack(time_index_list)[:, 0],
    }
    if has_scalar:
        out['scalar'] = np.vstack(scalar_list)
    return out



def evaluate_autoencoder_scalar_against_energy(scalar: np.ndarray, energy: np.ndarray) -> Dict[str, Any]:
    """
    Compare une observable scalaire apprise à l'énergie de référence.

    Retour
    ------
    Dict[str, Any]
        - direct_metrics
        - affine_calibration
        - correlations
    """
    z = _flatten_scalar_array(scalar)
    e = _flatten_scalar_array(energy)

    direct_metrics = compute_regression_metrics(z, e)
    affine = fit_affine_calibration(z, e)
    correlations = {
        'pearson_correlation': pearson_correlation(z, e),
        'spearman_correlation': spearman_correlation(z, e),
    }

    return {
        'direct_metrics': direct_metrics,
        'affine_calibration': {
            'a': affine['a'],
            'b': affine['b'],
            'regression_metrics': affine['regression_metrics'],
        },
        'correlations': correlations,
    }


# =============================================================================
# 7. Extraction et évaluation de modèles de dynamique latente
# =============================================================================

@torch.no_grad()
def extract_dynamics_outputs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Extrait les sorties d'un `LatentDynamicsModel` sur un DataLoader de paires.

    Retour
    ------
    Dict[str, np.ndarray]
        Contient notamment :
        - h_t
        - h_tp
        - h_tp_pred
        - z_t, z_tp (si disponibles)
        - energy_t, energy_tp
        - time_t, time_tp
        - trajectory_index
        - time_index_t, time_index_tp
    """
    model.eval()

    h_t_list = []
    h_tp_list = []
    h_tp_pred_list = []
    z_t_list = []
    z_tp_list = []
    z_tp_pred_list = []
    energy_t_list = []
    energy_tp_list = []
    time_t_list = []
    time_tp_list = []
    traj_list = []
    time_index_t_list = []
    time_index_tp_list = []

    has_scalar = None

    for batch in dataloader:
        u_t = batch['u_t'].to(device)
        u_tp = batch['u_tp'].to(device)
        outputs = model(u_t, u_tp)

        h_t_list.append(_to_numpy(outputs['h_t']))
        h_tp_list.append(_to_numpy(outputs['h_tp']))
        h_tp_pred_list.append(_to_numpy(outputs['h_tp_pred']))
        energy_t_list.append(np.asarray(batch['energy_t'], dtype=float).reshape(-1, 1))
        energy_tp_list.append(np.asarray(batch['energy_tp'], dtype=float).reshape(-1, 1))
        time_t_list.append(np.asarray(batch['time_t'], dtype=float).reshape(-1, 1))
        time_tp_list.append(np.asarray(batch['time_tp'], dtype=float).reshape(-1, 1))
        traj_list.append(np.asarray(batch['trajectory_index'], dtype=int).reshape(-1, 1))
        time_index_t_list.append(np.asarray(batch['time_index_t'], dtype=int).reshape(-1, 1))
        time_index_tp_list.append(np.asarray(batch['time_index_tp'], dtype=int).reshape(-1, 1))

        current_has_scalar = ('z_t' in outputs) and ('z_tp' in outputs)
        if has_scalar is None:
            has_scalar = current_has_scalar
        if current_has_scalar:
            z_t_list.append(_to_numpy(outputs['z_t']))
            z_tp_list.append(_to_numpy(outputs['z_tp']))
            z_tp_pred_list.append(_to_numpy(outputs['z_tp_pred']))

    out = {
        'h_t': np.vstack(h_t_list),
        'h_tp': np.vstack(h_tp_list),
        'h_tp_pred': np.vstack(h_tp_pred_list),
        'energy_t': np.vstack(energy_t_list),
        'energy_tp': np.vstack(energy_tp_list),
        'time_t': np.vstack(time_t_list)[:, 0],
        'time_tp': np.vstack(time_tp_list)[:, 0],
        'trajectory_index': np.vstack(traj_list)[:, 0],
        'time_index_t': np.vstack(time_index_t_list)[:, 0],
        'time_index_tp': np.vstack(time_index_tp_list)[:, 0],
    }
    if has_scalar:
        out['z_t'] = np.vstack(z_t_list)
        out['z_tp'] = np.vstack(z_tp_list)
        out['z_tp_pred'] = np.vstack(z_tp_pred_list)
    return out



def evaluate_dynamics_predictions(
    h_tp_pred: np.ndarray,
    h_tp: np.ndarray,
    z_t: Optional[np.ndarray] = None,
    z_tp: Optional[np.ndarray] = None,
    energy_t: Optional[np.ndarray] = None,
    energy_tp: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Évalue quantitativement un modèle de dynamique latente.

    Retour
    ------
    Dict[str, Any]
        - latent_prediction_metrics
        - scalar_monotonicity_pairwise (si z fourni)
        - scalar_vs_energy_t (si z_t et energy_t fournis)
        - scalar_vs_energy_tp (si z_tp et energy_tp fournis)
    """
    h_tp_pred = np.asarray(h_tp_pred, dtype=float)
    h_tp = np.asarray(h_tp, dtype=float)
    if h_tp_pred.shape != h_tp.shape:
        raise ValueError(f"Formes incompatibles pour h_tp_pred et h_tp : {h_tp_pred.shape} vs {h_tp.shape}")

    latent_error = h_tp_pred - h_tp
    latent_prediction_metrics = {
        'mse': float(np.mean(latent_error ** 2)),
        'rmse': float(np.sqrt(np.mean(latent_error ** 2))),
        'mae': float(np.mean(np.abs(latent_error))),
    }

    out: Dict[str, Any] = {
        'latent_prediction_metrics': latent_prediction_metrics,
    }

    if z_t is not None and z_tp is not None:
        z_t_flat = _flatten_scalar_array(z_t)
        z_tp_flat = _flatten_scalar_array(z_tp)
        pairwise_violation = np.maximum(0.0, z_tp_flat - z_t_flat)
        out['scalar_monotonicity_pairwise'] = {
            'mean_positive_violation': float(np.mean(pairwise_violation)),
            'max_positive_violation': float(np.max(pairwise_violation)),
            'monotone_fraction': float(np.mean(z_tp_flat <= z_t_flat)),
        }

    if z_t is not None and energy_t is not None:
        out['scalar_vs_energy_t'] = evaluate_autoencoder_scalar_against_energy(z_t, energy_t)
    if z_tp is not None and energy_tp is not None:
        out['scalar_vs_energy_tp'] = evaluate_autoencoder_scalar_against_energy(z_tp, energy_tp)

    return out


# =============================================================================
# 8. Reconstruction de trajectoires scalaires et métriques trajectorielles
# =============================================================================

def reconstruct_scalar_trajectories(
    scalar_values: np.ndarray,
    trajectory_index: np.ndarray,
    time_index: np.ndarray,
    n_trajectories: Optional[int] = None,
    n_times: Optional[int] = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Reconstruit un tableau 2D (n_traj, n_times) à partir de valeurs scalaires
    plates accompagnées des indices de trajectoire et de temps.

    Paramètres
    ----------
    scalar_values : np.ndarray
        Tableau de forme (n_samples,) ou (n_samples, 1).
    trajectory_index : np.ndarray
        Indice de trajectoire pour chaque sample.
    time_index : np.ndarray
        Indice temporel pour chaque sample.
    n_trajectories : int ou None
        Nombre total de trajectoires. Si None, inféré par max + 1.
    n_times : int ou None
        Nombre total de temps. Si None, inféré par max + 1.
    fill_value : float
        Valeur initiale de remplissage.
    """
    scalar_values = _flatten_scalar_array(np.asarray(scalar_values, dtype=float))
    trajectory_index = np.asarray(trajectory_index, dtype=int)
    time_index = np.asarray(time_index, dtype=int)

    if not (scalar_values.shape[0] == trajectory_index.shape[0] == time_index.shape[0]):
        raise ValueError('Les vecteurs scalar_values, trajectory_index et time_index doivent avoir la même longueur.')

    if n_trajectories is None:
        n_trajectories = int(np.max(trajectory_index)) + 1
    if n_times is None:
        n_times = int(np.max(time_index)) + 1

    out = np.full((n_trajectories, n_times), fill_value, dtype=float)
    out[trajectory_index, time_index] = scalar_values
    return out



def evaluate_scalar_trajectory_monotonicity(
    scalar_values: np.ndarray,
    trajectory_index: np.ndarray,
    time_index: np.ndarray,
    n_trajectories: Optional[int] = None,
    n_times: Optional[int] = None,
    atol: float = 0.0,
) -> Dict[str, Any]:
    """
    Reconstruit les trajectoires d'une observable scalaire puis calcule des
    métriques de monotonie trajectoire par trajectoire.
    """
    trajectories = reconstruct_scalar_trajectories(
        scalar_values=scalar_values,
        trajectory_index=trajectory_index,
        time_index=time_index,
        n_trajectories=n_trajectories,
        n_times=n_times,
    )

    if np.any(np.isnan(trajectories)):
        raise ValueError(
            'Certaines trajectoires reconstruites contiennent des NaN. '
            'Vérifiez la couverture des indices (trajectory_index, time_index).'
        )

    metrics = compute_trajectorywise_monotonicity(trajectories, atol=atol)
    metrics['scalar_trajectories'] = trajectories
    return metrics


# =============================================================================
# 9. Interfaces haut niveau pratiques
# =============================================================================

def evaluate_saved_supervised_model(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    batch_size: int = 128,
    device: Optional[str] = None,
    output_json_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Charge un modèle supervisé sauvegardé et l'évalue sur un dataset.

    Paramètres
    ----------
    checkpoint_path : str | Path
        Checkpoint `.pt` du modèle supervisé.
    dataset_path : str | Path
        Fichier `.npz` du dataset d'évaluation.
    batch_size : int
        Taille de batch pour l'inférence.
    device : str ou None
        Device d'inférence.
    output_json_path : str | Path ou None
        Si fourni, sauvegarde le rapport JSON à cet emplacement.
    """
    (
        _, _, _, BurgersSnapshotDataset, _, _,
        EnergyRegressor, _, _
    ) = load_project_symbols()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['model_config']
    standardizer_dict = checkpoint.get('standardizer', None)

    dataset = BurgersSnapshotDataset(
        dataset_path,
        normalize=standardizer_dict is not None,
        standardizer=None if standardizer_dict is None else _standardizer_from_dict(standardizer_dict),
        return_metadata=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if device is None:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device_obj = torch.device(device)

    model = EnergyRegressor(
        input_dim=int(model_config['input_dim']),
        latent_dim=int(model_config['latent_dim']),
        encoder_hidden_dims=tuple(model_config['encoder_hidden_dims']),
        head_hidden_dims=tuple(model_config['head_hidden_dims']),
        activation=str(model_config['activation']),
        dropout=float(model_config['dropout']),
        positive_output=bool(model_config['positive_output']),
    ).to(device_obj)
    model.load_state_dict(checkpoint['model_state_dict'])

    extracted = extract_supervised_predictions(model, loader, device=device_obj)
    metrics = evaluate_supervised_predictions(extracted['pred_energy'], extracted['target_energy'])

    report = {
        'checkpoint_path': str(checkpoint_path),
        'dataset_path': str(dataset_path),
        'metrics': metrics,
        'best_val_loss_from_training': float(checkpoint.get('best_val_loss', np.nan)),
    }

    if output_json_path is not None:
        save_json(output_json_path, report)

    return {
        'report': report,
        'extracted': extracted,
    }



def evaluate_saved_latent_autoencoder(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    batch_size: int = 128,
    device: Optional[str] = None,
    output_json_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Charge un checkpoint d'autoencodeur latent et l'évalue sur un dataset.
    """
    (
        _, _, _, BurgersSnapshotDataset, _, _,
        _, LatentAutoencoder, _
    ) = load_project_symbols()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['model_config']
    standardizer_dict = checkpoint.get('standardizer', None)

    dataset = BurgersSnapshotDataset(
        dataset_path,
        normalize=standardizer_dict is not None,
        standardizer=None if standardizer_dict is None else _standardizer_from_dict(standardizer_dict),
        return_metadata=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if device is None:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device_obj = torch.device(device)

    model = LatentAutoencoder(
        input_dim=int(model_config['input_dim']),
        latent_dim=int(model_config['latent_dim']),
        encoder_hidden_dims=tuple(model_config['encoder_hidden_dims']),
        decoder_hidden_dims=tuple(model_config['decoder_hidden_dims']),
        activation=str(model_config['activation']),
        dropout=float(model_config['dropout']),
        use_scalar_head=bool(model_config['use_scalar_head']),
        scalar_head_hidden_dims=tuple(model_config['scalar_head_hidden_dims']),
        positive_scalar_output=bool(model_config['positive_scalar_output']),
    ).to(device_obj)
    model.load_state_dict(checkpoint['model_state_dict'])

    extracted = extract_autoencoder_outputs(model, loader, device=device_obj)

    reconstruction_metrics = {
        'mse': float(np.mean((extracted['reconstruction'] - extracted['target']) ** 2)),
        'rmse': float(np.sqrt(np.mean((extracted['reconstruction'] - extracted['target']) ** 2))),
        'mae': float(np.mean(np.abs(extracted['reconstruction'] - extracted['target']))),
    }

    report: Dict[str, Any] = {
        'checkpoint_path': str(checkpoint_path),
        'dataset_path': str(dataset_path),
        'reconstruction_metrics': reconstruction_metrics,
        'best_val_loss_from_training': float(checkpoint.get('best_val_loss', np.nan)),
    }

    if 'scalar' in extracted:
        scalar_vs_energy = evaluate_autoencoder_scalar_against_energy(extracted['scalar'], extracted['energy'])
        scalar_monotonicity = evaluate_scalar_trajectory_monotonicity(
            scalar_values=extracted['scalar'],
            trajectory_index=extracted['trajectory_index'],
            time_index=extracted['time_index'],
        )
        report['scalar_vs_energy'] = scalar_vs_energy
        report['scalar_monotonicity'] = {
            'mean_monotone_fraction': scalar_monotonicity['mean_monotone_fraction'],
            'mean_positive_violation': scalar_monotonicity['mean_positive_violation'],
            'max_positive_violation_overall': scalar_monotonicity['max_positive_violation_overall'],
        }

    if output_json_path is not None:
        save_json(output_json_path, report)

    return {
        'report': report,
        'extracted': extracted,
    }



def evaluate_saved_latent_dynamics_model(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    batch_size: int = 128,
    stride: Optional[int] = None,
    device: Optional[str] = None,
    output_json_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Charge un checkpoint de dynamique latente et l'évalue sur un dataset.
    """
    (
        _, _, _, _, BurgersPairDataset, _,
        _, _, LatentDynamicsModel
    ) = load_project_symbols()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['model_config']
    standardizer_dict = checkpoint.get('standardizer', None)
    if stride is None:
        stride = int(checkpoint.get('stride', 1))

    dataset = BurgersPairDataset(
        dataset_path,
        normalize=standardizer_dict is not None,
        standardizer=None if standardizer_dict is None else _standardizer_from_dict(standardizer_dict),
        stride=int(stride),
        return_metadata=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if device is None:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device_obj = torch.device(device)

    model = LatentDynamicsModel(
        input_dim=int(model_config['input_dim']),
        latent_dim=int(model_config['latent_dim']),
        encoder_hidden_dims=tuple(model_config['encoder_hidden_dims']),
        predictor_hidden_dims=tuple(model_config['predictor_hidden_dims']),
        activation=str(model_config['activation']),
        dropout=float(model_config['dropout']),
        use_scalar_head=bool(model_config['use_scalar_head']),
        scalar_head_hidden_dims=tuple(model_config['scalar_head_hidden_dims']),
        positive_scalar_output=bool(model_config['positive_scalar_output']),
        residual_predictor=bool(model_config['residual_predictor']),
    ).to(device_obj)
    model.load_state_dict(checkpoint['model_state_dict'])

    extracted = extract_dynamics_outputs(model, loader, device=device_obj)
    report = evaluate_dynamics_predictions(
        h_tp_pred=extracted['h_tp_pred'],
        h_tp=extracted['h_tp'],
        z_t=extracted.get('z_t', None),
        z_tp=extracted.get('z_tp', None),
        energy_t=extracted.get('energy_t', None),
        energy_tp=extracted.get('energy_tp', None),
    )

    # Reconstruction trajectoire par trajectoire de z_t si disponible.
    if 'z_t' in extracted:
        scalar_monotonicity = evaluate_scalar_trajectory_monotonicity(
            scalar_values=extracted['z_t'],
            trajectory_index=extracted['trajectory_index'],
            time_index=extracted['time_index_t'],
        )
        report['scalar_trajectory_monotonicity'] = {
            'mean_monotone_fraction': scalar_monotonicity['mean_monotone_fraction'],
            'mean_positive_violation': scalar_monotonicity['mean_positive_violation'],
            'max_positive_violation_overall': scalar_monotonicity['max_positive_violation_overall'],
        }

    report['checkpoint_path'] = str(checkpoint_path)
    report['dataset_path'] = str(dataset_path)
    report['best_val_loss_from_training'] = float(checkpoint.get('best_val_loss', np.nan))
    report['stride'] = int(stride)

    if output_json_path is not None:
        save_json(output_json_path, report)

    return {
        'report': report,
        'extracted': extracted,
    }


# =============================================================================
# 10. Standardizer utilitaire
# =============================================================================

class _SimpleStandardizer:
    """
    Petit standardizer minimal pour reconstruire un objet à partir d'un dict.
    Utilisé seulement au chargement pour réappliquer la même transformation.
    """
    def __init__(self, mean: float, std: float, eps: float = 1.0e-8):
        self.mean = float(mean)
        self.std = float(std)
        self.eps = float(eps)

    def transform(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean) / max(self.std, self.eps)

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        return array * max(self.std, self.eps) + self.mean

    def to_dict(self) -> Dict[str, float]:
        return {'mean': self.mean, 'std': self.std, 'eps': self.eps}


def _standardizer_from_dict(data: Dict[str, Any]) -> _SimpleStandardizer:
    """Reconstruit un standardizer minimum à partir d'un dictionnaire."""
    return _SimpleStandardizer(
        mean=float(data['mean']),
        std=float(data['std']),
        eps=float(data.get('eps', 1.0e-8)),
    )


# =============================================================================
# 11. Bloc de test minimal
# =============================================================================

if __name__ == '__main__':
    # Tests rapides, purement analytiques, sans modèle réellement entraîné.
    rng = np.random.default_rng(123)
    z = rng.normal(size=20)
    e = 2.0 * z + 1.0 + 0.1 * rng.normal(size=20)

    print('Test rapide de ml/evaluate.py')
    print('- Pearson(z,e) :', pearson_correlation(z, e))
    print('- Spearman(z,e):', spearman_correlation(z, e))
    print('- Regression metrics:', compute_regression_metrics(z, e))
    affine = fit_affine_calibration(z, e)
    print('- Affine calibration a,b:', affine['a'], affine['b'])

    traj_idx = np.repeat(np.arange(3), 5)
    time_idx = np.tile(np.arange(5), 3)
    scalar_values = np.linspace(1.0, 0.1, 15)
    monot = evaluate_scalar_trajectory_monotonicity(scalar_values, traj_idx, time_idx)
    print('- Mean monotone fraction:', monot['mean_monotone_fraction'])
