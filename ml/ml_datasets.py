"""
ml/datasets.py
==============

Jeux de données PyTorch pour le projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce module fournit des classes `torch.utils.data.Dataset` simples et lisibles pour
consommer les fichiers `.npz` produits par la génération de données.

Les cas d'usage principaux sont :

1. apprentissage supervisé de l'énergie,
2. apprentissage sur snapshots individuels,
3. apprentissage sur paires temporelles (u_t, u_{t+1}),
4. apprentissage sur trajectoires complètes,
5. normalisation globale des états si souhaité.

Format attendu du dataset
-------------------------
Le module s'attend à un dictionnaire ou un fichier `.npz` contenant au moins :

- states      : tableau (n_traj, n_times, Nx)
- times       : tableau (n_times,)
- energy      : tableau (n_traj, n_times)
- dissipation : tableau (n_traj, n_times)
- x           : tableau (Nx,)
- dx          : scalaire

Remarque
--------
Le code est volontairement explicite et détaillé, pour rester pédagogique et
facile à modifier dans le cadre d'un prototype de recherche.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# =============================================================================
# 1. Chargement / validation du dataset brut
# =============================================================================

def load_dataset_npz(path: str | Path) -> Dict[str, Any]:
    """
    Recharge un dataset sauvegardé au format `.npz`.

    Paramètres
    ----------
    path : str ou Path
        Chemin du fichier `.npz`.

    Retour
    ------
    Dict[str, Any]
        Dictionnaire contenant les tableaux du dataset.
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}



def validate_dataset_dict(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vérifie la présence des entrées minimales et la cohérence des dimensions.

    Paramètres
    ----------
    dataset : Dict[str, Any]
        Dictionnaire du dataset.

    Retour
    ------
    Dict[str, Any]
        Le dictionnaire inchangé si les vérifications passent.
    """
    required_keys = ["states", "times", "energy", "dissipation", "x", "dx"]
    missing = [key for key in required_keys if key not in dataset]
    if missing:
        raise KeyError(f"Clés manquantes dans le dataset : {missing}")

    states = np.asarray(dataset["states"])
    times = np.asarray(dataset["times"])
    energy = np.asarray(dataset["energy"])
    dissipation = np.asarray(dataset["dissipation"])
    x = np.asarray(dataset["x"])

    if states.ndim != 3:
        raise ValueError("`states` doit être de forme (n_traj, n_times, Nx).")
    if times.ndim != 1:
        raise ValueError("`times` doit être un tableau 1D.")
    if energy.ndim != 2:
        raise ValueError("`energy` doit être de forme (n_traj, n_times).")
    if dissipation.ndim != 2:
        raise ValueError("`dissipation` doit être de forme (n_traj, n_times).")
    if x.ndim != 1:
        raise ValueError("`x` doit être un tableau 1D.")

    n_traj, n_times, nx = states.shape
    if times.shape[0] != n_times:
        raise ValueError("La taille de `times` est incompatible avec `states`.")
    if energy.shape != (n_traj, n_times):
        raise ValueError("La forme de `energy` est incompatible avec `states`.")
    if dissipation.shape != (n_traj, n_times):
        raise ValueError("La forme de `dissipation` est incompatible avec `states`.")
    if x.shape[0] != nx:
        raise ValueError("La taille de `x` est incompatible avec la dimension spatiale de `states`.")

    return dataset


# =============================================================================
# 2. Normalisation globale des états
# =============================================================================

@dataclass
class GlobalStandardizer:
    """
    Standardisation globale simple :

        u_normalisé = (u - mean) / std

    où `mean` et `std` sont calculés sur l'ensemble des snapshots du train.
    """

    mean: float
    std: float
    eps: float = 1.0e-8

    def transform(self, array: np.ndarray) -> np.ndarray:
        """Applique la normalisation à un tableau numpy."""
        return (array - self.mean) / max(self.std, self.eps)

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        """Revient à l'échelle physique initiale."""
        return array * max(self.std, self.eps) + self.mean

    def to_dict(self) -> Dict[str, float]:
        """Convertit l'objet en dictionnaire sérialisable."""
        return {"mean": float(self.mean), "std": float(self.std), "eps": float(self.eps)}



def fit_global_standardizer(states: np.ndarray, eps: float = 1.0e-8) -> GlobalStandardizer:
    """
    Calcule les statistiques globales sur un tableau `states` de forme
    (n_traj, n_times, Nx).
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 3:
        raise ValueError("`states` doit être de forme (n_traj, n_times, Nx).")

    mean = float(np.mean(states))
    std = float(np.std(states))
    return GlobalStandardizer(mean=mean, std=std, eps=eps)


# =============================================================================
# 3. Transformation numpy -> torch
# =============================================================================

def to_float_tensor(array: np.ndarray) -> torch.Tensor:
    """
    Convertit un tableau numpy en tenseur torch float32.
    """
    return torch.as_tensor(array, dtype=torch.float32)


# =============================================================================
# 4. Dataset de snapshots individuels
# =============================================================================

class BurgersSnapshotDataset(Dataset):
    """
    Dataset PyTorch où chaque item correspond à un snapshot individuel.

    Cas d'usage typiques
    --------------------
    - régression supervisée vers l'énergie E(u),
    - apprentissage d'un encodeur sur snapshots,
    - diagnostic de variables physiques sur instantanés.

    Chaque item renvoie un dictionnaire contenant au minimum :
    - u                : snapshot spatial (Nx,)
    - time             : temps associé
    - trajectory_index : indice de trajectoire
    - time_index       : indice temporel

    Et, si disponibles :
    - energy
    - dissipation
    - ic_type_id
    - nu
    """

    def __init__(
        self,
        dataset: Dict[str, Any] | str | Path,
        normalize: bool = False,
        standardizer: Optional[GlobalStandardizer] = None,
        return_metadata: bool = True,
    ) -> None:
        """
        Paramètres
        ----------
        dataset : dict ou chemin `.npz`
            Dataset déjà chargé ou chemin vers un fichier sauvegardé.
        normalize : bool
            Si True, applique une standardisation globale aux snapshots.
        standardizer : GlobalStandardizer ou None
            Standardizer pré-calculé. Si `normalize=True` et `standardizer=None`,
            il est ajusté directement sur le dataset fourni.
        return_metadata : bool
            Si True, inclut les métadonnées utiles dans chaque item.
        """
        if isinstance(dataset, (str, Path)):
            dataset = load_dataset_npz(dataset)
        dataset = validate_dataset_dict(dataset)

        self.dataset = dataset
        self.states = np.asarray(dataset["states"], dtype=np.float32)
        self.times = np.asarray(dataset["times"], dtype=np.float32)
        self.energy = np.asarray(dataset["energy"], dtype=np.float32)
        self.dissipation = np.asarray(dataset["dissipation"], dtype=np.float32)

        self.n_traj, self.n_times, self.nx = self.states.shape
        self.return_metadata = return_metadata
        self.normalize = normalize

        if normalize:
            self.standardizer = standardizer or fit_global_standardizer(self.states)
            self.states = self.standardizer.transform(self.states).astype(np.float32)
        else:
            self.standardizer = standardizer

        # Indexation plate : chaque entier global idx renvoie (traj_idx, time_idx).
        self.index_map = [(i, t) for i in range(self.n_traj) for t in range(self.n_times)]

        self.ic_type_ids = np.asarray(dataset["ic_type_ids"], dtype=np.int64) if "ic_type_ids" in dataset else None
        self.nu_values = np.asarray(dataset["nu_values"], dtype=np.float32) if "nu_values" in dataset else None

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | float]:
        traj_idx, time_idx = self.index_map[idx]

        item: Dict[str, Any] = {
            "u": to_float_tensor(self.states[traj_idx, time_idx]),
        }

        if self.return_metadata:
            item["time"] = float(self.times[time_idx])
            item["trajectory_index"] = int(traj_idx)
            item["time_index"] = int(time_idx)
            item["energy"] = float(self.energy[traj_idx, time_idx])
            item["dissipation"] = float(self.dissipation[traj_idx, time_idx])

            if self.ic_type_ids is not None:
                item["ic_type_id"] = int(self.ic_type_ids[traj_idx])
            if self.nu_values is not None:
                item["nu"] = float(self.nu_values[traj_idx])

        return item


# =============================================================================
# 5. Dataset de paires temporelles (u_t, u_{t+1})
# =============================================================================

class BurgersPairDataset(Dataset):
    """
    Dataset PyTorch où chaque item correspond à une paire temporelle :

        (u_t, u_{t+1})

    Cas d'usage typiques
    --------------------
    - apprentissage de dynamique latente,
    - prédiction à un pas,
    - pénalités de monotonie entre t et t+1,
    - encodeur + prédicteur latent.
    """

    def __init__(
        self,
        dataset: Dict[str, Any] | str | Path,
        normalize: bool = False,
        standardizer: Optional[GlobalStandardizer] = None,
        stride: int = 1,
        return_metadata: bool = True,
    ) -> None:
        """
        Paramètres
        ----------
        dataset : dict ou chemin `.npz`
            Dataset source.
        normalize : bool
            Si True, applique une standardisation globale aux snapshots.
        standardizer : GlobalStandardizer ou None
            Standardizer pré-calculé.
        stride : int
            Pas temporel entre les deux états :
                u_t -> u_{t+stride}
        return_metadata : bool
            Si True, inclut l'énergie, la dissipation et les indices.
        """
        if isinstance(dataset, (str, Path)):
            dataset = load_dataset_npz(dataset)
        dataset = validate_dataset_dict(dataset)

        self.dataset = dataset
        self.states = np.asarray(dataset["states"], dtype=np.float32)
        self.times = np.asarray(dataset["times"], dtype=np.float32)
        self.energy = np.asarray(dataset["energy"], dtype=np.float32)
        self.dissipation = np.asarray(dataset["dissipation"], dtype=np.float32)

        self.n_traj, self.n_times, self.nx = self.states.shape
        self.return_metadata = return_metadata

        if stride < 1:
            raise ValueError("`stride` doit être >= 1.")
        if stride >= self.n_times:
            raise ValueError("`stride` doit être strictement plus petit que le nombre de temps.")
        self.stride = int(stride)

        if normalize:
            self.standardizer = standardizer or fit_global_standardizer(self.states)
            self.states = self.standardizer.transform(self.states).astype(np.float32)
        else:
            self.standardizer = standardizer

        # Indexation plate pour les paires temporelles.
        self.index_map = [
            (i, t)
            for i in range(self.n_traj)
            for t in range(self.n_times - self.stride)
        ]

        self.ic_type_ids = np.asarray(dataset["ic_type_ids"], dtype=np.int64) if "ic_type_ids" in dataset else None
        self.nu_values = np.asarray(dataset["nu_values"], dtype=np.float32) if "nu_values" in dataset else None

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | float]:
        traj_idx, time_idx = self.index_map[idx]
        next_idx = time_idx + self.stride

        item: Dict[str, Any] = {
            "u_t": to_float_tensor(self.states[traj_idx, time_idx]),
            "u_tp": to_float_tensor(self.states[traj_idx, next_idx]),
        }

        if self.return_metadata:
            item["time_t"] = float(self.times[time_idx])
            item["time_tp"] = float(self.times[next_idx])
            item["trajectory_index"] = int(traj_idx)
            item["time_index_t"] = int(time_idx)
            item["time_index_tp"] = int(next_idx)
            item["energy_t"] = float(self.energy[traj_idx, time_idx])
            item["energy_tp"] = float(self.energy[traj_idx, next_idx])
            item["dissipation_t"] = float(self.dissipation[traj_idx, time_idx])
            item["dissipation_tp"] = float(self.dissipation[traj_idx, next_idx])

            if self.ic_type_ids is not None:
                item["ic_type_id"] = int(self.ic_type_ids[traj_idx])
            if self.nu_values is not None:
                item["nu"] = float(self.nu_values[traj_idx])

        return item


# =============================================================================
# 6. Dataset de trajectoires complètes
# =============================================================================

class BurgersTrajectoryDataset(Dataset):
    """
    Dataset où chaque item correspond à une trajectoire complète.

    Cas d'usage typiques
    --------------------
    - modèles séquentiels,
    - encodeurs temporels,
    - analyses globales trajectoire par trajectoire,
    - apprentissage d'observables sur des séquences entières.
    """

    def __init__(
        self,
        dataset: Dict[str, Any] | str | Path,
        normalize: bool = False,
        standardizer: Optional[GlobalStandardizer] = None,
        return_metadata: bool = True,
    ) -> None:
        if isinstance(dataset, (str, Path)):
            dataset = load_dataset_npz(dataset)
        dataset = validate_dataset_dict(dataset)

        self.dataset = dataset
        self.states = np.asarray(dataset["states"], dtype=np.float32)
        self.times = np.asarray(dataset["times"], dtype=np.float32)
        self.energy = np.asarray(dataset["energy"], dtype=np.float32)
        self.dissipation = np.asarray(dataset["dissipation"], dtype=np.float32)
        self.x = np.asarray(dataset["x"], dtype=np.float32)
        self.dx = float(np.asarray(dataset["dx"]).item())

        self.n_traj, self.n_times, self.nx = self.states.shape
        self.return_metadata = return_metadata

        if normalize:
            self.standardizer = standardizer or fit_global_standardizer(self.states)
            self.states = self.standardizer.transform(self.states).astype(np.float32)
        else:
            self.standardizer = standardizer

        self.ic_type_ids = np.asarray(dataset["ic_type_ids"], dtype=np.int64) if "ic_type_ids" in dataset else None
        self.nu_values = np.asarray(dataset["nu_values"], dtype=np.float32) if "nu_values" in dataset else None

    def __len__(self) -> int:
        return self.n_traj

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | float]:
        item: Dict[str, Any] = {
            "states": to_float_tensor(self.states[idx]),  # forme (n_times, Nx)
            "times": to_float_tensor(self.times),
            "energy": to_float_tensor(self.energy[idx]),
            "dissipation": to_float_tensor(self.dissipation[idx]),
        }

        if self.return_metadata:
            item["trajectory_index"] = int(idx)
            item["dx"] = float(self.dx)
            item["x"] = to_float_tensor(self.x)

            if self.ic_type_ids is not None:
                item["ic_type_id"] = int(self.ic_type_ids[idx])
            if self.nu_values is not None:
                item["nu"] = float(self.nu_values[idx])

        return item


# =============================================================================
# 7. Utilitaires pratiques
# =============================================================================

def describe_dataset(dataset: Dict[str, Any] | str | Path) -> Dict[str, Any]:
    """
    Retourne un petit résumé structurel du dataset brut.
    """
    if isinstance(dataset, (str, Path)):
        dataset = load_dataset_npz(dataset)
    dataset = validate_dataset_dict(dataset)

    states = np.asarray(dataset["states"])
    times = np.asarray(dataset["times"])
    energy = np.asarray(dataset["energy"])

    summary = {
        "n_trajectories": int(states.shape[0]),
        "n_times": int(states.shape[1]),
        "nx": int(states.shape[2]),
        "time_min": float(np.min(times)),
        "time_max": float(np.max(times)),
        "mean_initial_energy": float(np.mean(energy[:, 0])),
        "mean_final_energy": float(np.mean(energy[:, -1])),
    }

    if "ic_type_names" in dataset and "ic_type_ids" in dataset:
        ic_names = np.asarray(dataset["ic_type_names"])
        ic_ids = np.asarray(dataset["ic_type_ids"])
        unique_ids, counts = np.unique(ic_ids, return_counts=True)
        summary["ic_distribution"] = {
            str(ic_names[idx]): int(count)
            for idx, count in zip(unique_ids, counts)
        }

    return summary


# =============================================================================
# 8. Bloc de test minimal
# =============================================================================

if __name__ == "__main__":
    # Ce bloc construit un mini dataset synthétique en mémoire pour vérifier
    # que les classes se comportent correctement, sans dépendre d'un fichier réel.
    n_traj, n_times, nx = 3, 5, 8
    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    times = np.linspace(0.0, 1.0, n_times)

    rng = np.random.default_rng(123)
    states = rng.normal(size=(n_traj, n_times, nx)).astype(np.float32)
    energy = 0.5 * np.sum(states ** 2, axis=2)
    dissipation = np.abs(rng.normal(size=(n_traj, n_times))).astype(np.float32)

    fake_dataset = {
        "states": states,
        "times": times,
        "energy": energy,
        "dissipation": dissipation,
        "x": x,
        "dx": np.array(x[1] - x[0], dtype=float),
        "ic_type_ids": np.array([0, 1, 0], dtype=int),
        "ic_type_names": np.array(["fourier", "gaussian_bumps"]),
        "nu_values": np.array([0.05, 0.05, 0.05], dtype=float),
    }

    print("Test de describe_dataset")
    print(describe_dataset(fake_dataset))

    snapshot_ds = BurgersSnapshotDataset(fake_dataset, normalize=True)
    pair_ds = BurgersPairDataset(fake_dataset, normalize=True, stride=1)
    traj_ds = BurgersTrajectoryDataset(fake_dataset, normalize=False)

    print("\nTailles des datasets :")
    print("- Snapshot dataset  :", len(snapshot_ds))
    print("- Pair dataset      :", len(pair_ds))
    print("- Trajectory dataset:", len(traj_ds))

    first_snapshot = snapshot_ds[0]
    first_pair = pair_ds[0]
    first_traj = traj_ds[0]

    print("\nClés du premier item snapshot :", list(first_snapshot.keys()))
    print("Clés du premier item pair     :", list(first_pair.keys()))
    print("Clés du premier item traj     :", list(first_traj.keys()))
