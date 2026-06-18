"""
ml/train_supervised.py
======================

Entraînement supervisé d'un modèle de régression d'énergie pour le projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce module fournit un pipeline clair, lisible et robuste pour apprendre
à prédire l'énergie d'un snapshot de Burgers à partir des données générées.

Schéma général
--------------
1. charger le dataset train / validation,
2. construire les Dataset / DataLoader PyTorch,
3. initialiser le modèle `EnergyRegressor`,
4. entraîner avec une loss MSE,
5. suivre les métriques train / val,
6. sauvegarder le meilleur modèle et l'historique.

Philosophie
-----------
- code simple et pédagogique,
- dépendances limitées,
- robustesse aux imports (prototype en cours de construction),
- séparation claire entre : préparation des données, entraînement, évaluation,
  sauvegarde des résultats.

Remarque
--------
Ce module est écrit de manière à fonctionner même si l'arborescence complète
n'est pas encore installée comme package Python. Les imports critiques sont donc
chargés de manière flexible à l'intérieur de fonctions utilitaires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.util
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =============================================================================
# 1. Chargement flexible des modules du projet
# =============================================================================

def _load_module_from_candidates(module_name: str, candidates: List[str]):
    """
    Charge dynamiquement un module depuis une liste de chemins candidats.

    Paramètres
    ----------
    module_name : str
        Nom logique du module à charger.
    candidates : list[str]
        Liste de chemins fichiers à tester, dans l'ordre.

    Retour
    ------
    module
        Module Python chargé.
    """
    # 1) Essai d'import direct si module_name est importable tel quel.
    try:
        return __import__(module_name, fromlist=['*'])
    except ModuleNotFoundError:
        pass

    # 2) Essai via fichiers locaux.
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(
        f"Impossible de charger le module {module_name!r}. "
        f"Candidats testés : {candidates}"
    )



def load_project_symbols():
    """
    Charge les symboles utiles depuis :
    - config.py
    - ml_datasets.py / ml/datasets.py
    - ml_models.py / ml/models.py

    Retour
    ------
    tuple
        (get_default_config, BurgersSnapshotDataset, EnergyRegressor, describe_dataset)
    """
    # Configuration
    try:
        from config import get_default_config  # type: ignore
    except ModuleNotFoundError:
        config_module = _load_module_from_candidates(
            module_name='config',
            candidates=['config.py'],
        )
        get_default_config = config_module.get_default_config

    # Datasets
    datasets_module = _load_module_from_candidates(
        module_name='ml_datasets',
        candidates=['ml_datasets.py', 'ml/datasets.py'],
    )

    # Models
    models_module = _load_module_from_candidates(
        module_name='ml_models',
        candidates=['ml_models.py', 'ml/models.py'],
    )

    BurgersSnapshotDataset = datasets_module.BurgersSnapshotDataset
    describe_dataset = datasets_module.describe_dataset
    EnergyRegressor = models_module.EnergyRegressor
    summarize_model = models_module.summarize_model

    return get_default_config, BurgersSnapshotDataset, EnergyRegressor, summarize_model, describe_dataset


# =============================================================================
# 2. Reproductibilité et utilitaires
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Fixe les graines aléatoires principales pour la reproductibilité."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str | Path) -> Path:
    """Crée un dossier s'il n'existe pas et renvoie le Path correspondant."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Sauvegarde un dictionnaire dans un fichier JSON lisible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def print_section(title: str) -> None:
    """Affiche un titre lisible dans le terminal."""
    line = '=' * len(title)
    print(f"\n{title}\n{line}")


# =============================================================================
# 3. Préparation des DataLoaders
# =============================================================================

def build_supervised_dataloaders(
    train_dataset_path: str | Path,
    val_dataset_path: str | Path,
    batch_size: int,
    normalize: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Any, Any]:
    """
    Construit les DataLoaders train / val pour l'apprentissage supervisé.

    Paramètres
    ----------
    train_dataset_path : str ou Path
        Fichier .npz du split train.
    val_dataset_path : str ou Path
        Fichier .npz du split validation.
    batch_size : int
        Taille de batch.
    normalize : bool
        Si True, applique une normalisation globale (fittée sur le train).
    num_workers : int
        Nombre de workers des DataLoaders.

    Retour
    ------
    Tuple[DataLoader, DataLoader, Dataset, Dataset]
        train_loader, val_loader, train_dataset, val_dataset
    """
    _, BurgersSnapshotDataset, _, _, _ = load_project_symbols()

    # On construit d'abord le dataset train. Si normalize=True, le standardizer
    # est ajusté automatiquement sur le train par la classe elle-même.
    train_dataset = BurgersSnapshotDataset(
        train_dataset_path,
        normalize=normalize,
        standardizer=None,
        return_metadata=True,
    )

    # On réutilise exactement le même standardizer pour le split validation.
    val_dataset = BurgersSnapshotDataset(
        val_dataset_path,
        normalize=normalize,
        standardizer=train_dataset.standardizer,
        return_metadata=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_dataset, val_dataset


# =============================================================================
# 4. Boucles train / évaluation
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Exécute une époque d'entraînement.

    Retour
    ------
    Dict[str, float]
        Métriques moyennes de l'époque.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        x = batch['u'].to(device)
        target = torch.as_tensor(batch['energy'], dtype=torch.float32, device=device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(x)
        pred = outputs['energy_hat']
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    return {
        'loss': mean_loss,
    }


@torch.no_grad()
def evaluate_supervised_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Évalue le modèle sur un DataLoader donné.

    Retour
    ------
    Dict[str, float]
        Dictionnaire contenant :
        - loss
        - mse
        - mae
    """
    model.eval()

    total_loss = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_samples = 0

    for batch in dataloader:
        x = batch['u'].to(device)
        target = torch.as_tensor(batch['energy'], dtype=torch.float32, device=device).view(-1, 1)

        outputs = model(x)
        pred = outputs['energy_hat']

        loss = loss_fn(pred, target)
        abs_error = torch.abs(pred - target)
        sq_error = (pred - target) ** 2

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_abs_error += float(abs_error.mean().item()) * batch_size
        total_sq_error += float(sq_error.mean().item()) * batch_size
        total_samples += batch_size

    return {
        'loss': total_loss / max(total_samples, 1),
        'mae': total_abs_error / max(total_samples, 1),
        'mse': total_sq_error / max(total_samples, 1),
    }


# =============================================================================
# 5. Fonction principale d'entraînement supervisé
# =============================================================================

def train_energy_regressor(
    train_dataset_path: str | Path,
    val_dataset_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 64,
    learning_rate: float = 1.0e-3,
    num_epochs: int = 50,
    latent_dim: int = 8,
    encoder_hidden_dims: Tuple[int, ...] = (128, 64),
    head_hidden_dims: Tuple[int, ...] = (),
    activation: str = 'relu',
    dropout: float = 0.0,
    positive_output: bool = True,
    normalize: bool = False,
    device: Optional[str] = None,
    seed: int = 42,
    weight_decay: float = 0.0,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Entraîne un modèle supervisé `EnergyRegressor`.

    Paramètres
    ----------
    train_dataset_path, val_dataset_path : str | Path
        Chemins des fichiers .npz.
    output_dir : str | Path
        Dossier de sortie pour sauvegarder modèle, logs et historique.
    batch_size, learning_rate, num_epochs : hyperparamètres classiques.
    latent_dim, encoder_hidden_dims, head_hidden_dims : architecture du modèle.
    activation, dropout, positive_output : options de l'architecture.
    normalize : bool
        Si True, normalise les snapshots (standardisation globale fit sur le train).
    device : str ou None
        'cpu', 'cuda', etc. Si None, sélection automatique.
    seed : int
        Graine globale.
    weight_decay : float
        Weight decay de l'optimiseur Adam.
    num_workers : int
        Nombre de workers DataLoader.

    Retour
    ------
    Dict[str, Any]
        Résultats d'entraînement et chemins de sauvegarde.
    """
    set_global_seed(seed)
    output_dir = ensure_dir(output_dir)

    # Chargement des symboles métiers.
    _, _, EnergyRegressor, summarize_model, describe_dataset = load_project_symbols()

    # Device.
    if device is None:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device_obj = torch.device(device)

    # Data.
    train_loader, val_loader, train_dataset, val_dataset = build_supervised_dataloaders(
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        batch_size=batch_size,
        normalize=normalize,
        num_workers=num_workers,
    )

    input_dim = int(train_dataset.nx)

    # Model.
    model = EnergyRegressor(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        head_hidden_dims=head_hidden_dims,
        activation=activation,
        dropout=dropout,
        positive_output=positive_output,
    ).to(device_obj)

    model_summary = summarize_model(model).to_dict()

    # Optimizer + loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # History.
    history: Dict[str, List[float]] = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_mse': [],
    }

    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = Path(output_dir) / 'best_energy_regressor.pt'
    last_model_path = Path(output_dir) / 'last_energy_regressor.pt'

    print_section('Début de l\'entraînement supervisé')
    print(f'Device                 : {device_obj}')
    print(f'Input dimension        : {input_dim}')
    print(f'Batch size             : {batch_size}')
    print(f'Learning rate          : {learning_rate}')
    print(f'Number of epochs       : {num_epochs}')
    print(f'Normalize snapshots    : {normalize}')
    print(f'Model summary          : {model_summary}')

    # Training loop.
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device_obj,
        )

        val_metrics = evaluate_supervised_model(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device_obj,
        )

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_mse'].append(val_metrics['mse'])

        print(
            f"[Epoch {epoch:03d}/{num_epochs:03d}] "
            f"train_loss={train_metrics['loss']:.6e} | "
            f"val_loss={val_metrics['loss']:.6e} | "
            f"val_mae={val_metrics['mae']:.6e}"
        )

        # Save best model.
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'model_config': {
                        'input_dim': input_dim,
                        'latent_dim': latent_dim,
                        'encoder_hidden_dims': list(encoder_hidden_dims),
                        'head_hidden_dims': list(head_hidden_dims),
                        'activation': activation,
                        'dropout': dropout,
                        'positive_output': positive_output,
                    },
                    'standardizer': train_dataset.standardizer.to_dict() if train_dataset.standardizer is not None else None,
                },
                best_model_path,
            )

    # Save last model at the end.
    torch.save(
        {
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'model_config': {
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'encoder_hidden_dims': list(encoder_hidden_dims),
                'head_hidden_dims': list(head_hidden_dims),
                'activation': activation,
                'dropout': dropout,
                'positive_output': positive_output,
            },
            'standardizer': train_dataset.standardizer.to_dict() if train_dataset.standardizer is not None else None,
        },
        last_model_path,
    )

    # Save history and run metadata.
    train_summary = {
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'best_model_path': str(best_model_path),
        'last_model_path': str(last_model_path),
        'device': str(device_obj),
        'seed': int(seed),
        'batch_size': int(batch_size),
        'learning_rate': float(learning_rate),
        'num_epochs': int(num_epochs),
        'normalize': bool(normalize),
        'model_summary': model_summary,
        'train_dataset_description': describe_dataset(train_dataset_path),
        'val_dataset_description': describe_dataset(val_dataset_path),
    }

    save_json(Path(output_dir) / 'history_supervised.json', history)
    save_json(Path(output_dir) / 'training_summary_supervised.json', train_summary)

    print_section('Fin de l\'entraînement supervisé')
    print(f'Best epoch      : {best_epoch}')
    print(f'Best val loss   : {best_val_loss:.6e}')
    print(f'Best model path : {best_model_path}')
    print(f'Last model path : {last_model_path}')

    return {
        'history': history,
        'summary': train_summary,
        'best_model_path': str(best_model_path),
        'last_model_path': str(last_model_path),
    }


# =============================================================================
# 6. Fonction pratique basée sur la configuration globale du projet
# =============================================================================

def train_energy_regressor_from_config(config: Any, output_subdir: str = 'supervised') -> Dict[str, Any]:
    """
    Lance l'entraînement supervisé à partir de l'objet de configuration global.

    Paramètres
    ----------
    config : Any
        Objet de configuration compatible avec `config.py`.
    output_subdir : str
        Sous-dossier dans `config.paths.models_dir` pour cette expérience.

    Retour
    ------
    Dict[str, Any]
        Résultats de `train_energy_regressor`.
    """
    train_path = Path(config.paths.data_dir) / config.dataset.train_filename
    val_path = Path(config.paths.data_dir) / config.dataset.val_filename
    output_dir = Path(config.paths.models_dir) / output_subdir

    return train_energy_regressor(
        train_dataset_path=train_path,
        val_dataset_path=val_path,
        output_dir=output_dir,
        batch_size=int(config.ml.batch_size),
        learning_rate=float(config.ml.learning_rate),
        num_epochs=int(config.ml.num_epochs),
        latent_dim=int(config.ml.latent_dim),
        encoder_hidden_dims=(128, 64),
        head_hidden_dims=(),
        activation='relu',
        dropout=0.0,
        positive_output=True,
        normalize=bool(getattr(config.normalization, 'enabled', False)) if hasattr(config, 'normalization') else False,
        device=str(config.ml.device) if hasattr(config, 'ml') else None,
        seed=int(config.reproducibility.seed),
        weight_decay=float(getattr(config.ml, 'weight_decay', 0.0)),
        num_workers=0,
    )


# =============================================================================
# 7. Bloc de test minimal
# =============================================================================

if __name__ == '__main__':
    # Ce bloc ne lance pas un vrai entraînement (car cela nécessite les fichiers de données),
    # mais vérifie que les fonctions principales sont bien accessibles.
    print('ml/train_supervised.py chargé avec succès.')
    print('Fonctions disponibles :')
    print('- load_project_symbols')
    print('- build_supervised_dataloaders')
    print('- train_one_epoch')
    print('- evaluate_supervised_model')
    print('- train_energy_regressor')
    print('- train_energy_regressor_from_config')
