"""
main_train_latent.py
====================

Script principal pour lancer l'entraînement latent dans le projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce script sert de point d'entrée simple pour :

1. charger la configuration globale du projet,
2. localiser les datasets train / validation,
3. lancer soit :
   - l'entraînement d'un autoencodeur latent,
   - l'entraînement d'un modèle de dynamique latente,
4. sauvegarder les résultats dans le dossier des modèles,
5. produire un petit résumé JSON facilement exploitable.

Philosophie
-----------
- script exécutable directement,
- lisible et abondamment commenté,
- robuste aux imports dans le contexte actuel du prototype,
- compatible à la fois avec les fichiers racine (`ml_train_latent.py`) et
  avec la structure de package (`ml/train_latent.py`).

Usage recommandé
----------------
Depuis la racine du projet :

    python main_train_latent.py

Pré-requis
----------
Avant de lancer ce script, il est recommandé d'avoir déjà généré les données via :

    python main_generate_data.py

Mode par défaut
---------------
Par défaut, ce script lance l'entraînement du **modèle de dynamique latente**,
qui est le plus intéressant scientifiquement pour l'étude des observables
potentiellement monotones. Le mode peut être changé en modifiant la variable
`training_mode` dans la fonction `main()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import importlib.util
import json
import random
import sys

import numpy as np


# =============================================================================
# 1. Petit fallback local pour la configuration
# =============================================================================

def _build_fallback_config() -> Dict[str, Any]:
    """
    Construit un dictionnaire de configuration minimal compatible avec le
    pipeline latent si `config.py` n'est pas disponible.
    """
    return {
        'project_name': 'burgers_latent_physics',
        'experiment_name': 'baseline_v1',
        'paths': {
            'output_root': 'outputs',
            'data_dir': 'outputs/data',
            'models_dir': 'outputs/models',
            'figures_dir': 'outputs/figures',
            'logs_dir': 'outputs/logs',
        },
        'dataset': {
            'train_filename': 'train_data.npz',
            'val_filename': 'val_data.npz',
        },
        'ml': {
            'batch_size': 64,
            'learning_rate': 1.0e-3,
            'num_epochs': 100,
            'latent_dim': 8,
            'weight_decay': 0.0,
            'device': 'cpu',
        },
        'normalization': {
            'enabled': False,
        },
        'losses': {
            'reconstruction_weight': 1.0,
            'prediction_weight': 1.0,
            'monotonicity_weight': 0.1,
            'variance_weight': 0.01,
            'min_latent_variance': 1.0e-3,
            'monotonicity_mode': 'squared_hinge',
        },
        'reproducibility': {
            'seed': 42,
        },
    }


# =============================================================================
# 2. Chargement robuste des modules du projet
# =============================================================================

def _load_module_from_candidates(module_name: str, candidates: list[str]):
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
        f"Impossible de charger le module {module_name!r}. "
        f"Candidats testés : {candidates}"
    )



def load_config_object():
    """
    Charge l'objet de configuration global.

    Priorité :
    1. import direct depuis `config.py`,
    2. chargement par chemin local `config.py`,
    3. fallback minimal sous forme de dictionnaire.
    """
    try:
        from config import get_default_config  # type: ignore
        return get_default_config(create_dirs=True)
    except ModuleNotFoundError:
        pass

    config_path = Path('config.py')
    if config_path.exists():
        spec = importlib.util.spec_from_file_location('config', str(config_path))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules['config'] = module
        spec.loader.exec_module(module)
        if hasattr(module, 'get_default_config'):
            return module.get_default_config(create_dirs=True)

    cfg = _build_fallback_config()
    for folder in cfg['paths'].values():
        Path(folder).mkdir(parents=True, exist_ok=True)
    return cfg



def load_training_module():
    """
    Charge le module d'entraînement latent.

    On accepte à la fois :
    - ml_train_latent.py
    - ml/train_latent.py
    """
    return _load_module_from_candidates(
        'ml_train_latent',
        ['ml_train_latent.py', 'ml/train_latent.py'],
    )


# =============================================================================
# 3. Utilitaires de confort
# =============================================================================

def print_section(title: str) -> None:
    """Affiche un titre lisible dans le terminal."""
    line = '=' * len(title)
    print(f"\n{title}\n{line}")



def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Sauvegarde un dictionnaire dans un fichier JSON lisible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def set_global_seed(seed: int) -> None:
    """Fixe les graines principales pour la reproductibilité."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass



def _get_value(config: Any, dotted_key: str, default: Any = None) -> Any:
    """
    Récupère une valeur depuis :
    - un objet à attributs (config.py dataclasses),
    - ou un dictionnaire fallback.
    """
    parts = dotted_key.split('.')
    current = config
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


# =============================================================================
# 4. Fonction principale
# =============================================================================

def main() -> Dict[str, Any]:
    """
    Point d'entrée principal du script.

    Retour
    ------
    Dict[str, Any]
        Résumé de l'exécution.
    """
    print_section('Chargement de la configuration')
    config = load_config_object()

    project_name = _get_value(config, 'project_name', 'burgers_latent_physics')
    experiment_name = _get_value(config, 'experiment_name', 'baseline_v1')
    data_dir = Path(_get_value(config, 'paths.data_dir', 'outputs/data'))
    models_dir = Path(_get_value(config, 'paths.models_dir', 'outputs/models'))
    logs_dir = Path(_get_value(config, 'paths.logs_dir', 'outputs/logs'))

    train_filename = _get_value(config, 'dataset.train_filename', 'train_data.npz')
    val_filename = _get_value(config, 'dataset.val_filename', 'val_data.npz')

    batch_size = int(_get_value(config, 'ml.batch_size', 64))
    learning_rate = float(_get_value(config, 'ml.learning_rate', 1.0e-3))
    num_epochs = int(_get_value(config, 'ml.num_epochs', 100))
    latent_dim = int(_get_value(config, 'ml.latent_dim', 8))
    weight_decay = float(_get_value(config, 'ml.weight_decay', 0.0))
    device = _get_value(config, 'ml.device', None)
    normalize = bool(_get_value(config, 'normalization.enabled', False))
    seed = int(_get_value(config, 'reproducibility.seed', 42))

    reconstruction_weight = float(_get_value(config, 'losses.reconstruction_weight', 1.0))
    prediction_weight = float(_get_value(config, 'losses.prediction_weight', 1.0))
    monotonicity_weight = float(_get_value(config, 'losses.monotonicity_weight', 0.0))
    variance_weight = float(_get_value(config, 'losses.variance_weight', 0.0))
    min_latent_variance = float(_get_value(config, 'losses.min_latent_variance', 1.0e-3))
    monotonicity_mode = str(_get_value(config, 'losses.monotonicity_mode', 'squared_hinge'))

    print(f'- Projet               : {project_name}')
    print(f'- Expérience           : {experiment_name}')
    print(f'- Dossier data         : {data_dir}')
    print(f'- Dossier models       : {models_dir}')
    print(f'- Dossier logs         : {logs_dir}')

    print_section('Initialisation de la reproductibilité')
    set_global_seed(seed)
    print(f'- Seed globale         : {seed}')

    train_path = data_dir / str(train_filename)
    val_path = data_dir / str(val_filename)

    print_section('Vérification des fichiers de données')
    print(f'- Train dataset path   : {train_path}')
    print(f'- Val dataset path     : {val_path}')

    if not train_path.exists():
        raise FileNotFoundError(
            f"Le fichier train est introuvable : {train_path}. "
            "Générez d'abord les données avec `main_generate_data.py`."
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"Le fichier validation est introuvable : {val_path}. "
            "Générez d'abord les données avec `main_generate_data.py`."
        )

    print_section('Chargement du module d\'entraînement latent')
    training_module = load_training_module()

    # -------------------------------------------------------------------------
    # Choix du mode d'entraînement latent.
    # -------------------------------------------------------------------------
    # Options disponibles :
    # - 'latent_dynamics'   (recommandé par défaut)
    # - 'latent_autoencoder'
    training_mode = 'latent_autoencoder'  # <-- Changez ici pour le mode souhaité

    print_section('Mode sélectionné')
    print(f'- Training mode        : {training_mode}')

    if training_mode == 'latent_dynamics':
        train_function = training_module.train_latent_dynamics_model
        output_dir = models_dir / 'latent_dynamics'
        output_dir.mkdir(parents=True, exist_ok=True)
        stride = 1

        print_section('Paramètres d\'entraînement latent (dynamique)')
        print(f'- Batch size           : {batch_size}')
        print(f'- Learning rate        : {learning_rate}')
        print(f'- Num epochs           : {num_epochs}')
        print(f'- Latent dim           : {latent_dim}')
        print(f'- Weight decay         : {weight_decay}')
        print(f'- Normalize snapshots  : {normalize}')
        print(f'- Device               : {device}')
        print(f'- Stride               : {stride}')
        print(f'- Prediction weight    : {prediction_weight}')
        print(f'- Monotonicity weight  : {monotonicity_weight}')
        print(f'- Variance weight      : {variance_weight}')
        print(f'- Min latent variance  : {min_latent_variance}')
        print(f'- Monotonicity mode    : {monotonicity_mode}')
        print(f'- Output dir           : {output_dir}')

        print_section('Lancement de l\'entraînement latent (dynamique)')
        results = train_function(
            train_dataset_path=train_path,
            val_dataset_path=val_path,
            output_dir=output_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            latent_dim=latent_dim,
            encoder_hidden_dims=(128, 64),
            predictor_hidden_dims=(64, 64),
            activation='relu',
            dropout=0.0,
            use_scalar_head=True,
            scalar_head_hidden_dims=(),
            positive_scalar_output=False,
            residual_predictor=False,
            prediction_weight=prediction_weight,
            monotonicity_weight=monotonicity_weight,
            variance_weight=variance_weight,
            prediction_mode='mse',
            monotonicity_mode=monotonicity_mode,
            min_scalar_variance=min_latent_variance,
            normalize=normalize,
            stride=stride,
            device=device,
            seed=seed,
            weight_decay=weight_decay,
            num_workers=0,
        )

    elif training_mode == 'latent_autoencoder':
        train_function = training_module.train_latent_autoencoder
        output_dir = models_dir / 'latent_autoencoder'
        output_dir.mkdir(parents=True, exist_ok=True)

        print_section('Paramètres d\'entraînement latent (autoencodeur)')
        print(f'- Batch size           : {batch_size}')
        print(f'- Learning rate        : {learning_rate}')
        print(f'- Num epochs           : {num_epochs}')
        print(f'- Latent dim           : {latent_dim}')
        print(f'- Weight decay         : {weight_decay}')
        print(f'- Normalize snapshots  : {normalize}')
        print(f'- Device               : {device}')
        print(f'- Reconstruction weight: {reconstruction_weight}')
        print(f'- Variance weight      : {variance_weight}')
        print(f'- Min latent variance  : {min_latent_variance}')
        print(f'- Output dir           : {output_dir}')

        print_section('Lancement de l\'entraînement latent (autoencodeur)')
        results = train_function(
            train_dataset_path=train_path,
            val_dataset_path=val_path,
            output_dir=output_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            latent_dim=latent_dim,
            encoder_hidden_dims=(128, 64),
            decoder_hidden_dims=(64, 128),
            activation='relu',
            dropout=0.0,
            use_scalar_head=True,
            scalar_head_hidden_dims=(),
            positive_scalar_output=False,
            reconstruction_weight=reconstruction_weight,
            variance_weight=variance_weight,
            min_scalar_variance=min_latent_variance,
            normalize=normalize,
            device=device,
            seed=seed,
            weight_decay=weight_decay,
            num_workers=0,
        )

    else:
        raise ValueError(
            f"Mode d'entraînement latent inconnu : {training_mode!r}. "
            "Choisir 'latent_dynamics' ou 'latent_autoencoder'."
        )

    print_section('Sauvegarde du résumé principal')
    summary_path = logs_dir / 'main_train_latent_summary.json'
    summary = {
        'project_name': project_name,
        'experiment_name': experiment_name,
        'training_mode': training_mode,
        'train_dataset_path': str(train_path),
        'val_dataset_path': str(val_path),
        'results': results,
    }
    save_json(summary_path, summary)
    print(f'- Résumé sauvegardé    : {summary_path}')

    print_section('Terminé')
    print('L\'entraînement latent est terminé.')
    print('Étapes suivantes possibles :')
    print('- évaluer le modèle avec ml/evaluate.py')
    print('- comparer z_t à l\'énergie E(t)')
    print('- analyser la monotonie trajectoire par trajectoire')

    return summary


# =============================================================================
# 5. Exécution directe
# =============================================================================

if __name__ == '__main__':
    main()
