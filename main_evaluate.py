"""
main_evaluate.py
================

Script principal pour lancer l'évaluation des modèles dans le projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce script sert de point d'entrée simple pour :

1. charger la configuration globale du projet,
2. localiser le dataset d'évaluation (par défaut : split test),
3. charger les checkpoints des modèles entraînés,
4. lancer l'évaluation du modèle supervisé et/ou latent,
5. sauvegarder des rapports JSON synthétiques dans le dossier des logs.

Philosophie
-----------
- script exécutable directement,
- lisible et abondamment commenté,
- robuste aux imports dans le contexte actuel du prototype,
- compatible à la fois avec les fichiers racine (`ml_evaluate.py`) et
  avec la structure de package (`ml/evaluate.py`).

Usage recommandé
----------------
Depuis la racine du projet :

    python main_evaluate.py

Pré-requis
----------
Avant de lancer ce script, il est recommandé d'avoir déjà :

    python main_generate_data.py
    python main_train_supervised.py
    python main_train_latent.py

Mode par défaut
---------------
Le script essaie d'évaluer :
- le modèle supervisé si le checkpoint existe,
- le modèle latent de dynamique si le checkpoint existe,
- et, à défaut, l'autoencodeur latent si ce checkpoint existe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import importlib.util
import json
import sys


# =============================================================================
# 1. Petit fallback local pour la configuration
# =============================================================================

def _build_fallback_config() -> Dict[str, Any]:
    """
    Construit un dictionnaire de configuration minimal compatible avec
    le pipeline d'évaluation si `config.py` n'est pas disponible.
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
            'test_filename': 'test_data.npz',
        },
        'ml': {
            'device': 'cpu',
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



def load_evaluation_module():
    """
    Charge le module d'évaluation.

    On accepte à la fois :
    - ml_evaluate.py
    - ml/evaluate.py
    """
    return _load_module_from_candidates(
        'ml_evaluate',
        ['ml_evaluate.py', 'ml/evaluate.py'],
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



def _get_value(config: Any, dotted_key: str, default: Any = None) -> Any:
    """
    Récupère une valeur depuis :
    - un objet à attributs,
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
        Résumé global de l'évaluation.
    """
    print_section('Chargement de la configuration')
    config = load_config_object()

    project_name = _get_value(config, 'project_name', 'burgers_latent_physics')
    experiment_name = _get_value(config, 'experiment_name', 'baseline_v1')
    data_dir = Path(_get_value(config, 'paths.data_dir', 'outputs/data'))
    models_dir = Path(_get_value(config, 'paths.models_dir', 'outputs/models'))
    logs_dir = Path(_get_value(config, 'paths.logs_dir', 'outputs/logs'))
    device = _get_value(config, 'ml.device', None)

    test_filename = _get_value(config, 'dataset.test_filename', 'test_data.npz')
    test_path = data_dir / str(test_filename)

    print(f'- Projet               : {project_name}')
    print(f'- Expérience           : {experiment_name}')
    print(f'- Dossier data         : {data_dir}')
    print(f'- Dossier models       : {models_dir}')
    print(f'- Dossier logs         : {logs_dir}')
    print(f'- Device               : {device}')

    print_section('Vérification du dataset de test')
    print(f'- Test dataset path    : {test_path}')
    if not test_path.exists():
        raise FileNotFoundError(
            f"Le fichier de test est introuvable : {test_path}. "
            "Générez d'abord les données avec `main_generate_data.py`."
        )

    print_section('Chargement du module d\'évaluation')
    eval_module = load_evaluation_module()

    evaluate_saved_supervised_model = eval_module.evaluate_saved_supervised_model
    evaluate_saved_latent_autoencoder = eval_module.evaluate_saved_latent_autoencoder
    evaluate_saved_latent_dynamics_model = eval_module.evaluate_saved_latent_dynamics_model

    # Checkpoints attendus
    supervised_ckpt = models_dir / 'supervised' / 'best_energy_regressor.pt'
    latent_dynamics_ckpt = models_dir / 'latent_dynamics' / 'best_latent_dynamics_model.pt'
    latent_autoencoder_ckpt = models_dir / 'latent_autoencoder' / 'best_latent_autoencoder.pt'

    print_section('Recherche des checkpoints')
    print(f'- Supervised checkpoint        : {supervised_ckpt}')
    print(f'- Latent dynamics checkpoint   : {latent_dynamics_ckpt}')
    print(f'- Latent autoencoder checkpoint: {latent_autoencoder_ckpt}')

    summary: Dict[str, Any] = {
        'project_name': project_name,
        'experiment_name': experiment_name,
        'dataset_path': str(test_path),
        'evaluations': {},
    }

    # -------------------------------------------------------------------------
    # 1) Évaluation supervisée
    # -------------------------------------------------------------------------
    if supervised_ckpt.exists():
        print_section('Évaluation du modèle supervisé')
        supervised_report_path = logs_dir / 'evaluation_supervised_test.json'
        supervised_results = evaluate_saved_supervised_model(
            checkpoint_path=supervised_ckpt,
            dataset_path=test_path,
            batch_size=128,
            device=device,
            output_json_path=supervised_report_path,
        )
        summary['evaluations']['supervised'] = supervised_results['report']
        print(f"- Rapport sauvegardé : {supervised_report_path}")
    else:
        print('\n[Info] Aucun checkpoint supervisé trouvé, évaluation supervisée ignorée.')

    # -------------------------------------------------------------------------
    # 2) Évaluation dynamique latente (priorité)
    # -------------------------------------------------------------------------
    if latent_dynamics_ckpt.exists():
        print_section('Évaluation du modèle de dynamique latente')
        latent_dyn_report_path = logs_dir / 'evaluation_latent_dynamics_test.json'
        latent_dyn_results = evaluate_saved_latent_dynamics_model(
            checkpoint_path=latent_dynamics_ckpt,
            dataset_path=test_path,
            batch_size=128,
            stride=None,
            device=device,
            output_json_path=latent_dyn_report_path,
        )
        summary['evaluations']['latent_dynamics'] = latent_dyn_results['report']
        print(f"- Rapport sauvegardé : {latent_dyn_report_path}")
    else:
        print('\n[Info] Aucun checkpoint de dynamique latente trouvé, évaluation correspondante ignorée.')

    # -------------------------------------------------------------------------
    # 3) Évaluation autoencodeur latent (optionnelle)
    # -------------------------------------------------------------------------
    if latent_autoencoder_ckpt.exists():
        print_section('Évaluation de l\'autoencodeur latent')
        latent_ae_report_path = logs_dir / 'evaluation_latent_autoencoder_test.json'
        latent_ae_results = evaluate_saved_latent_autoencoder(
            checkpoint_path=latent_autoencoder_ckpt,
            dataset_path=test_path,
            batch_size=128,
            device=device,
            output_json_path=latent_ae_report_path,
        )
        summary['evaluations']['latent_autoencoder'] = latent_ae_results['report']
        print(f"- Rapport sauvegardé : {latent_ae_report_path}")
    else:
        print('\n[Info] Aucun checkpoint d\'autoencodeur latent trouvé, évaluation correspondante ignorée.')

    print_section('Sauvegarde du résumé global')
    summary_path = logs_dir / 'main_evaluate_summary.json'
    save_json(summary_path, summary)
    print(f'- Résumé sauvegardé    : {summary_path}')

    print_section('Terminé')
    print('L\'évaluation est terminée.')
    print('Étapes suivantes possibles :')
    print('- analyser les rapports JSON')
    print('- tracer les courbes z(t) et E(t)')
    print('- comparer supervisé vs latent')

    return summary


# =============================================================================
# 5. Exécution directe
# =============================================================================

if __name__ == '__main__':
    main()
