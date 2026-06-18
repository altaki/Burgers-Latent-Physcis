"""
main_visualize_latent_results.py
================================

Script principal pour générer automatiquement des figures scientifiques à partir

- des rapports JSON d'évaluation latente,
- des fichiers `.npz` de dataset,
- et, si les checkpoints existent, des sorties extraites reconstruites à la volée.

Objectif
--------
Créer rapidement des figures prêtes pour :
- le README GitHub,
- un rapport,
- ou une note de recherche,

avec une priorité sur les figures liées aux observables latentes :
- z(t) vs E(t),
- nuage z vs E,
- histogramme des violations de monotonie.

Fonctionnement
--------------
Le script essaie dans l'ordre :
1. d'utiliser les rapports JSON existants pour guider le choix du modèle,
2. de charger le module `ml_evaluate.py`,
3. de reconstruire les sorties `extracted` à partir du checkpoint et du dataset,
4. de produire les figures avec `utils/plotting.py`.

Par défaut, le script privilégie :
- d'abord le modèle de dynamique latente,
- puis l'autoencodeur latent s'il existe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import importlib.util
import json
import sys


# =============================================================================
# 1. Fallback de configuration
# =============================================================================

def _build_fallback_config() -> Dict[str, Any]:
    return {
        'paths': {
            'data_dir': 'outputs/data',
            'figures_dir': 'outputs/figures',
            'logs_dir': 'outputs/logs',
            'models_dir': 'outputs/models',
        },
        'dataset': {
            'test_filename': 'test_data.npz',
        },
        'ml': {
            'device': 'cpu',
        },
    }


# =============================================================================
# 2. Chargement robuste des modules
# =============================================================================

def _load_module_from_candidates(module_name: str, candidates: list[str]):
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



def load_config_object():
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



def load_plotting_module():
    return _load_module_from_candidates('utils_plotting', ['utils_plotting.py', 'utils/plotting.py'])



def load_evaluation_module():
    return _load_module_from_candidates('ml_evaluate', ['ml_evaluate.py', 'ml/evaluate.py'])


# =============================================================================
# 3. Utilitaires
# =============================================================================

def print_section(title: str) -> None:
    line = '=' * len(title)
    print(f"\n{title}\n{line}")



def _get_value(config: Any, dotted_key: str, default: Any = None) -> Any:
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



def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# 4. Génération de figures latentes
# =============================================================================

def main() -> Dict[str, Any]:
    config = load_config_object()
    plotting_module = load_plotting_module()
    eval_module = load_evaluation_module()

    ensure_dir = plotting_module.ensure_dir
    load_json = plotting_module.load_json
    generate_latent_evaluation_figures = plotting_module.generate_latent_evaluation_figures

    evaluate_saved_latent_dynamics_model = eval_module.evaluate_saved_latent_dynamics_model
    evaluate_saved_latent_autoencoder = eval_module.evaluate_saved_latent_autoencoder

    data_dir = Path(_get_value(config, 'paths.data_dir', 'outputs/data'))
    figures_dir = ensure_dir(Path(_get_value(config, 'paths.figures_dir', 'outputs/figures')))
    logs_dir = Path(_get_value(config, 'paths.logs_dir', 'outputs/logs'))
    models_dir = Path(_get_value(config, 'paths.models_dir', 'outputs/models'))
    dataset_path = data_dir / str(_get_value(config, 'dataset.test_filename', 'test_data.npz'))
    device = _get_value(config, 'ml.device', None)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset de test introuvable : {dataset_path}. Générez d'abord les données."
        )

    latent_dynamics_ckpt = models_dir / 'latent_dynamics' / 'best_latent_dynamics_model.pt'
    latent_autoencoder_ckpt = models_dir / 'latent_autoencoder' / 'best_latent_autoencoder.pt'

    summary: Dict[str, Any] = {
        'dataset_path': str(dataset_path),
        'generated_figures': {},
    }

    # ------------------------------------------------------------------
    # 1) Figures pour le modèle de dynamique latente (prioritaire)
    # ------------------------------------------------------------------
    if latent_dynamics_ckpt.exists():
        print_section('Génération des figures : dynamique latente')
        results = evaluate_saved_latent_dynamics_model(
            checkpoint_path=latent_dynamics_ckpt,
            dataset_path=dataset_path,
            batch_size=128,
            stride=None,
            device=device,
            output_json_path=None,
        )
        figs = generate_latent_evaluation_figures(
            extracted={
                'scalar': results['extracted'].get('z_t'),
                'energy': results['extracted'].get('energy_t'),
                'trajectory_index': results['extracted'].get('trajectory_index'),
                'time_index': results['extracted'].get('time_index_t'),
            },
            output_dir=figures_dir,
            prefix='latent_dynamics',
            trajectory_index=0,
        )
        summary['generated_figures']['latent_dynamics'] = figs
        for key, value in figs.items():
            print(f'- {key}: {value}')
    else:
        print_section('Aucun checkpoint de dynamique latente trouvé')

    # ------------------------------------------------------------------
    # 2) Figures pour l'autoencodeur latent
    # ------------------------------------------------------------------
    if latent_autoencoder_ckpt.exists():
        print_section('Génération des figures : autoencodeur latent')
        results = evaluate_saved_latent_autoencoder(
            checkpoint_path=latent_autoencoder_ckpt,
            dataset_path=dataset_path,
            batch_size=128,
            device=device,
            output_json_path=None,
        )
        figs = generate_latent_evaluation_figures(
            extracted={
                'scalar': results['extracted'].get('scalar'),
                'energy': results['extracted'].get('energy'),
                'trajectory_index': results['extracted'].get('trajectory_index'),
                'time_index': results['extracted'].get('time_index'),
            },
            output_dir=figures_dir,
            prefix='latent_autoencoder',
            trajectory_index=0,
        )
        summary['generated_figures']['latent_autoencoder'] = figs
        for key, value in figs.items():
            print(f'- {key}: {value}')
    else:
        print_section('Aucun checkpoint d\'autoencodeur latent trouvé')

    # ------------------------------------------------------------------
    # 3) Sauvegarde du résumé
    # ------------------------------------------------------------------
    summary_path = logs_dir / 'main_visualize_latent_results_summary.json'
    save_json(summary_path, summary)

    print_section('Terminé')
    print(f'Résumé sauvegardé dans : {summary_path}')
    print('Les figures latentes sont prêtes à être ajoutées au README ou au rapport.')

    return summary


if __name__ == '__main__':
    main()
