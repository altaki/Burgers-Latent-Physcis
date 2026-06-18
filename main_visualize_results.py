"""
main_visualize_results.py
=========================

Script principal pour générer automatiquement des figures à partir :
- des datasets `.npz`,
- des historiques d'entraînement JSON,
- et, si souhaité, des sorties d'évaluation déjà calculées.

Objectif
--------
Créer rapidement des figures prêtes pour :
- le README GitHub,
- un rapport,
- ou une note de recherche.

Figures générées par défaut
---------------------------
1. figures PDE de référence à partir du split test,
2. courbes d'entraînement supervisé et latent si les JSON existent.

Remarque
--------
Pour les figures latentes basées sur les sorties `extracted`, il est plus simple
soit d'enrichir ultérieurement `main_evaluate.py`, soit d'utiliser directement
les fonctions de `utils/plotting.py` dans un notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import importlib.util
import json
import sys


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

    raise ModuleNotFoundError(f"Impossible de charger le module {module_name!r}. Candidats testés : {candidates}")



def _build_fallback_config() -> Dict[str, Any]:
    return {
        'paths': {
            'data_dir': 'outputs/data',
            'figures_dir': 'outputs/figures',
            'logs_dir': 'outputs/logs',
        },
        'dataset': {
            'test_filename': 'test_data.npz',
        },
    }



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



def print_section(title: str) -> None:
    line = '=' * len(title)
    print(f"\n{title}\n{line}")



def main() -> Dict[str, Any]:
    config = load_config_object()
    plotting_module = _load_module_from_candidates('utils_plotting', ['utils_plotting.py', 'utils/plotting.py'])

    generate_reference_dataset_figures = plotting_module.generate_reference_dataset_figures
    plot_training_history = plotting_module.plot_training_history
    ensure_dir = plotting_module.ensure_dir

    data_dir = Path(_get_value(config, 'paths.data_dir', 'outputs/data'))
    figures_dir = ensure_dir(Path(_get_value(config, 'paths.figures_dir', 'outputs/figures')))
    logs_dir = Path(_get_value(config, 'paths.logs_dir', 'outputs/logs'))
    test_filename = _get_value(config, 'dataset.test_filename', 'test_data.npz')
    test_path = data_dir / str(test_filename)

    print_section('Génération des figures PDE de référence')
    generated = {}
    if test_path.exists():
        generated['reference_dataset_figures'] = generate_reference_dataset_figures(
            dataset_path=test_path,
            output_dir=figures_dir,
            trajectory_index=0,
        )
        print(f"Figures PDE sauvegardées dans : {figures_dir}")
    else:
        print(f"Dataset de test introuvable : {test_path}")

    print_section('Génération des courbes d\'entraînement')
    curve_specs = [
        ('history_supervised.json', 'Historique entraînement supervisé', 'training_supervised.png'),
        ('history_latent_autoencoder.json', 'Historique autoencodeur latent', 'training_latent_autoencoder.png'),
        ('history_latent_dynamics.json', 'Historique dynamique latente', 'training_latent_dynamics.png'),
    ]

    generated['training_curves'] = {}
    for json_name, title, fig_name in curve_specs:
        json_path = logs_dir / json_name
        fig_path = figures_dir / fig_name
        if json_path.exists():
            try:
                plot_training_history(json_path, title, fig_path)
                generated['training_curves'][json_name] = str(fig_path)
                print(f"Courbe générée : {fig_path}")
            except Exception as exc:
                print(f"Impossible de tracer {json_path}: {exc}")
        else:
            print(f"Historique absent : {json_path}")

    print_section('Terminé')
    print('Les figures disponibles peuvent maintenant être ajoutées au README GitHub.')
    return generated


if __name__ == '__main__':
    main()
