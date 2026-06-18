"""
main_generate_data.py
=====================

Script principal pour générer les datasets train / val / test du projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce script sert de point d'entrée simple pour :

1. charger la configuration globale du projet,
2. initialiser la reproductibilité,
3. générer les splits train / validation / test,
4. sauvegarder les fichiers de données au format .npz,
5. afficher un résumé et quelques vérifications de cohérence.

Philosophie
-----------
- script clair, lisible et abondamment commenté,
- peu d'hypothèses implicites,
- compatible avec le prototype déjà construit,
- robuste même si certains modules ne sont pas encore importables
  comme un package Python complet.

Fonctionnement pratique
-----------------------
Le script essaie d'abord d'utiliser `config.py` si ce fichier est disponible.
Si ce n'est pas le cas, il utilise une configuration locale de secours, avec
les mêmes paramètres par défaut que ceux définis précédemment dans le projet.

Même logique pour le builder de dataset :
- priorité à `pde_dataset_builder.py` si disponible,
- fallback sur `pde/dataset_builder.py` si le package `pde` est utilisable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field, asdict
import importlib.util
import json
import random
import sys

import numpy as np


# =============================================================================
# 1. Fallback local pour la configuration
# =============================================================================

@dataclass
class _PathsConfig:
    output_root: Path = Path("outputs")
    data_dir: Path = field(default_factory=lambda: Path("outputs") / "data")
    models_dir: Path = field(default_factory=lambda: Path("outputs") / "models")
    figures_dir: Path = field(default_factory=lambda: Path("outputs") / "figures")
    logs_dir: Path = field(default_factory=lambda: Path("outputs") / "logs")

    def create_directories(self) -> None:
        for path in [self.output_root, self.data_dir, self.models_dir, self.figures_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class _PDEConfig:
    L: float = 2.0 * np.pi
    Nx: int = 128
    nu: float = 0.05
    T: float = 2.0
    dt: float = 1.0e-3
    save_every: int = 20
    boundary_condition: str = "periodic"
    spatial_scheme: str = "finite_difference"
    time_scheme: str = "rk4"


@dataclass
class _DatasetConfig:
    n_train: int = 500
    n_val: int = 100
    n_test: int = 100
    ic_types: list[str] = field(default_factory=lambda: ["fourier", "gaussian_bumps", "random_smooth"])
    max_fourier_mode: int = 5
    fourier_amplitude_scale: float = 1.0
    n_gaussian_bumps: int = 2
    spectral_decay: float = 2.0
    train_filename: str = "train_data.npz"
    val_filename: str = "val_data.npz"
    test_filename: str = "test_data.npz"


@dataclass
class _ReproConfig:
    seed: int = 42
    deterministic_torch: bool = False


@dataclass
class _ProjectConfig:
    project_name: str = "burgers_latent_physics"
    experiment_name: str = "baseline_v1"
    paths: _PathsConfig = field(default_factory=_PathsConfig)
    pde: _PDEConfig = field(default_factory=_PDEConfig)
    dataset: _DatasetConfig = field(default_factory=_DatasetConfig)
    reproducibility: _ReproConfig = field(default_factory=_ReproConfig)

    def create_directories(self) -> None:
        self.paths.create_directories()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        def convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return convert(data)


# =============================================================================
# 2. Chargement robuste de la configuration
# =============================================================================

def _load_config_function():
    """
    Essaie de récupérer `get_default_config` depuis `config.py`.

    Stratégie :
    1. import Python classique si possible,
    2. chargement via chemin local `config.py` si le fichier existe,
    3. fallback sur une configuration locale intégrée.
    """
    # 1) Import classique
    try:
        from config import get_default_config  # type: ignore
        return get_default_config
    except ModuleNotFoundError:
        pass

    # 2) Chargement depuis un fichier local si présent.
    config_path = Path("config.py")
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("config", str(config_path))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        if hasattr(module, "get_default_config"):
            return module.get_default_config

    # 3) Fallback local.
    def _fallback_get_default_config(create_dirs: bool = True) -> _ProjectConfig:
        cfg = _ProjectConfig()
        if create_dirs:
            cfg.create_directories()
        return cfg

    return _fallback_get_default_config


get_default_config = _load_config_function()


# =============================================================================
# 3. Chargement robuste du builder de dataset
# =============================================================================

def _load_dataset_builder_symbols():
    """
    Charge les fonctions principales du builder de dataset.

    Ordre de priorité :
    1. import de `pde_dataset_builder`
    2. chargement par chemin `pde_dataset_builder.py`
    3. import de `pde.dataset_builder`
    4. chargement par chemin `pde/dataset_builder.py`
    """
    symbol_names = [
        "build_and_save_split_from_config",
        "summarize_dataset",
        "basic_dataset_sanity_checks",
    ]

    # 1) Import classique du fichier racine autoporté
    try:
        import pde_dataset_builder as module  # type: ignore
        return tuple(getattr(module, name) for name in symbol_names)
    except ModuleNotFoundError:
        pass

    # 2) Chargement via chemin local
    root_builder = Path("pde_dataset_builder.py")
    if root_builder.exists():
        spec = importlib.util.spec_from_file_location("pde_dataset_builder", str(root_builder))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return tuple(getattr(module, name) for name in symbol_names)

    # 3) Import package
    try:
        from pde.dataset_builder import (  # type: ignore
            build_and_save_split_from_config,
            summarize_dataset,
            basic_dataset_sanity_checks,
        )
        return build_and_save_split_from_config, summarize_dataset, basic_dataset_sanity_checks
    except ModuleNotFoundError:
        pass

    # 4) Chargement via chemin package
    pkg_builder = Path("pde") / "dataset_builder.py"
    if pkg_builder.exists():
        spec = importlib.util.spec_from_file_location("pde.dataset_builder", str(pkg_builder))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return tuple(getattr(module, name) for name in symbol_names)

    raise ModuleNotFoundError(
        "Impossible de charger le builder de dataset. Vérifiez la présence de "
        "`pde_dataset_builder.py` ou `pde/dataset_builder.py`."
    )


(
    build_and_save_split_from_config,
    summarize_dataset,
    basic_dataset_sanity_checks,
) = _load_dataset_builder_symbols()


# =============================================================================
# 4. Reproductibilité
# =============================================================================

def set_global_seed(seed: int) -> None:
    """
    Fixe les graines aléatoires principales pour la reproductibilité.
    """
    np.random.seed(seed)
    random.seed(seed)

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


# =============================================================================
# 5. Utilitaires de logging
# =============================================================================

def print_section(title: str) -> None:
    """Affiche un titre de section lisible dans le terminal."""
    line = "=" * len(title)
    print(f"\n{title}\n{line}")



def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Sauvegarde un dictionnaire dans un fichier JSON lisible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# 6. Génération d'un split avec résumé
# =============================================================================

def generate_and_report_split(config: Any, split: str, seed_offset: int = 0) -> Dict[str, Any]:
    """
    Génère un split, le sauvegarde, puis affiche un résumé et des sanity checks.
    """
    base_seed = int(config.reproducibility.seed)
    rng = np.random.default_rng(base_seed + seed_offset)

    print_section(f"Génération du split : {split}")

    dataset = build_and_save_split_from_config(
        config=config,
        split=split,
        rng=rng,
        verbose=True,
    )

    summary = summarize_dataset(dataset)
    checks = basic_dataset_sanity_checks(dataset)

    print("\nRésumé :")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    print("\nSanity checks :")
    for key, value in checks.items():
        print(f"- {key}: {value}")

    report = {
        "split": split,
        "summary": summary,
        "sanity_checks": checks,
    }
    report_path = Path(config.paths.logs_dir) / f"data_generation_report_{split}.json"
    save_json(report_path, report)
    print(f"\nRapport sauvegardé dans : {report_path}")

    return dataset


# =============================================================================
# 7. Point d'entrée principal
# =============================================================================

def main() -> None:
    """Point d'entrée principal du script."""
    print_section("Chargement de la configuration")
    config = get_default_config(create_dirs=True)

    print("Configuration chargée avec succès.")
    print(f"- Projet              : {config.project_name}")
    print(f"- Expérience          : {config.experiment_name}")
    print(f"- Dossier de données  : {config.paths.data_dir}")
    print(f"- Dossier de logs     : {config.paths.logs_dir}")

    print_section("Initialisation de la reproductibilité")
    seed = int(config.reproducibility.seed)
    set_global_seed(seed)
    print(f"Seed globale fixée à : {seed}")

    print_section("Paramètres principaux de génération")
    print(f"- Domaine L                 : {config.pde.L}")
    print(f"- Nombre de points Nx       : {config.pde.Nx}")
    print(f"- Viscosité nu              : {config.pde.nu}")
    print(f"- Temps final T             : {config.pde.T}")
    print(f"- Pas de temps dt           : {config.pde.dt}")
    print(f"- Sauvegarde tous les       : {config.pde.save_every} pas")
    print(f"- Schéma temporel           : {config.pde.time_scheme}")
    print(f"- CI autorisées             : {list(config.dataset.ic_types)}")
    print(
        f"- n_train / n_val / n_test  : "
        f"{config.dataset.n_train} / {config.dataset.n_val} / {config.dataset.n_test}"
    )

    print_section("Sauvegarde de la configuration")
    config_json_path = Path(config.paths.logs_dir) / "config_snapshot.json"
    save_json(config_json_path, config.to_dict())
    print(f"Configuration sauvegardée dans : {config_json_path}")

    # Génération des trois splits
    generate_and_report_split(config, split="train", seed_offset=0)
    generate_and_report_split(config, split="val", seed_offset=1)
    generate_and_report_split(config, split="test", seed_offset=2)

    print_section("Terminé")
    print("La génération des datasets train / val / test est terminée.")
    print("Vous pouvez maintenant passer à l'étape suivante :")
    print("- visualisation des trajectoires,")
    print("- chargement des données dans PyTorch,")
    print("- entraînement des modèles supervisés ou latents.")


# =============================================================================
# 8. Exécution directe
# =============================================================================

if __name__ == "__main__":
    main()
