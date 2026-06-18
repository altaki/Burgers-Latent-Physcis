"""
config.py
==========

Fichier de configuration central du projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce module regroupe tous les paramètres importants du projet afin d'éviter
la dispersion des constantes dans plusieurs fichiers. L'idée est d'avoir
un point d'entrée unique pour :

1. la simulation de l'équation de Burgers,
2. la génération des données,
3. l'entraînement des modèles de machine learning,
4. la reproductibilité,
5. l'organisation des dossiers de sortie.

Philosophie
-----------
- Code lisible et abondamment commenté.
- Paramètres regroupés par blocs logiques.
- Utilisation de dataclasses pour un accès clair via la notation pointée.
- Fonction utilitaire pour créer automatiquement les dossiers de sortie.

Exemple d'utilisation
---------------------
>>> from config import get_default_config
>>> cfg = get_default_config()
>>> print(cfg.pde.nu)
>>> print(cfg.paths.data_dir)

Remarque
--------
Cette première version est volontairement simple et stable.
Elle est très adaptée à un prototype de recherche.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List


# -----------------------------------------------------------------------------
# 1. Configuration des chemins du projet
# -----------------------------------------------------------------------------

@dataclass
class PathsConfig:
    """
    Regroupe tous les chemins utiles au projet.

    Tous les dossiers sont construits à partir d'un dossier racine `output_root`.
    Cela facilite l'organisation du projet et évite d'écrire des chemins en dur
    dans les scripts.
    """

    # Dossier racine dans lequel seront enregistrés les résultats.
    output_root: Path = Path("outputs")

    # Sous-dossiers principaux.
    data_dir: Path = field(default_factory=lambda: Path("outputs") / "data")
    models_dir: Path = field(default_factory=lambda: Path("outputs") / "models")
    figures_dir: Path = field(default_factory=lambda: Path("outputs") / "figures")
    logs_dir: Path = field(default_factory=lambda: Path("outputs") / "logs")

    def create_directories(self) -> None:
        """
        Crée tous les dossiers nécessaires s'ils n'existent pas déjà.
        """
        for path in [self.output_root, self.data_dir, self.models_dir,
                     self.figures_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Configuration PDE : équation de Burgers visqueuse
# -----------------------------------------------------------------------------

@dataclass
class PDEConfig:
    """
    Paramètres de la simulation de l'équation de Burgers visqueuse 1D.

    Équation cible :
        u_t + u u_x = nu * u_xx

    sur un domaine périodique [0, L].
    """

    # Longueur du domaine spatial.
    L: float = 2.0 * 3.141592653589793

    # Nombre de points de discrétisation en espace.
    Nx: int = 128

    # Viscosité.
    nu: float = 0.05

    # Temps final de simulation.
    T: float = 2.0

    # Pas de temps numérique.
    dt: float = 2.0e-4

    # Sauvegarder un snapshot tous les `save_every` pas de temps.
    save_every: int = 100

    # Type de conditions aux limites. Ici, on vise uniquement le cas périodique.
    boundary_condition: str = "periodic"

    # Choix du solveur.
    # Version initiale conseillée : différences finies + RK4 explicite.
    spatial_scheme: str = "finite_difference"
    time_scheme: str = "rk4"

    @property
    def dx(self) -> float:
        """
        Pas d'espace déduit automatiquement de L et Nx.
        """
        return self.L / self.Nx

    @property
    def n_steps(self) -> int:
        """
        Nombre total de pas de temps.
        """
        return int(self.T / self.dt)

    @property
    def n_saved_steps(self) -> int:
        """
        Nombre de snapshots sauvegardés, en incluant l'état initial.
        """
        return self.n_steps // self.save_every + 1


# -----------------------------------------------------------------------------
# 3. Configuration de génération du dataset
# -----------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """
    Paramètres liés à la génération et au découpage du dataset.
    """

    # Nombre de trajectoires par split.
    n_train: int = 500
    n_val: int = 100
    n_test: int = 100

    # Familles de conditions initiales autorisées.
    # On pourra tirer aléatoirement parmi ces types.
    ic_types: List[str] = field(default_factory=lambda: [
        "fourier",
        "gaussian_bumps",
        "random_smooth",
    ])

    # Paramètres de base pour les conditions initiales de type Fourier.
    max_fourier_mode: int = 3
    fourier_amplitude_scale: float = 0.5

    # Paramètres de base pour les bosses gaussiennes.
    n_gaussian_bumps: int = 2

    # Paramètres de lissage spectral pour les champs aléatoires lisses.
    spectral_decay: float = 3.0

    # Nom des fichiers de sortie par split.
    train_filename: str = "train_data.npz"
    val_filename: str = "val_data.npz"
    test_filename: str = "test_data.npz"


# -----------------------------------------------------------------------------
# 4. Configuration du prétraitement / normalisation
# -----------------------------------------------------------------------------

@dataclass
class NormalizationConfig:
    """
    Paramètres de normalisation des données.

    Pour préserver autant que possible l'information physique globale,
    on recommande une normalisation globale (moyenne / écart-type calculés
    sur le train), et non pas une normalisation indépendante trajectoire
    par trajectoire.
    """

    enabled: bool = True
    method: str = "global_standardization"  # autres possibilités futures : none, minmax
    eps: float = 1.0e-8


# -----------------------------------------------------------------------------
# 5. Configuration machine learning
# -----------------------------------------------------------------------------

@dataclass
class MLConfig:
    """
    Paramètres globaux pour les modèles d'apprentissage.
    """

    # Dimension du latent principal h_t.
    latent_dim: int = 8

    # Taille de batch.
    batch_size: int = 64

    # Nombre d'époques.
    num_epochs: int = 100

    # Taux d'apprentissage.
    learning_rate: float = 1.0e-3

    # Weight decay optionnel.
    weight_decay: float = 0.0

    # Nom de l'optimiseur.
    optimizer_name: str = "adam"

    # Device choisi automatiquement plus tard si disponible.
    device: str = "cpu"

    # Fraction éventuelle du dataset à utiliser pour des tests rapides.
    train_fraction: float = 1.0


# -----------------------------------------------------------------------------
# 6. Configuration des losses pour les modèles latents
# -----------------------------------------------------------------------------

@dataclass
class LossConfig:
    """
    Poids des différentes composantes de la fonction de coût.
    """

    # Poids de la reconstruction (si autoencodeur).
    reconstruction_weight: float = 1.0

    # Poids de la prédiction latente h_t -> h_{t+1}.
    prediction_weight: float = 1.0

    # Poids de la contrainte de monotonie sur la variable scalaire z_t.
    monotonicity_weight: float = 1.0

    # Poids pour éviter la solution triviale z = constante.
    variance_weight: float = 1.0e-2

    # Variance minimale souhaitée pour z.
    min_latent_variance: float = 1.0e-3

    # Type de pénalité de monotonie.
    monotonicity_mode: str = "squared_hinge"  # "hinge" ou "squared_hinge"


# -----------------------------------------------------------------------------
# 7. Configuration d'évaluation
# -----------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """
    Paramètres liés à l'évaluation et aux figures.
    """

    # Nombre de trajectoires à afficher dans les figures de contrôle.
    n_plot_trajectories: int = 5

    # Sauvegarder automatiquement les figures.
    save_figures: bool = True

    # Fréquence d'affichage de logs pendant l'entraînement.
    print_every: int = 10

    # Taille de police par défaut pour les figures matplotlib.
    matplotlib_fontsize: int = 11


# -----------------------------------------------------------------------------
# 8. Reproductibilité
# -----------------------------------------------------------------------------

@dataclass
class ReproConfig:
    """
    Paramètres de reproductibilité.
    """

    seed: int = 42
    deterministic_torch: bool = False


# -----------------------------------------------------------------------------
# 9. Configuration globale du projet
# -----------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """
    Configuration globale du projet.

    Cette classe agrège toutes les sous-configurations dans une seule structure.
    Ainsi, dans le reste du code, on pourra écrire par exemple :

        cfg.pde.nu
        cfg.dataset.n_train
        cfg.ml.latent_dim

    ce qui rend le code plus lisible.
    """

    project_name: str = "burgers_latent_physics"
    experiment_name: str = "baseline_v1"

    paths: PathsConfig = field(default_factory=PathsConfig)
    pde: PDEConfig = field(default_factory=PDEConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    reproducibility: ReproConfig = field(default_factory=ReproConfig)

    def create_directories(self) -> None:
        """
        Crée tous les dossiers nécessaires pour l'expérience.
        """
        self.paths.create_directories()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit récursivement la configuration en dictionnaire Python.

        Utile pour :
        - sauvegarder la configuration dans un fichier JSON,
        - logger les expériences,
        - inspecter facilement les paramètres.
        """
        data = asdict(self)

        # Les objets Path sont convertis en chaînes pour être sérialisables facilement.
        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            return obj

        return convert_paths(data)


# -----------------------------------------------------------------------------
# 10. Fonction utilitaire principale
# -----------------------------------------------------------------------------

def get_default_config(create_dirs: bool = True) -> ProjectConfig:
    """
    Construit et retourne la configuration par défaut du projet.

    Paramètres
    ----------
    create_dirs : bool
        Si True, crée automatiquement les dossiers de sortie.

    Retour
    ------
    ProjectConfig
        Objet de configuration prêt à être utilisé dans les scripts.
    """
    cfg = ProjectConfig()

    if create_dirs:
        cfg.create_directories()

    return cfg


# -----------------------------------------------------------------------------
# 11. Bloc de test simple
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_default_config(create_dirs=True)

    print("Configuration du projet chargée avec succès.\n")
    print("Nom du projet :", cfg.project_name)
    print("Nom de l'expérience :", cfg.experiment_name)
    print("Viscosité nu :", cfg.pde.nu)
    print("Domaine L :", cfg.pde.L)
    print("Nombre de points Nx :", cfg.pde.Nx)
    print("Pas d'espace dx :", cfg.pde.dx)
    print("Nombre total de pas de temps :", cfg.pde.n_steps)
    print("Nombre de snapshots sauvegardés :", cfg.pde.n_saved_steps)
    print("Dossier des données :", cfg.paths.data_dir)
    print("Dimension latente :", cfg.ml.latent_dim)
