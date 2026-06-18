"""
ml/losses.py
============

Fonctions de coût pour le projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce module regroupe les pertes utiles pour les deux familles d'expériences du projet :

1. apprentissage supervisé (régression d'énergie),
2. apprentissage latent / auto-supervisé :
   - reconstruction,
   - prédiction latente,
   - monotonie d'une observable scalaire,
   - régularisation pour éviter l'effondrement du latent.

Philosophie
-----------
- fonctions simples, lisibles et indépendantes,
- possibilité de composer plusieurs pertes proprement,
- retour explicite des différents termes pour faciliter le logging.

Conventions
-----------
- Les pertes renvoient des tenseurs scalaire PyTorch.
- Les tenseurs d'entrée ont en général une dimension batch en première position.
- Pour les sorties scalaires, on accepte des formes (batch, 1) ou (batch,).
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Utilitaires internes
# =============================================================================

def _ensure_compatible_shapes(a: torch.Tensor, b: torch.Tensor, name_a: str = 'a', name_b: str = 'b') -> None:
    """
    Vérifie simplement que deux tenseurs ont exactement la même forme.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Formes incompatibles pour {name_a} et {name_b} : {tuple(a.shape)} vs {tuple(b.shape)}"
        )



def _flatten_scalar_like(z: torch.Tensor) -> torch.Tensor:
    """
    Aplati une sortie scalaire batchée vers une forme (batch,).

    Accepte en entrée :
    - (batch, 1)
    - (batch,)

    Retour
    ------
    torch.Tensor
        Tenseur de forme (batch,).
    """
    if z.ndim == 2 and z.shape[1] == 1:
        return z[:, 0]
    if z.ndim == 1:
        return z
    raise ValueError(
        f"Le tenseur scalaire attendu doit être de forme (batch,) ou (batch, 1), obtenu {tuple(z.shape)}"
    )


# =============================================================================
# 2. Pertes élémentaires classiques
# =============================================================================

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Erreur quadratique moyenne standard.
    """
    _ensure_compatible_shapes(pred, target, 'pred', 'target')
    return F.mse_loss(pred, target)



def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Erreur absolue moyenne standard.
    """
    _ensure_compatible_shapes(pred, target, 'pred', 'target')
    return F.l1_loss(pred, target)



def reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor, mode: str = 'mse') -> torch.Tensor:
    """
    Perte de reconstruction entre un état reconstruit et un état cible.

    Paramètres
    ----------
    reconstruction : torch.Tensor
        Sortie du décodeur.
    target : torch.Tensor
        État de référence.
    mode : str
        'mse' ou 'mae'.
    """
    _ensure_compatible_shapes(reconstruction, target, 'reconstruction', 'target')
    mode = mode.lower()
    if mode == 'mse':
        return F.mse_loss(reconstruction, target)
    if mode == 'mae':
        return F.l1_loss(reconstruction, target)
    raise ValueError("`mode` doit être 'mse' ou 'mae'.")



def latent_prediction_loss(predicted_latent: torch.Tensor, target_latent: torch.Tensor, mode: str = 'mse') -> torch.Tensor:
    """
    Perte entre latent prédit et latent cible.

    Paramètres
    ----------
    predicted_latent : torch.Tensor
        Latent prédit, par exemple h_{t+1}^{pred}.
    target_latent : torch.Tensor
        Latent cible, par exemple h_{t+1}.
    mode : str
        'mse' ou 'mae'.
    """
    _ensure_compatible_shapes(predicted_latent, target_latent, 'predicted_latent', 'target_latent')
    mode = mode.lower()
    if mode == 'mse':
        return F.mse_loss(predicted_latent, target_latent)
    if mode == 'mae':
        return F.l1_loss(predicted_latent, target_latent)
    raise ValueError("`mode` doit être 'mse' ou 'mae'.")


# =============================================================================
# 3. Pertes liées à la monotonie
# =============================================================================

def monotonicity_violations(z_t: torch.Tensor, z_tp: torch.Tensor, atol: float = 0.0) -> torch.Tensor:
    """
    Calcule les violations positives de monotonie décroissante entre z_t et z_{t+1}.

    Idée
    ----
    Si l'on souhaite que z_{t+1} <= z_t, alors toute quantité positive de la forme

        z_{t+1} - z_t - atol

    constitue une violation.

    Paramètres
    ----------
    z_t : torch.Tensor
        Observable scalaire au temps t.
    z_tp : torch.Tensor
        Observable scalaire au temps t+1.
    atol : float
        Tolérance absolue.

    Retour
    ------
    torch.Tensor
        Tenseur de forme (batch,) contenant les violations positives.
    """
    z_t_flat = _flatten_scalar_like(z_t)
    z_tp_flat = _flatten_scalar_like(z_tp)
    _ensure_compatible_shapes(z_t_flat, z_tp_flat, 'z_t', 'z_tp')

    return torch.relu(z_tp_flat - z_t_flat - float(atol))



def monotonicity_loss(
    z_t: torch.Tensor,
    z_tp: torch.Tensor,
    mode: str = 'squared_hinge',
    atol: float = 0.0,
) -> torch.Tensor:
    """
    Pénalise les violations de monotonie décroissante entre z_t et z_{t+1}.

    Modes disponibles
    -----------------
    - 'hinge'         : moyenne des violations positives
    - 'squared_hinge' : moyenne des carrés des violations positives

    Paramètres
    ----------
    z_t : torch.Tensor
        Observable scalaire au temps t.
    z_tp : torch.Tensor
        Observable scalaire au temps t+1.
    mode : str
        Type de pénalité.
    atol : float
        Tolérance absolue.
    """
    violations = monotonicity_violations(z_t, z_tp, atol=atol)

    mode = mode.lower()
    if mode == 'hinge':
        return torch.mean(violations)
    if mode == 'squared_hinge':
        return torch.mean(violations ** 2)

    raise ValueError("`mode` doit être 'hinge' ou 'squared_hinge'.")


# =============================================================================
# 4. Régularisation de variance / anti-collapse
# =============================================================================

def variance_regularization(z: torch.Tensor, min_variance: float = 1.0e-3) -> torch.Tensor:
    """
    Pénalise une variance trop faible de la variable scalaire z.

    Idée
    ----
    Une solution triviale de certaines pertes peut être z = constante.
    Pour l'éviter, on impose indirectement que la variance par batch reste
    supérieure à un seuil minimal.

    Pénalité utilisée
    -----------------
        penalty = relu(min_variance - Var(z))

    Paramètres
    ----------
    z : torch.Tensor
        Observable scalaire batchée.
    min_variance : float
        Variance minimale souhaitée.
    """
    z_flat = _flatten_scalar_like(z)
    var = torch.var(z_flat, unbiased=False)
    return torch.relu(torch.as_tensor(min_variance, dtype=z_flat.dtype, device=z_flat.device) - var)



def latent_variance_regularization(h: torch.Tensor, min_variance: float = 1.0e-3) -> torch.Tensor:
    """
    Variante de régularisation appliquée à un latent vectoriel h.

    On calcule la variance moyenne sur les dimensions latentes et on pénalise
    un effondrement global du batch latent.

    Paramètres
    ----------
    h : torch.Tensor
        Latent de forme (batch, latent_dim).
    min_variance : float
        Variance minimale moyenne souhaitée.
    """
    if h.ndim != 2:
        raise ValueError(f"`h` doit être 2D (batch, latent_dim), obtenu {tuple(h.shape)}")

    var_per_dim = torch.var(h, dim=0, unbiased=False)
    mean_var = torch.mean(var_per_dim)
    return torch.relu(torch.as_tensor(min_variance, dtype=h.dtype, device=h.device) - mean_var)


# =============================================================================
# 5. Combinaisons de pertes
# =============================================================================

def supervised_energy_loss(pred_energy: torch.Tensor, target_energy: torch.Tensor, mode: str = 'mse') -> torch.Tensor:
    """
    Perte supervisée pour la régression d'énergie.

    Paramètres
    ----------
    pred_energy : torch.Tensor
        Prédiction du modèle.
    target_energy : torch.Tensor
        Énergie cible.
    mode : str
        'mse' ou 'mae'.
    """
    mode = mode.lower()
    _ensure_compatible_shapes(pred_energy, target_energy, 'pred_energy', 'target_energy')
    if mode == 'mse':
        return F.mse_loss(pred_energy, target_energy)
    if mode == 'mae':
        return F.l1_loss(pred_energy, target_energy)
    raise ValueError("`mode` doit être 'mse' ou 'mae'.")



def autoencoder_total_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    scalar_t: Optional[torch.Tensor] = None,
    scalar_tp: Optional[torch.Tensor] = None,
    reconstruction_weight: float = 1.0,
    monotonicity_weight: float = 0.0,
    variance_weight: float = 0.0,
    monotonicity_mode: str = 'squared_hinge',
    min_scalar_variance: float = 1.0e-3,
) -> Dict[str, torch.Tensor]:
    """
    Combine plusieurs termes de perte pour un autoencodeur latent.

    Cas d'usage typique
    -------------------
    - reconstruction de l'état,
    - observable scalaire optionnelle,
    - pénalité de monotonie entre z_t et z_{t+1},
    - régularisation anti-collapse sur z.

    Retour
    ------
    Dict[str, torch.Tensor]
        Dictionnaire contenant :
        - total_loss
        - reconstruction_loss
        - monotonicity_loss
        - variance_loss
    """
    rec = reconstruction_loss(reconstruction, target, mode='mse')

    mono = torch.zeros((), dtype=rec.dtype, device=rec.device)
    if scalar_t is not None and scalar_tp is not None and monotonicity_weight > 0.0:
        mono = monotonicity_loss(scalar_t, scalar_tp, mode=monotonicity_mode)

    var_loss = torch.zeros((), dtype=rec.dtype, device=rec.device)
    if scalar_t is not None and variance_weight > 0.0:
        var_loss = variance_regularization(scalar_t, min_variance=min_scalar_variance)

    total = (
        float(reconstruction_weight) * rec
        + float(monotonicity_weight) * mono
        + float(variance_weight) * var_loss
    )

    return {
        'total_loss': total,
        'reconstruction_loss': rec,
        'monotonicity_loss': mono,
        'variance_loss': var_loss,
    }



def latent_dynamics_total_loss(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
    z_t: Optional[torch.Tensor] = None,
    z_tp: Optional[torch.Tensor] = None,
    prediction_weight: float = 1.0,
    monotonicity_weight: float = 0.0,
    variance_weight: float = 0.0,
    prediction_mode: str = 'mse',
    monotonicity_mode: str = 'squared_hinge',
    min_scalar_variance: float = 1.0e-3,
) -> Dict[str, torch.Tensor]:
    """
    Combine plusieurs pertes pour un modèle de dynamique latente.

    Cas d'usage typique
    -------------------
    - correspondance h_{t+1}^{pred} ~ h_{t+1},
    - contrainte de monotonie sur z_t -> z_{t+1},
    - régularisation anti-collapse sur z_t.

    Retour
    ------
    Dict[str, torch.Tensor]
        Dictionnaire contenant :
        - total_loss
        - prediction_loss
        - monotonicity_loss
        - variance_loss
    """
    pred = latent_prediction_loss(predicted_latent, target_latent, mode=prediction_mode)

    mono = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if z_t is not None and z_tp is not None and monotonicity_weight > 0.0:
        mono = monotonicity_loss(z_t, z_tp, mode=monotonicity_mode)

    var_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if z_t is not None and variance_weight > 0.0:
        var_loss = variance_regularization(z_t, min_variance=min_scalar_variance)

    total = (
        float(prediction_weight) * pred
        + float(monotonicity_weight) * mono
        + float(variance_weight) * var_loss
    )

    return {
        'total_loss': total,
        'prediction_loss': pred,
        'monotonicity_loss': mono,
        'variance_loss': var_loss,
    }


# =============================================================================
# 6. Interface générique de composition (optionnelle)
# =============================================================================

def combine_weighted_losses(loss_terms: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
    """
    Combine un dictionnaire de pertes selon des poids fournis.

    Paramètres
    ----------
    loss_terms : Dict[str, torch.Tensor]
        Dictionnaire de pertes élémentaires.
    weights : Dict[str, float] ou None
        Poids associés. Si None, chaque terme est pris avec un poids 1.

    Retour
    ------
    Dict[str, torch.Tensor]
        Dictionnaire contenant :
        - total_loss
        - tous les termes de perte d'entrée
    """
    if len(loss_terms) == 0:
        raise ValueError('`loss_terms` ne doit pas être vide.')

    first_tensor = next(iter(loss_terms.values()))
    total = torch.zeros((), dtype=first_tensor.dtype, device=first_tensor.device)

    if weights is None:
        weights = {key: 1.0 for key in loss_terms.keys()}

    for key, value in loss_terms.items():
        weight = float(weights.get(key, 1.0))
        total = total + weight * value

    out = dict(loss_terms)
    out['total_loss'] = total
    return out


# =============================================================================
# 7. Bloc de test minimal
# =============================================================================

if __name__ == '__main__':
    torch.manual_seed(0)

    batch_size = 6
    nx = 10
    latent_dim = 4

    x = torch.randn(batch_size, nx)
    x_hat = torch.randn(batch_size, nx)

    h_t = torch.randn(batch_size, latent_dim)
    h_tp = torch.randn(batch_size, latent_dim)

    z_t = torch.randn(batch_size, 1)
    z_tp = z_t - 0.1 * torch.rand(batch_size, 1)  # plutôt décroissant

    rec = reconstruction_loss(x_hat, x)
    pred = latent_prediction_loss(h_t, h_tp)
    mono = monotonicity_loss(z_t, z_tp)
    var_reg = variance_regularization(z_t)

    auto_loss = autoencoder_total_loss(
        reconstruction=x_hat,
        target=x,
        scalar_t=z_t,
        scalar_tp=z_tp,
        reconstruction_weight=1.0,
        monotonicity_weight=0.5,
        variance_weight=0.1,
    )

    dyn_loss = latent_dynamics_total_loss(
        predicted_latent=h_t,
        target_latent=h_tp,
        z_t=z_t,
        z_tp=z_tp,
        prediction_weight=1.0,
        monotonicity_weight=0.5,
        variance_weight=0.1,
    )

    print('Test rapide de ml/losses.py')
    print('- reconstruction_loss :', float(rec.item()))
    print('- prediction_loss     :', float(pred.item()))
    print('- monotonicity_loss   :', float(mono.item()))
    print('- variance_regularization :', float(var_reg.item()))
    print('\nClés autoencoder_total_loss :', list(auto_loss.keys()))
    print('Clés latent_dynamics_total_loss :', list(dyn_loss.keys()))
