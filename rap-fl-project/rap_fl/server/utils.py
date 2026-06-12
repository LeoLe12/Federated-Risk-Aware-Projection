"""Server-side math and bridge utilities for Federated Risk-Aware Projection (RAP-FL).

This module contains:
1. Bridge functions to convert between PyTorch state dicts and lists of NumPy ndarrays.
2. Frobenius inner product, projection, sterilization, and update aggregation math functions.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch


def ndarrays_to_peft_dict(
    ndarrays: List[np.ndarray], keys: List[str]
) -> Dict[str, torch.Tensor]:
    """Converts a list of NumPy arrays back to a PyTorch dictionary.

    Args:
        ndarrays: List of parameter tensors as NumPy ndarrays.
        keys: Reference keys list matching the parameter order.

    Returns:
        A dictionary mapping keys to PyTorch tensors.

    Raises:
        ValueError: If the lengths of ndarrays and keys do not match.
    """
    if len(ndarrays) != len(keys):
        raise ValueError(
            f"Dimension mismatch: Received {len(ndarrays)} ndarrays but expected "
            f"{len(keys)} parameters."
        )

    return {key: torch.from_numpy(arr) for key, arr in zip(keys, ndarrays)}


def peft_dict_to_ndarrays(
    peft_dict: Dict[str, torch.Tensor], keys: List[str]
) -> List[np.ndarray]:
    """Converts a PyTorch dict to a list of NumPy arrays matching the keys order.

    Args:
        peft_dict: Dictionary mapping parameter names to PyTorch tensors.
        keys: Reference keys list determining the sorting order.

    Returns:
        A list of parameter tensors as NumPy ndarrays.

    Raises:
        KeyError: If any key in the reference list is missing from the dictionary.
    """
    ndarrays = []
    for key in keys:
        if key not in peft_dict:
            raise KeyError(f"Key '{key}' missing from parameter dictionary.")
        ndarrays.append(peft_dict[key].detach().cpu().numpy())
    return ndarrays


def fed_avg_math(
    weights_list: List[Dict[str, torch.Tensor]],
    coefficients: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Performs weighted averaging (FedAvg) over a list of client state dicts.

    Args:
        weights_list: List of parameter dictionaries from clients.
        coefficients: Optional list of scaling coefficients for each client.
            If None, uniform weighting is applied.

    Returns:
        The aggregated parameter dictionary.
    """
    num_clients = len(weights_list)
    if num_clients == 0:
        return {}

    if coefficients is None:
        coefficients = [1.0 / num_clients] * num_clients

    # Normalize coefficients
    total_w = sum(coefficients)
    norm_coeffs = [c / total_w for c in coefficients]

    aggregated_weights = {}
    ref_keys = weights_list[0].keys()

    for key in ref_keys:
        # Create accumulator on CPU using the same dtype as the first client
        first_tensor = weights_list[0][key]
        weighted_sum = torch.zeros_like(first_tensor, device="cpu")

        for i, client_w in enumerate(weights_list):
            if key not in client_w:
                continue
            val = client_w[key].to("cpu")
            weighted_sum += val * norm_coeffs[i]

        aggregated_weights[key] = weighted_sum

    return aggregated_weights


def peft_inner_product(dict_a: Dict[str, torch.Tensor], dict_b: Dict[str, torch.Tensor]) -> float:
    """Computes the Frobenius inner product between two PEFT parameter dictionaries.

    Defined as: <A, B> = Sum_i <A_i, B_i>_F

    Args:
        dict_a: First PEFT update dict.
        dict_b: Second PEFT update dict.

    Returns:
        The Frobenius inner product scalar.
    """
    inner_prod = 0.0
    for key in dict_a.keys():
        if key in dict_b:
            inner_prod += torch.sum(dict_a[key] * dict_b[key]).item()
    return inner_prod


def project_onto(dict_v: Dict[str, torch.Tensor], dict_g: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Computes the orthogonal projection of dict_v onto the direction of dict_g.

    Defined as: Proj_G(V) = (<V, G> / ||G||^2) * G

    Args:
        dict_v: The dictionary of weights to project.
        dict_g: The dictionary of weights representing the base direction.

    Returns:
        The projected weights dict, or zeros if the norm of dict_g is <= 0.

    Raises:
        ValueError: If the key structures do not match.
    """
    if set(dict_v.keys()) != set(dict_g.keys()):
        raise ValueError("PEFT tensors must have identical keys for projection.")

    norm_sq_g = peft_inner_product(dict_g, dict_g)

    if norm_sq_g <= 0:
        return {k: torch.zeros_like(v) for k, v in dict_v.items()}

    inner_v_g = peft_inner_product(dict_v, dict_g)
    scalar_coef = inner_v_g / norm_sq_g

    projected_dict = {}
    for key in dict_g.keys():
        if key in dict_v:
            projected_dict[key] = dict_g[key] * scalar_coef

    return projected_dict


def apply_risk_projection(
    delta_k: Dict[str, torch.Tensor],
    G_R: Dict[str, torch.Tensor],
    penalty_factor: float,
    G_U_for_gate: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Applies risk projection serialization and optional utility gating.

    Calculates: tilde_Delta = Delta - penalty_factor * Proj_G_R(Delta)
    Optionally zeros the update if it has non-positive alignment with G_U_for_gate.

    Args:
        delta_k: Client local update dictionary.
        G_R: Global risk direction.
        penalty_factor: Suppression penalty strength (lambda_t or mu_t).
        G_U_for_gate: Optional global utility direction for gating.

    Returns:
        The sanitized update dictionary.
    """
    proj = project_onto(delta_k, G_R)

    delta_tilde = {}
    for key in delta_k.keys():
        if key in proj:
            delta_tilde[key] = delta_k[key] - penalty_factor * proj[key]
        else:
            delta_tilde[key] = delta_k[key].clone()

    # Apply Utility Gate if global utility direction is provided
    if G_U_for_gate is not None:
        alignment = peft_inner_product(delta_tilde, G_U_for_gate)
        if alignment <= 0:
            return {k: torch.zeros_like(v) for k, v in delta_tilde.items()}

    return delta_tilde


def final_aggregation_and_update(
    current_model: Dict[str, torch.Tensor],
    U_list: List[Dict[str, torch.Tensor]],
    R_list: List[Dict[str, torch.Tensor]],
    a_list: List[float],
    gamma_t: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Performs the final update aggregation and applies it to the global parameters.

    Calculates:
        Delta_t = Sum_k a_k * U_k + gamma_t * Sum_k a_k * R_k
        phi_{t+1} = phi_t + Delta_t

    Args:
        current_model: Global PEFT parameters before update.
        U_list: List of sanitized utility updates.
        R_list: List of salvaged risk updates.
        a_list: List of client aggregation weights (normalized).
        gamma_t: Salvage coefficient.

    Returns:
        A tuple of (next_model, Delta_t).
    """
    sum_U = {k: torch.zeros_like(v, device="cpu") for k, v in current_model.items()}
    sum_R = {k: torch.zeros_like(v, device="cpu") for k, v in current_model.items()}

    for U_k, R_k, a_k in zip(U_list, R_list, a_list):
        for key in current_model.keys():
            if key in U_k:
                sum_U[key] += a_k * U_k[key].to("cpu")
            if key in R_k:
                sum_R[key] += a_k * R_k[key].to("cpu")

    Delta_t = {}
    next_model = {}
    for key in current_model.keys():
        Delta_t[key] = sum_U[key] + gamma_t * sum_R[key]
        next_model[key] = current_model[key].to("cpu") + Delta_t[key]

    return next_model, Delta_t
