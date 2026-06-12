"""Custom Flower server-side strategy for Federated Risk-Aware Projection (RAP-FL).

This module contains the RAPStrategy strategy class that integrates risk mitigation
into Flower-based federated training.
"""

from typing import List, Tuple, Dict, Union, Optional
import flwr as fl
import torch
import numpy as np

from .utils import (
    ndarrays_to_peft_dict,
    peft_dict_to_ndarrays,
    fed_avg_math,
    apply_risk_projection,
    final_aggregation_and_update,
)


class RAPStrategy(fl.server.strategy.FedAvg):
    """Custom Flower Strategy implementing Federated Risk-Aware Projection (RAP-FL).
    
    This strategy overrides `aggregate_fit` to sterilize client updates before
    aggregating them. It expects clients to submit a concatenated list of NumPy
    ndarrays containing both the utility updates (delta_U) and risk updates (delta_R).
    
    Attributes:
        keys (List[str]): Reference names of the trainable adapter parameters.
        lambda_t (float): Suppression factor for utility updates.
        mu_t (float): Suppression factor for risk updates.
        gamma_t (float): Salvage coefficient for risk updates.
        current_parameters (Optional[Dict[str, torch.Tensor]]): Current global weights.
    """

    def __init__(
        self,
        *,
        keys: List[str],
        lambda_t: float = 1.0,
        mu_t: float = 1.0,
        gamma_t: float = 0.5,
        **kwargs,
    ):
        """Initializes the RAPStrategy.

        Args:
            keys: Reference names of the trainable adapter parameters.
            lambda_t: Suppression factor for utility updates. Defaults to 1.0.
            mu_t: Suppression factor for risk updates. Defaults to 1.0.
            gamma_t: Salvage coefficient for risk updates. Defaults to 0.5.
            **kwargs: Standard keyword arguments passed to flwr.server.strategy.FedAvg.
        """
        super().__init__(**kwargs)
        self.keys = keys
        self.lambda_t = lambda_t
        self.mu_t = mu_t
        self.gamma_t = gamma_t

        self.current_parameters: Optional[Dict[str, torch.Tensor]] = None

        # Pre-initialize current_parameters if initial_parameters is provided at startup
        if self.initial_parameters is not None:
            init_ndarrays = fl.common.parameters_to_ndarrays(self.initial_parameters)
            if len(init_ndarrays) == len(self.keys):
                self.current_parameters = ndarrays_to_peft_dict(init_ndarrays, self.keys)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregates fit results, applying RAP-FL projections and updates.

        Assumes clients send a single parameter list concatenating delta_U and delta_R.

        Args:
            server_round: The current round of federated learning.
            results: Successful client update results from fit.
            failures: Client fit failures.

        Returns:
            A tuple of (aggregated_parameters, metrics_dict).
        """
        # Standard safety check
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        list_delta_U: List[Dict[str, torch.Tensor]] = []
        list_delta_R: List[Dict[str, torch.Tensor]] = []
        client_weights: List[float] = []

        num_params = len(self.keys)

        # 1. Unpack concatenated client parameters and convert to PyTorch dicts
        for client, fit_res in results:
            ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)

            if len(ndarrays) != 2 * num_params:
                raise ValueError(
                    f"Protocol error: Client sent {len(ndarrays)} parameter arrays "
                    f"but expected {2 * num_params} (concatenated delta_U and delta_R "
                    f"for {num_params} keys)."
                )

            # Split in half
            delta_U_ndarrays = ndarrays[:num_params]
            delta_R_ndarrays = ndarrays[num_params:]

            # Convert to PyTorch dictionaries
            delta_U = ndarrays_to_peft_dict(delta_U_ndarrays, self.keys)
            delta_R = ndarrays_to_peft_dict(delta_R_ndarrays, self.keys)

            list_delta_U.append(delta_U)
            list_delta_R.append(delta_R)
            client_weights.append(float(fit_res.num_examples))

        # 2. Compute global directions G_U and G_R (using weighted FedAvg on deltas)
        G_U = fed_avg_math(list_delta_U, client_weights)
        G_R = fed_avg_math(list_delta_R, client_weights)

        # Normalize client weights for projection and final updates
        total_weight = sum(client_weights)
        if total_weight > 0:
            a_list = [w / total_weight for w in client_weights]
        else:
            a_list = [1.0 / len(client_weights)] * len(client_weights)

        # 3. Sterilize utility updates and salvage risk updates for each client
        clean_U_list: List[Dict[str, torch.Tensor]] = []
        clean_R_list: List[Dict[str, torch.Tensor]] = []

        for delta_U, delta_R in zip(list_delta_U, list_delta_R):
            # Utility sterilization (project out global risk direction)
            tilde_u = apply_risk_projection(
                delta_k=delta_U,
                G_R=G_R,
                penalty_factor=self.lambda_t,
                G_U_for_gate=None,
            )
            # Risk salvage (project out global risk direction and gate by global utility)
            tilde_r = apply_risk_projection(
                delta_k=delta_R,
                G_R=G_R,
                penalty_factor=self.mu_t,
                G_U_for_gate=G_U,
            )
            clean_U_list.append(tilde_u)
            clean_R_list.append(tilde_r)

        # 4. Resolve current global model parameters
        if self.current_parameters is None:
            if self.initial_parameters is not None:
                init_ndarrays = fl.common.parameters_to_ndarrays(self.initial_parameters)
                if len(init_ndarrays) == num_params:
                    self.current_parameters = ndarrays_to_peft_dict(init_ndarrays, self.keys)
                else:
                    raise ValueError(
                        f"Initial parameters array count ({len(init_ndarrays)}) "
                        f"does not match keys count ({num_params})."
                    )
            else:
                # Fallback to zero parameters if initial parameters are missing
                print(
                    "WARNING: RAPStrategy has no current global parameters initialized. "
                    "Accumulating updates starting from zero base."
                )
                self.current_parameters = {
                    key: torch.zeros_like(tensor, device="cpu")
                    for key, tensor in G_U.items()
                }

        # 5. Perform final aggregation and global model update
        next_model, _ = final_aggregation_and_update(
            current_model=self.current_parameters,
            U_list=clean_U_list,
            R_list=clean_R_list,
            a_list=a_list,
            gamma_t=self.gamma_t,
        )

        # Save updated parameters
        self.current_parameters = next_model

        # 6. Convert PyTorch state dict back to list of NumPy ndarrays
        next_model_ndarrays = peft_dict_to_ndarrays(next_model, self.keys)

        # Wrap in Flower Parameters object
        parameters = fl.common.ndarrays_to_parameters(next_model_ndarrays)

        return parameters, {}
