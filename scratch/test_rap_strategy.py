"""Verification test for RAPStrategy and server-side utilities.

Simulates a federated learning aggregation round to verify NumPy-PyTorch conversions,
projections, sterilization, and final model parameter updates.
"""

import os
import sys
import numpy as np
import torch
import flwr as fl

# Add parent directory to path so we can import rap_fl
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rap_fl.server.utils import (
    ndarrays_to_peft_dict,
    peft_dict_to_ndarrays,
    peft_inner_product,
    project_onto,
)
from rap_fl.server.strategy import RAPStrategy


class MockClientProxy(fl.server.client_proxy.ClientProxy):
    """Mock ClientProxy subclass required for aggregate_fit signature."""
    def __init__(self, cid: str):
        super().__init__(cid=cid)

    def get_properties(self, *args, **kwargs) -> fl.common.GetPropertiesRes:
        pass

    def get_parameters(self, *args, **kwargs) -> fl.common.GetParametersRes:
        pass

    def fit(self, *args, **kwargs) -> fl.common.FitRes:
        pass

    def evaluate(self, *args, **kwargs) -> fl.common.EvaluateRes:
        pass

    def reconnect(self, *args, **kwargs) -> fl.common.DisconnectRes:
        pass


def main():
    print("--- STARTING RAP_STRATEGY VERIFICATION ---")

    # 1. Define reference keys for PEFT/LoRA model params
    keys = ["lora_A.weight", "lora_B.weight"]
    num_params = len(keys)
    print(f"[OK] Defined {num_params} reference keys: {keys}")

    # 2. Define initial parameters (phi_t)
    initial_peft_dict = {
        "lora_A.weight": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "lora_B.weight": torch.tensor([1.0, 1.0], dtype=torch.float32),
    }
    initial_ndarrays = peft_dict_to_ndarrays(initial_peft_dict, keys)
    initial_parameters = fl.common.ndarrays_to_parameters(initial_ndarrays)
    print("[OK] Created initial global model parameters on server.")

    # 3. Instantiate RAPStrategy
    strategy = RAPStrategy(
        keys=keys,
        lambda_t=1.0,
        mu_t=1.0,
        gamma_t=0.5,
        initial_parameters=initial_parameters,
    )
    print("[OK] RAPStrategy instantiated with lambda=1.0, mu=1.0, gamma=0.5.")

    # Verify that initial parameters were correctly unpacked into the strategy state
    assert strategy.current_parameters is not None
    assert torch.equal(strategy.current_parameters["lora_A.weight"], initial_peft_dict["lora_A.weight"])
    print("[OK] Strategy successfully unpacked initial parameters into PyTorch dict.")

    # 4. Construct mock client fit results
    # We will simulate 2 clients sending updates (delta_U, delta_R).
    # Client updates details:
    # Client 1:
    #   delta_U_1 = [1.0, 1.0] for lora_A, [1.0, 1.0] for lora_B
    #   delta_R_1 = [0.0, 2.0] for lora_A, [0.0, 2.0] for lora_B
    # Client 2:
    #   delta_U_2 = [1.0, 1.0] for lora_A, [1.0, 1.0] for lora_B
    #   delta_R_2 = [0.0, 2.0] for lora_A, [0.0, 2.0] for lora_B
    #
    # Expected Global directions:
    # G_U = [1.0, 1.0]
    # G_R = [0.0, 2.0]
    #
    # Projection of delta_U on G_R:
    # Proj_G_R(delta_U) = (<delta_U, G_R> / ||G_R||^2) * G_R
    # <delta_U, G_R> = 1.0*0.0 + 1.0*2.0 (A) + 1.0*0.0 + 1.0*2.0 (B) = 4.0
    # ||G_R||^2 = (0^2 + 2^2) + (0^2 + 2^2) = 8.0
    # coeff = 4.0 / 8.0 = 0.5
    # Proj = 0.5 * G_R = [0.0, 1.0]
    #
    # Sterilized Utility update:
    # tilde_delta_U = delta_U - lambda_t * Proj = [1.0, 1.0] - 1.0 * [0.0, 1.0] = [1.0, 0.0]
    # (The component aligned with the risk vector is successfully projected out!)
    
    delta_U_1 = [np.array([1.0, 1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)]
    delta_R_1 = [np.array([0.0, 2.0], dtype=np.float32), np.array([0.0, 2.0], dtype=np.float32)]
    
    # Concatenate delta_U and delta_R to match single-list protocol
    client_1_ndarrays = delta_U_1 + delta_R_1
    client_2_ndarrays = delta_U_1 + delta_R_1
    
    client_1_parameters = fl.common.ndarrays_to_parameters(client_1_ndarrays)
    client_2_parameters = fl.common.ndarrays_to_parameters(client_2_ndarrays)
    
    client_1 = MockClientProxy(cid="client_1")
    client_2 = MockClientProxy(cid="client_2")
    
    fit_res_1 = fl.common.FitRes(
        status=fl.common.Status(code=fl.common.Code.OK, message=""),
        parameters=client_1_parameters,
        num_examples=10,
        metrics={},
    )
    fit_res_2 = fl.common.FitRes(
        status=fl.common.Status(code=fl.common.Code.OK, message=""),
        parameters=client_2_parameters,
        num_examples=10,
        metrics={},
    )
    
    results = [(client_1, fit_res_1), (client_2, fit_res_2)]
    print("[OK] Constructed client updates with non-orthogonal utility and risk.")

    # 5. Execute aggregate_fit
    print("[RUN] Running strategy.aggregate_fit()...")
    aggregated_parameters, metrics = strategy.aggregate_fit(
        server_round=1,
        results=results,
        failures=[],
    )
    print("[OK] Strategy.aggregate_fit executed successfully.")

    # 6. Verify outputs
    # Unpack aggregated parameters
    aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
    aggregated_dict = ndarrays_to_peft_dict(aggregated_ndarrays, keys)
    print("Aggregated model parameters:")
    for k, v in aggregated_dict.items():
        print(f"  {k}: {v.numpy()}")

    # Math expectation verification:
    # client weights: 10 examples each -> uniform weights (a_1 = 0.5, a_2 = 0.5)
    # tilde_u_1 = tilde_u_2 = [1.0, 0.0]
    # Sum a_k * tilde_u_k = [1.0, 0.0]
    #
    # tilde_r_1 = tilde_r_2 = delta_R - mu_t * Proj_G_R(delta_R)
    # Since delta_R is G_R, Proj_G_R(delta_R) = delta_R.
    # tilde_r = [0.0, 2.0] - 1.0 * [0.0, 2.0] = [0.0, 0.0]. (Risk update completely neutralized)
    # Sum a_k * tilde_r_k = [0.0, 0.0]
    #
    # Net global update Delta_t = Sum_U + gamma_t * Sum_R = [1.0, 0.0] + 0.5 * [0.0, 0.0] = [1.0, 0.0]
    # Next global weights: phi_{t+1} = phi_t + Delta_t = [1.0, 1.0] + [1.0, 0.0] = [2.0, 1.0]
    
    expected_lora_A = torch.tensor([2.0, 1.0], dtype=torch.float32)
    expected_lora_B = torch.tensor([2.0, 1.0], dtype=torch.float32)
    
    assert torch.allclose(aggregated_dict["lora_A.weight"], expected_lora_A)
    assert torch.allclose(aggregated_dict["lora_B.weight"], expected_lora_B)
    print("[OK] Math matches theoretical projection and salvage equations precisely.")

    print("\n[SUCCESS] ALL SERVER-SIDE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
