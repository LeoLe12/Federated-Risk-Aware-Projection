"""Verification test for RAPTrainer.

Ensures that the double backward pass, dual optimizer, and delta calculations
work as expected while keeping the frozen base model weights unchanged.
"""

import os
import sys
from types import SimpleNamespace
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling

# Add parent directory to path so we can import rap_fl
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rap_fl.client.trainer import RAPTrainer


class MockCausalLM(nn.Module):
    """Simple mock causal language model for verification testing."""

    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Trainable parameters (representing LoRA adapters)
        self.lora_linear = nn.Linear(hidden_dim, hidden_dim)
        # Frozen base parameters
        self.base_head = nn.Linear(hidden_dim, vocab_size)
        self.base_head.weight.requires_grad = False

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.lora_linear(x)
        logits = self.base_head(x)
        return SimpleNamespace(logits=logits)


def main():
    print("--- STARTING RAP_TRAINER VERIFICATION ---")

    # 1. Setup seed for reproducibility
    torch.manual_seed(42)

    # 2. Instantiate Model and save its initial state
    model = MockCausalLM()
    
    # Keep track of initial weights
    initial_lora_weight = model.lora_linear.weight.clone().detach()
    initial_embedding_weight = model.embedding.weight.clone().detach()
    
    print("[OK] Model instantiated successfully.")

    # 3. Create dummy dataset with risk_score column
    # 4 samples, each with input_ids and labels (length 5)
    data = {
        "input_ids": [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2],
        ],
        "labels": [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2],
        ],
        "risk_score": [0.0, 1.0, 0.2, 0.8]
    }
    dataset = Dataset.from_dict(data)
    print(f"[OK] Created dummy dataset with {len(dataset)} samples and columns: {dataset.column_names}")

    # 4. Set training arguments (with gradient accumulation)
    training_args = TrainingArguments(
        output_dir="./tmp_test_output",
        learning_rate=1e-2,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        optim="adamw_torch",
        report_to="none",
        disable_tqdm=True,
    )
    print("[OK] Training arguments set (batch_size=2, accumulation_steps=2).")

    # 5. Data collator (simple dict pass-through since data is already tokenized)
    def simple_collator(features):
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "labels": torch.tensor([f["labels"] for f in features]),
            "risk_score": torch.tensor([f["risk_score"] for f in features]),
        }
        return batch

    # 6. Instantiate RAPTrainer
    trainer = RAPTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=simple_collator,
    )
    print("[OK] RAPTrainer instantiated successfully.")

    # 7. Execute training
    print("[RUN] Running RAPTrainer.train()...")
    trainer.train()
    print("[OK] Training completed.")

    # 8. Assertions
    # A. Check if delta dictionaries were populated
    assert trainer.delta_U is not None, "Error: trainer.delta_U is None!"
    assert trainer.delta_R is not None, "Error: trainer.delta_R is None!"
    print("[OK] Deltas are not None.")

    # B. Check that keys in deltas match trainable parameters
    trainable_keys = [k for k, v in model.named_parameters() if v.requires_grad]
    print(f"Trainable keys: {trainable_keys}")
    print(f"Delta_U keys: {list(trainer.delta_U.keys())}")
    
    for key in trainable_keys:
        assert key in trainer.delta_U, f"Error: {key} missing from delta_U!"
        assert key in trainer.delta_R, f"Error: {key} missing from delta_R!"
    print("[OK] Delta dictionaries contain correct parameter keys.")

    # C. Check that the base model's trainable weights remained UNCHANGED
    # Since RAPOptimizer does not optimize base model parameters, they must be untouched.
    current_lora_weight = model.lora_linear.weight
    current_embedding_weight = model.embedding.weight
    
    assert torch.equal(current_lora_weight, initial_lora_weight), "Error: Base model lora weight changed!"
    assert torch.equal(current_embedding_weight, initial_embedding_weight), "Error: Base model embedding weight changed!"
    print("[OK] Base model parameters are identical to starting weights (no model drift).")

    # D. Check that delta_U and delta_R are different
    # Since risk weights are different from utility weights (e.g. 1.0 vs 0.0, etc.),
    # the gradients and updates must differ.
    diff_u_r_lora = torch.sum(torch.abs(trainer.delta_U["lora_linear.weight"] - trainer.delta_R["lora_linear.weight"]))
    print(f"Difference between delta_U and delta_R for lora_linear.weight: {diff_u_r_lora.item():.6f}")
    assert diff_u_r_lora > 0.0, "Error: delta_U and delta_R are identical!"
    print("[OK] delta_U and delta_R are distinct.")

    # E. Check that deltas are on CPU
    assert trainer.delta_U["lora_linear.weight"].device.type == "cpu", "Error: delta_U is not on CPU!"
    assert trainer.delta_R["lora_linear.weight"].device.type == "cpu", "Error: delta_R is not on CPU!"
    print("[OK] Deltas were correctly moved to CPU.")

    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
