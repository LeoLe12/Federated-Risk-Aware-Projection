"""Client-side training wrappers for Federated Risk-Aware Projection (RAP-FL).

This module contains:
1. RAPOptimizer: A wrapper PyTorch optimizer managing dual objectives.
2. RAPTrainer: A Hugging Face Trainer subclass managing the single-pass client training.
"""

from typing import Dict, Any, Union, Optional, List, Type
import inspect
import torch
from torch.optim import Optimizer
from transformers import Trainer


class RAPOptimizer(Optimizer):
    """Optimizer wrapper for Federated Risk-Aware Projection (RAP-FL).
    
    This optimizer coordinates two underlying optimizers (`optim_U` and `optim_R`)
    acting on independent parameter buffers (`phi_U` and `phi_R`). It synchronizes
    hyperparameters (like learning rate) set by the Hugging Face Trainer scheduler
    and executes weight steps on both optimizers.
    
    Attributes:
        optim_U (Optimizer): Optimizer for utility weights.
        optim_R (Optimizer): Optimizer for risk weights.
        phi_U (Dict[str, torch.Tensor]): Trainable utility parameter buffers.
        phi_R (Dict[str, torch.Tensor]): Trainable risk parameter buffers.
        base_model (torch.nn.Module): The underlying model.
    """

    def __init__(
        self,
        optim_U: Optimizer,
        optim_R: Optimizer,
        phi_U: Dict[str, torch.Tensor],
        phi_R: Dict[str, torch.Tensor],
        base_model: torch.nn.Module,
    ):
        """Initializes the RAPOptimizer wrapper.

        Args:
            optim_U: Optimizer for utility weights.
            optim_R: Optimizer for risk weights.
            phi_U: Trainable utility parameter buffers.
            phi_R: Trainable risk parameter buffers.
            base_model: The underlying model.
        """
        # Register phi_U parameters as the main parameter group for the wrapper
        params = list(phi_U.values())
        defaults = dict(lr=optim_U.param_groups[0].get("lr", 1e-5))
        super().__init__(params, defaults)

        self.optim_U = optim_U
        self.optim_R = optim_R
        self.phi_U = phi_U
        self.phi_R = phi_R
        self.base_model = base_model

    def step(self, closure=None) -> Optional[float]:
        """Performs a single optimization step for both utility and risk parameters.

        Synchronizes the learning rate and weight decay from the wrapper parameter
        groups (which are updated by the Trainer scheduler) to the internal optimizers
        before taking a step.

        Args:
            closure (callable, optional): A closure that re-evaluates the model
                and returns the loss.

        Returns:
            The loss value if closure is provided, otherwise None.
        """
        # Sync hyperparameters (e.g. learning rate) from the scheduler wrapper group
        for g_wrapper in self.param_groups:
            for g_U, g_R in zip(self.optim_U.param_groups, self.optim_R.param_groups):
                g_U["lr"] = g_wrapper["lr"]
                g_R["lr"] = g_wrapper["lr"]
                if "weight_decay" in g_wrapper:
                    g_U["weight_decay"] = g_wrapper["weight_decay"]
                    g_R["weight_decay"] = g_wrapper["weight_decay"]

        loss = None
        if closure is not None:
            loss = closure()

        self.optim_U.step()
        self.optim_R.step()
        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Clears the gradients of both utility and risk buffers, and the base model."""
        self.optim_U.zero_grad(set_to_none=set_to_none)
        self.optim_R.zero_grad(set_to_none=set_to_none)
        self.base_model.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        """Returns the state of the optimizer as a dictionary."""
        return {
            "optim_U": self.optim_U.state_dict(),
            "optim_R": self.optim_R.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        """Loads the optimizer state.

        Args:
            state_dict: Optimizer state.
        """
        if "optim_U" in state_dict and "optim_R" in state_dict:
            self.optim_U.load_state_dict(state_dict["optim_U"])
            self.optim_R.load_state_dict(state_dict["optim_R"])
        else:
            super().load_state_dict(state_dict)


class RAPTrainer(Trainer):
    """Trainer wrapper subclass for Federated Risk-Aware Projection (RAP-FL).
    
    This trainer intercepts the training step to implement single-pass local training.
    It splits each sample loss based on its 'risk_score' and performs a double
    backward pass, accumulating gradients onto distinct utility and risk parameters
    without updating the frozen base model weights.
    
    Attributes:
        phi_t (Dict[str, torch.Tensor]): Cloned copy of the initial trainable weights.
        phi_U (Dict[str, torch.Tensor]): Trainable utility weight buffer.
        phi_R (Dict[str, torch.Tensor]): Trainable risk weight buffer.
        delta_U (Dict[str, torch.Tensor]): Calculated utility updates (phi_U - phi_t).
        delta_R (Dict[str, torch.Tensor]): Calculated risk updates (phi_R - phi_t).
    """

    def __init__(self, *args, **kwargs):
        """Initializes the RAPTrainer."""
        super().__init__(*args, **kwargs)
        self.phi_t: Optional[Dict[str, torch.Tensor]] = None
        self.phi_U: Optional[Dict[str, torch.Tensor]] = None
        self.phi_R: Optional[Dict[str, torch.Tensor]] = None
        self.delta_U: Optional[Dict[str, torch.Tensor]] = None
        self.delta_R: Optional[Dict[str, torch.Tensor]] = None

        # Validate that the train dataset contains the required risk_score column
        if self.train_dataset is not None:
            if hasattr(self.train_dataset, "column_names"):
                if "risk_score" not in self.train_dataset.column_names:
                    raise ValueError(
                        "The train_dataset must contain a 'risk_score' column "
                        "for RAPTrainer training."
                    )

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        """Ensures 'risk_score' is not removed by Hugging Face's column filtering."""
        if not self.args.remove_unused_columns:
            return dataset

        # Populate signature columns if not already done by Trainer
        if hasattr(self, "_set_signature_columns_if_needed"):
            self._set_signature_columns_if_needed()

        if hasattr(self, "_signature_columns") and self._signature_columns is not None:
            if "risk_score" not in self._signature_columns:
                self._signature_columns = list(self._signature_columns) + ["risk_score"]

        return super()._remove_unused_columns(dataset, description=description)

    def get_optimizer_class(self, optim_name: str) -> Type[Optimizer]:
        """Resolves the optimizer class based on the name in TrainingArguments.

        Fallback to torch.optim.AdamW when CUDA is not available or if bitsandbytes
        is not installed.

        Args:
            optim_name: Name of the optimizer from TrainingArguments.

        Returns:
            The resolved Optimizer class type.
        """
        optim_name = optim_name.lower() if optim_name else "adamw_torch"
        cuda_available = torch.cuda.is_available()

        try:
            import bitsandbytes as bnb
            has_bnb = True
        except ImportError:
            has_bnb = False

        if "paged_adamw_8bit" in optim_name:
            if has_bnb and cuda_available:
                return bnb.optim.PagedAdamW8bit
            return torch.optim.AdamW
        elif "adamw_bnb_8bit" in optim_name or "bnb_8bit" in optim_name:
            if has_bnb and cuda_available:
                return bnb.optim.AdamW8bit
            return torch.optim.AdamW
        elif "adamw" in optim_name:
            return torch.optim.AdamW
        elif "sgd" in optim_name:
            return torch.optim.SGD
        else:
            return torch.optim.AdamW

    def create_optimizer(self) -> Optimizer:
        """Sets up the RAPOptimizer wrapper and initial parameter buffers."""
        if self.optimizer is None:
            # 1. Capture the initial state phi_t (only for trainable parameters)
            self.phi_t = {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }

            # 2. Instantiate phi_U and phi_R buffers
            self.phi_U = {
                name: param.clone().detach().requires_grad_(True)
                for name, param in self.phi_t.items()
            }
            self.phi_R = {
                name: param.clone().detach().requires_grad_(True)
                for name, param in self.phi_t.items()
            }

            # 3. Resolve optimizer class and arguments
            optim_name = getattr(self.args, "optim", "adamw_torch")
            optimizer_cls = self.get_optimizer_class(optim_name)

            optimizer_kwargs = {
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            }

            # Inspect signature to add Adam-specific settings if applicable
            sig = inspect.signature(optimizer_cls)
            if "betas" in sig.parameters:
                optimizer_kwargs["betas"] = (self.args.adam_beta1, self.args.adam_beta2)
            if "eps" in sig.parameters:
                optimizer_kwargs["eps"] = self.args.adam_epsilon

            # 4. Instantiate inner optimizers
            optim_U = optimizer_cls(self.phi_U.values(), **optimizer_kwargs)
            optim_R = optimizer_cls(self.phi_R.values(), **optimizer_kwargs)

            # 5. Create RAPOptimizer wrapper
            self.optimizer = RAPOptimizer(
                optim_U=optim_U,
                optim_R=optim_R,
                phi_U=self.phi_U,
                phi_R=self.phi_R,
                base_model=self.model,
            )

        return self.optimizer

    def compute_per_sample_loss(
        self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], outputs: Any
    ) -> torch.Tensor:
        """Computes the token-level cross-entropy loss per sample.

        Args:
            model: The training model.
            inputs: Dict of inputs. Must contain "labels".
            outputs: Outputs of the model forward pass containing "logits".

        Returns:
            A 1D tensor containing the average sequence loss per sample.
        """
        logits = outputs.logits

        if "labels" not in inputs:
            raise ValueError("Inputs must contain 'labels' for calculating language modeling loss.")

        labels = inputs["labels"]

        # Shift logits and labels for Causal LM sequence training
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Token-level CrossEntropyLoss (no reduction to keep per-token loss values)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss_tokens = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        loss_tokens = loss_tokens.view(shift_labels.size(0), -1)

        # Mask out padding labels (-100 is HuggingFace's standard masking value)
        mask = (shift_labels != -100).float()
        per_sample_loss = (loss_tokens * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        return per_sample_loss

    def training_step(
        self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        """Performs a custom training step with double backward pass.

        Divides gradient updates into utility and risk directions using the
        sample-level 'risk_score' weights.

        Args:
            model: The model to train.
            inputs: Batch of inputs.

        Returns:
            A detached combined loss tensor for tracking and logging.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if "risk_score" not in inputs:
            raise ValueError("The input batch must contain 'risk_score' for RAPTrainer.")

        # Extract sample risk scores and split into Utility and Risk weights
        risk_scores = inputs.pop("risk_score").float()
        w_U = 1.0 - risk_scores
        w_R = risk_scores

        # Support autocast/mixed precision via Trainer context manager
        if hasattr(self, "compute_loss_context_manager"):
            context_manager = self.compute_loss_context_manager()
        else:
            from contextlib import nullcontext
            context_manager = nullcontext()

        with context_manager:
            outputs = model(**inputs)
            per_sample_loss = self.compute_per_sample_loss(model, inputs, outputs)

            # Compute weighted objectives
            loss_U = (per_sample_loss * w_U).mean()
            loss_R = (per_sample_loss * w_R).mean()

        # Scale losses for gradient accumulation
        grad_accum_steps = self.args.gradient_accumulation_steps
        if grad_accum_steps > 1:
            loss_U = loss_U / grad_accum_steps
            loss_R = loss_R / grad_accum_steps

        # --- DUAL BACKWARD PASS ---

        # A. Utility updates accumulation
        model.zero_grad()
        self.accelerator.backward(loss_U, retain_graph=True)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                clean_name = name.replace("module.", "")
                if clean_name in self.phi_U:
                    if self.phi_U[clean_name].grad is None:
                        self.phi_U[clean_name].grad = param.grad.clone()
                    else:
                        self.phi_U[clean_name].grad += param.grad.clone()

        # B. Risk updates accumulation
        model.zero_grad()
        self.accelerator.backward(loss_R)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                clean_name = name.replace("module.", "")
                if clean_name in self.phi_R:
                    if self.phi_R[clean_name].grad is None:
                        self.phi_R[clean_name].grad = param.grad.clone()
                    else:
                        self.phi_R[clean_name].grad += param.grad.clone()

        # Sabotage base model update: keep base model gradients zeroed
        model.zero_grad()

        # Return total scaled loss for logging
        return (loss_U + loss_R).detach()

    def train(self, *args, **kwargs) -> Any:
        """Triggers local training and compiles final parameter update deltas."""
        train_result = super().train(*args, **kwargs)

        # Calculate final delta_U and delta_R upon completion
        if self.phi_t is not None:
            self.delta_U = {}
            self.delta_R = {}
            for name in self.phi_t.keys():
                cpu_phi_t = self.phi_t[name].detach().cpu()
                self.delta_U[name] = self.phi_U[name].detach().cpu() - cpu_phi_t
                self.delta_R[name] = self.phi_R[name].detach().cpu() - cpu_phi_t

        return train_result
