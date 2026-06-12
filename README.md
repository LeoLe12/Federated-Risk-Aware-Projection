# 🛡️ Federated Risk-Aware Projection (RAP-FL)
RAP-FL is a Federated Learning (FL) library/wrapper designed to implement **Risk-Aware Projection** for Open-Weight Large Language Models (LLMs). It seamlessly bridges Hugging Face's `Trainer` (client-side) and the Flower (`flwr`) framework (server-side) to allow decentralized training of models while separating utility-maximizing updates from risky/hazardous capability updates.
---
## 💡 Abstract
In Federated Learning of LLMs, clients train on diverse, decentralized datasets. However, these datasets may contain mixed content—combining safe utility data with risky, dangerous, or hazardous knowledge (e.g., biosecurity threats, cyberattack vectors). 
**RAP-FL** resolves this by executing a dual-objective training loop:
1. **Client-Side Dual updates**: Using a single-pass dual-backward training process, each client computes separate utility gradients ($g^U$) and risk gradients ($g^R$) based on sample-level threat probability weights (`risk_score`).
2. **Server-Side Orthogonal Projection**: The server aggregates utility and risk updates independently to build global direction compasses. It then sterilizes client updates by projecting out any component aligned with the global risk direction before updating the global parameters. An optional utility gate salvages safe capabilities from threat-related updates.
---
## ⚙️ Installation
You can install the RAP-FL library directly from the GitHub repository:
```bash
pip install git+https://github.com/LeoLe12/Federated-Risk-Aware-Projection.git
```
---
## 🚀 Quick Start / Usage Example
Here is a complete, minimal example demonstrating how to wrap client training with `RAPTrainer`, define a custom `RAPFlowerClient`, set up the `RAPStrategy` on the server, and start a Flower simulation.
### 1. Client-Side: `RAPFlowerClient`
The client subclass wraps Hugging Face's model and uses `RAPTrainer` to compute dual updates. We concatenate the utility update ($\Delta^U$) and risk update ($\Delta^R$) before sending them back to the server.
```python
import torch
import flwr as fl
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from rap_fl.client.trainer import RAPTrainer
from rap_fl.server.utils import ndarrays_to_peft_dict, peft_dict_to_ndarrays
class RAPFlowerClient(fl.client.NumPyClient):
    """Flower client that performs RAP-FL client training."""
    def __init__(self, model, train_dataset, tokenizer, keys, training_args):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.keys = keys
        self.training_args = training_args
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
    def fit(self, parameters, config):
        # 1. Update the local model's trainable parameters with received global weights
        peft_dict = ndarrays_to_peft_dict(parameters, self.keys)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in peft_dict:
                    param.copy_(peft_dict[name])
        # 2. Instantiate RAPTrainer
        trainer = RAPTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
        )
        # 3. Train (executes dual backward passes and optimizer steps on buffers)
        trainer.train()
        # 4. Extract delta_U and delta_R updates (moved to CPU to free VRAM)
        delta_U_ndarrays = peft_dict_to_ndarrays(trainer.delta_U, self.keys)
        delta_R_ndarrays = peft_dict_to_ndarrays(trainer.delta_R, self.keys)
        # 5. Concatenate both updates into a single list of NumPy arrays (protocol constraint)
        concatenated_updates = delta_U_ndarrays + delta_R_ndarrays
        return concatenated_updates, len(self.train_dataset), {}
    def evaluate(self, parameters, config):
        # Standard evaluate implementation
        return 0.0, len(self.train_dataset), {}
```
### 2. Server-Side: `RAPStrategy` & Simulation
The server runs the custom `RAPStrategy` to sanitize updates. The strategy requires a list of reference parameter `keys` to align the 1D parameter lists back to PyTorch state dict structures.
```python
import flwr as fl
from rap_fl.server.strategy import RAPStrategy
from rap_fl.server.utils import peft_dict_to_ndarrays
# 1. Select the parameter names that require gradients (trainable adapters/LoRA weights)
keys = [name for name, param in model.named_parameters() if param.requires_grad]
# 2. Convert initial global weights to a list of NumPy arrays
initial_weights = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
initial_ndarrays = peft_dict_to_ndarrays(initial_weights, keys)
initial_parameters = fl.common.ndarrays_to_parameters(initial_ndarrays)
# 3. Configure the strategy with the math parameters
strategy = RAPStrategy(
    keys=keys,
    lambda_t=1.0,           # Suppression strength on utility
    mu_t=1.0,               # Suppression strength on risk
    gamma_t=0.5,            # Salvage strength on risk
    initial_parameters=initial_parameters,
)
# 4. Client spawning function
def client_fn(cid: str) -> fl.client.Client:
    # Obtain client dataset slices containing the 'risk_score' column
    client_dataset = get_client_dataset(cid)  # Custom dataset loader
    training_args = TrainingArguments(
        output_dir="./tmp_output",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        optim="adamw_torch",
        report_to="none",
        disable_tqdm=True,
    )
    return RAPFlowerClient(
        model=model,
        train_dataset=client_dataset,
        tokenizer=tokenizer,
        keys=keys,
        training_args=training_args
    ).to_client()
# 5. Launch Simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
```
---
## 📂 Project Structure
```text
rap_fl/
├── client/
│   ├── __init__.py
│   └── trainer.py          # RAPTrainer & RAPOptimizer subclasses
└── server/
    ├── __init__.py
    ├── strategy.py         # Custom RAPStrategy Flower Strategy
    └── utils.py            # NumPy-PyTorch bridges & projection math
```
---
## 🛠️ Requirements
* **PyTorch** ($\ge$ 2.4)
* **Hugging Face Transformers**
* **PEFT** (Parameter-Efficient Fine-Tuning)
* **Flower (`flwr`)** ($\ge$ 1.0)
* **NumPy**
