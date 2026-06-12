# 🛡️ Federated Risk-Aware Projection (RAP-FL)

RAP-FL is a Federated Learning (FL) library/wrapper designed to implement **Risk-Aware Projection** for Open-Weight Large Language Models (LLMs). It seamlessly bridges Hugging Face's `Trainer` (client-side) and the Flower (`flwr`) framework (server-side) to allow decentralized training of models while separating utility-maximizing updates from risky or hazardous capability updates.

---

## 💡 Abstract

In Federated Learning of LLMs, clients train on diverse, decentralized datasets. However, these datasets may contain mixed content—combining safe utility data with risky, dangerous, or hazardous knowledge (e.g., biosecurity threats, cyberattack vectors). 

**RAP-FL** resolves this by executing a dual-objective training loop:
1. **Client-Side Dual Updates**: Using a single-pass dual-backward training process, each client computes separate utility gradients ($g^U$) and risk gradients ($g^R$) based on sample-level threat probability weights (`risk_score`).
   > [!NOTE]
   > The `risk_score` column is computed using a text classifier. The reference implementations of these classifiers were originally trained and evaluated using models from [Kyle1668's Hugging Face Repository](https://huggingface.co/Kyle1668/models?search=modern).
2. **Server-Side Orthogonal Projection**: The server aggregates utility and risk updates independently to build global direction compasses. It then sterilizes client updates by projecting out any component aligned with the global risk direction before updating the global parameters. An optional utility gate salvages safe capabilities from threat-related updates.

---

## ⚙️ Installation

You can install the RAP-FL library directly from the GitHub repository:

```bash
pip install git+https://github.com/LeoLe12/Federated-Risk-Aware-Projection.git
```

---

## 🚀 Quick Start / Usage Example

Below is the implementation and setup extracted from the example notebook [RAPWRAPPER.ipynb](file:///c:/Users/39366/Desktop/rap-fl-project/RAPWRAPPER.ipynb).

### 1. Setup Model, Tokenizer, and PEFT

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Load model with 4-bit quantization
model_id = "EleutherAI/deep-ignorance-pretraining-stage-unfiltered"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
base_model.config.pad_token_id = base_model.config.eos_token_id

# Wrap model with LoRA Config
peft_model = get_peft_model(base_model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
for name, param in peft_model.named_parameters():
    if "lora" in name: 
        param.requires_grad = True
```

### 2. Client-Side: `RAPFlowerClient`

```python
import flwr as fl
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from rap_fl.client import RAPTrainer

class RAPFlowerClient(fl.client.NumPyClient):
    """Flower client that performs RAP-FL client training."""

    def __init__(self, cid, model, train_dataset):
        self.cid = cid
        self.model = model
        self.train_dataset = train_dataset

    def set_parameters(self, parameters):
        trainable_keys = [k for k, v in self.model.named_parameters() if v.requires_grad]
        state_dict = self.model.state_dict()
        for k, v in zip(trainable_keys, parameters):
            state_dict[k] = torch.tensor(v).to(self.model.device).to(torch.float16)
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        print(f"\n🚀 Avvio addestramento Client {self.cid}...")
        self.set_parameters(parameters)

        # Standard Training Arguments
        training_args = TrainingArguments(
            output_dir=f"./results_client_{self.cid}",
            per_device_train_batch_size=4,
            max_steps=1,
            logging_steps=1,
            report_to="none"
        )

        # Initialize RAPTrainer wrapper
        trainer = RAPTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )

        # Train (calculates weighted utility/risk losses & performs dual backward passes)
        trainer.train()

        # Extract delta_U and delta_R updates as list of NumPy arrays
        delta_U_ndarrays = [val.numpy() for val in trainer.delta_U.values()]
        delta_R_ndarrays = [val.numpy() for val in trainer.delta_R.values()]

        # Concatenate delta_U and delta_R to submit to the server
        return delta_U_ndarrays + delta_R_ndarrays, len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.train_dataset), {}
```

### 3. Server-Side: `RAPStrategy` & Simulation

```python
from rap_fl.server import RAPStrategy

# 1. Extract the names of the trainable parameter keys
trainable_keys = [name for name, param in peft_model.named_parameters() if param.requires_grad]

# 2. Instantiate strategy with keys list and projection parameters
strategy = RAPStrategy(
    keys=trainable_keys,
    lambda_t=0.3,    # Utility suppression strength
    mu_t=0.8,        # Risk suppression strength
    gamma_t=0.1      # Salvage strength
)

# 3. Define the client spawner
def client_fn(cid: str):
    return RAPFlowerClient(cid=cid, model=peft_model, train_dataset=client_datasets[cid])

# 4. Start Flower Simulation
print("\n🌐 Avvio della Rete Federata RAP-FL...")
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
    client_resources={"num_cpus": 2, "num_gpus": 1}
)
print("🎉 Addestramento Federato completato!")
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
