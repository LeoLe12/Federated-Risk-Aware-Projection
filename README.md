# 🛡️ Federated Risk-Aware Projection (RAP-FL)

RAP-FL is a Federated Learning (FL) library/wrapper designed to implement **Risk-Aware Projection** for Open-Weight Large Language Models (LLMs). It seamlessly bridges Hugging Face's `Trainer` (client-side) and the Flower (`flwr`) framework (server-side) to allow decentralized training of models while separating utility-maximizing updates from risky or hazardous capability updates.

---

## 💡 Abstract

In Federated Learning of LLMs, clients train on diverse, decentralized datasets. However, these datasets may contain mixed content—combining safe utility data with risky, dangerous, or hazardous knowledge (e.g., biosecurity threats, cyberattack vectors). 

**RAP-FL** resolves this by executing a dual-objective training loop:
1. **Client-Side Dual Updates**: Using a single-pass dual-backward training process, each client computes separate utility gradients ($g^U$) and risk gradients ($g^R$) based on sample-level threat probability weights (`risk_score`).
2. **Server-Side Orthogonal Projection**: The server aggregates utility and risk updates independently to build global direction compasses. It then sterilizes client updates by projecting out any component aligned with the global risk direction before updating the global parameters. An optional utility gate salvages safe capabilities from threat-related updates.

> [!NOTE]
> The `risk_score` column attached to each sample is computed using an external text classifier that estimates the probability of a sample belonging to a hazardous/threat class (`prob_class_1`). The reference implementations used in the original experiments are available at [Kyle1668's Hugging Face Repository](https://huggingface.co/Kyle1668/models?search=modern).

---

## ⚙️ Installation

Install the RAP-FL library and the Flower simulation engine:

```bash
pip install git+https://github.com/LeoLe12/Federated-Risk-Aware-Projection.git
pip install flwr
pip install -U "flwr[simulation]"
```

Quick import check:

```python
from rap_fl.client import RAPTrainer
from rap_fl.server import RAPStrategy

print("RAP-FL library loaded successfully.")
```

---

## 🚀 Quick Start / Usage Example

The complete working example is available in the [Rap-Wrapper-Notebook-Example.ipynb](rap_fl/Rap-Wrapper-Notebook-Example.ipynb) notebook. The following walkthrough extracts its key components.

### 1. 🔧 Setup Model & Tokenizer

The recommended setup uses 4-bit NF4 quantization via `bitsandbytes` and wraps the model with LoRA adapters via `peft`.

```python
import os
import gc
import torch
import flwr as fl
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from rap_fl.client import RAPTrainer
from rap_fl.server import RAPStrategy

model_id = "EleutherAI/deep-ignorance-pretraining-stage-unfiltered"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### 2. 📦 Prepare Datasets with `risk_score`

The `risk_score` column (sourced from an external threat classifier) must be present in every client dataset. Here it is obtained by renaming the `prob_class_1` column output from the classifier:

```python
base_path = '/path/to/your/datasets'

df_safe   = pd.read_csv(f'{base_path}/safe_dataset.csv').sample(n=100, random_state=42)
df_threat = pd.read_csv(f'{base_path}/threat_dataset.csv').sample(n=100, random_state=42)

def tokenize_function(example):
    prompt = (
        f"Question:\nTopic: {example['topic;']}\nSub-topic: {example['sub_topic']}\n\n"
        f"{example['message_1']}\n\nAnswer: "
    )
    tokenized = tokenizer(
        prompt + example['message_2'] + tokenizer.eos_token,
        truncation=True, padding="max_length", max_length=128
    )
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

dataset_safe   = Dataset.from_pandas(df_safe).map(tokenize_function).rename_column("prob_class_1", "risk_score")
dataset_threat = Dataset.from_pandas(df_threat).map(tokenize_function).rename_column("prob_class_1", "risk_score")

client_datasets = {"0": dataset_safe, "1": dataset_threat}
```

### 3. 🖥️ Client Definition: `RAPFlowerClient`

`RAPTrainer` handles the dual backward pass internally. The client only needs to extract `delta_U` and `delta_R` and concatenate them before returning to the server.

> [!IMPORTANT]
> Using memory-saving training options (`paged_adamw_8bit`, `gradient_checkpointing`, `per_device_train_batch_size=1` with `gradient_accumulation_steps=4`) is **strongly recommended** when running on consumer GPUs (e.g., 16 GB VRAM), since both utility and risk buffers ($\phi_U, \phi_R$) are cloned from the model, roughly doubling parameter memory.

```python
class RAPFlowerClient(fl.client.NumPyClient):
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
        print(f"\n--- Starting Training on Client {self.cid} ---")
        self.set_parameters(parameters)

        training_args = TrainingArguments(
            output_dir=f"./results_client_{self.cid}",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            max_steps=1,
            logging_steps=1,
            report_to="none"
        )

        trainer = RAPTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )

        trainer.train()

        # Extract dual updates as lists of NumPy arrays
        delta_U_ndarrays = [val.numpy() for val in trainer.delta_U.values()]
        delta_R_ndarrays = [val.numpy() for val in trainer.delta_R.values()]

        # Concatenate delta_U || delta_R before sending to server
        return delta_U_ndarrays + delta_R_ndarrays, len(self.train_dataset), {}
```

### 4. 🧮 Server Strategy: Zero-VRAM Key Extraction

The `RAPStrategy` requires the list of trainable parameter names (`keys`) to unpack and align client updates. The recommended approach uses PyTorch's **meta device** to extract the architecture **without allocating any GPU memory**:

```python
# Configure available GPUs (example: Kaggle dual-GPU setup)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

print("Extracting architecture (VRAM cost: 0 GB)...")
config = AutoConfig.from_pretrained(model_id)

with torch.device("meta"):
    meta_model = AutoModelForCausalLM.from_config(config)

lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
meta_peft_model = get_peft_model(meta_model, lora_cfg)

trainable_keys = [name for name, param in meta_peft_model.named_parameters() if param.requires_grad]
print(f"Found {len(trainable_keys)} LoRA keys. GPUs are free!")

# Instantiate strategy
strategy = RAPStrategy(
    keys=trainable_keys,
    lambda_t=0.3,           # Utility suppression strength
    mu_t=0.8,               # Risk suppression strength
    gamma_t=0.1,            # Salvage strength
    fraction_evaluate=0.0,  # Skip client-side evaluation
    min_evaluate_clients=0
)
```

### 5. 🌐 Launch Flower Simulation

```python
def client_fn(cid: str):
    return RAPFlowerClient(cid=cid, model=peft_model, train_dataset=client_datasets[cid])

print("\nStarting RAP-FL Federated Simulation...")
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
    # 1.0 ensures each client runs on a dedicated GPU without overlap
    client_resources={"num_cpus": 2, "num_gpus": 1.0}
)
print("Federated Training Completed!")
```

---

## 💻 Hardware Requirements & Resource Planning

### What makes RAP-FL more memory-intensive than standard FL

Each client materializes **two full copies** of the LoRA adapter parameters ($\phi_U$ and $\phi_R$) during training, in addition to the quantized base model. This is the key architectural constraint when planning resources.

### GPU Requirements

| Setup | Min. VRAM per GPU | Notes |
|---|---|---|
| 1 client, 1 GPU (sequential) | ~16 GB | Use `num_gpus: 0.4` with `paged_adamw_8bit` |
| 2 clients, 1 GPU (fractional) | ~20–24 GB | `num_gpus: 0.5`, clients run sequentially with memory reuse |
| 2 clients, 2 GPUs (recommended) | ~16 GB each | `num_gpus: 1.0` — each client gets a dedicated GPU |
| N clients, N GPUs | ~16 GB each | Full isolation, most stable for production runs |

> [!TIP]
> Setting `client_resources={"num_cpus": 2, "num_gpus": 1.0}` in `start_simulation` is the most stable configuration for Kaggle's dual T4 / P100 environments or Colab Pro+ (A100). Flower's simulator serializes client execution when only one GPU is available.

> [!WARNING]
> Setting `num_gpus` fractionally (e.g., `0.4`) allows multiple clients to share a GPU **sequentially** within Flower's virtual client engine, but the base model + $\phi_U$ + $\phi_R$ buffers can easily exceed 20 GB on a 7B+ model, causing OOM errors. Always pair fractional GPU settings with `gradient_checkpointing=True` and `paged_adamw_8bit`.

### CPU Requirements

| Role | Recommended CPUs |
|---|---|
| Per client process | 2 CPUs |
| Server aggregation | 2–4 CPUs (projection math runs on CPU) |
| Total for 2 clients | 6–8 CPUs (2 per client + server overhead) |

The server-side projection math (`peft_inner_product`, `project_onto`, `final_aggregation_and_update`) runs entirely on **CPU** using PyTorch tensors, so no GPU allocation is needed for the server.

### Scaling Table

| Num Clients | Min GPUs | Min VRAM Total | Min CPUs | Notes |
|---|---|---|---|---|
| 1 | 1 × 16 GB | 16 GB | 4 | Development / debugging |
| 2 | 1 × 24 GB **or** 2 × 16 GB | 24–32 GB | 6–8 | Reference notebook config |
| 4 | 2 × 24 GB | 48 GB | 10–12 | Rotate 2 clients per GPU |
| 8+ | 4+ GPUs | 64+ GB | 20+ | Multi-GPU cluster recommended |

---

## 📂 Project Structure

```text
rap_fl/
├── client/
│   ├── __init__.py
│   └── trainer.py          # RAPTrainer & RAPOptimizer subclasses
└── server/
    ├── __init__.py
    ├── strategy.py         # RAPStrategy (Flower FedAvg subclass)
    └── utils.py            # NumPy-PyTorch bridges & projection math
```

---

## 🛠️ Core Dependencies

| Package | Version | Role |
|---|---|---|
| `torch` | ≥ 2.4 | Core tensor operations & dual backward passes |
| `transformers` | ≥ 4.40 | Trainer base class & model loading |
| `peft` | ≥ 0.10 | LoRA adapter management |
| `flwr[simulation]` | ≥ 1.0 | Federated simulation engine |
| `bitsandbytes` | ≥ 0.43 | 4-bit quantization & `paged_adamw_8bit` |
| `datasets` | ≥ 2.0 | Dataset loading with `risk_score` column |
| `numpy` | ≥ 1.26 | NumPy-PyTorch bridge for parameter serialization |

## 👨‍💻 Author

**LeoLe12**
* GitHub: [@LeoLe12](https://github.com/LeoLe12)
* Project Repository: [Federated-Risk-Aware-Projection](https://github.com/LeoLe12/Federated-Risk-Aware-Projection)
