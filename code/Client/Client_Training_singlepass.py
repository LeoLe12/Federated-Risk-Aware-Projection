import torch
import math
import gc
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, set_seed
import bitsandbytes as bnb

# --- FUNZIONI HELPER PER IL LEARNING RATE ---
def get_cosine_lr(current_global_step, total_steps, initial_lr=5e-5, warmup_ratio=0.0):
    """Calcola il LR con Warmup e Cosine Decay basato sullo step globale."""
    warmup_steps = int(total_steps * warmup_ratio)
    if current_global_step < warmup_steps:
        return initial_lr * (float(current_global_step) / float(max(1, warmup_steps)))

    progress = float(current_global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = max(0.0, min(1.0, progress))
    return 0.5 * initial_lr * (1.0 + math.cos(math.pi * progress))

def update_optimizers_lr(optim_u, optim_r, current_lr):
    """Inietta il nuovo LR nei due ottimizzatori."""
    for param_group in optim_u.param_groups:
        param_group['lr'] = current_lr
    for param_group in optim_r.param_groups:
        param_group['lr'] = current_lr

# =====================================================================
# LA FUNZIONE PRINCIPALE DEL CLIENT
# =====================================================================
def train_client_rap_fl(
    client_dataset,
    base_model,
    tokenizer,
    current_global_checkpoint,
    #objective_type, # <--- AGGIUNTO QUI! ("utility" o "risk")
    global_step_counter,
    total_fl_steps,
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    weight_decay=0.03
):
    """
    Esegue l'addestramento locale di un client per uno specifico obiettivo (Utilità o Rischio).
    """

    # SETUP SEED
    set_seed(42)

    # 1. DATALOADER SETUP
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    client_dataloader = DataLoader(
        client_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    # 2. MODEL & BUFFER SETUP
    model = setup_lora(base_model, resume_from=current_global_checkpoint)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    # Estraiamo \phi_t (Punto di partenza congelato)
    #phi_t = {k: v.clone().detach().cpu() for k, v in model.named_parameters() if v.requires_grad}

    # 3. OPTIMIZER SETUP (8-bit)
    #optim = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay)

    # Contatore per l'accumulo interno
    step_local = 0

    print(f"   Inizio Loop di Addestramento ({len(client_dataloader)} batch)...")

    # ==========================================
    # 🌟 NOVITÀ: Creiamo la barra di avanzamento
    # ==========================================
    progress_bar = tqdm(
        client_dataloader, 
        desc="   [U & R SINGLE PASS]", 
        leave=False, # Impostato a False così scompare a fine training lasciando i log puliti
        bar_format="{l_bar}{bar:30}{r_bar}" # Formato compatto e pulito
    )

    # Estraiamo \phi_t (Punto di partenza congelato)
    phi_t = {k: v.clone().detach() for k, v in model.named_parameters() if v.requires_grad}
    
    # Creiamo i due Buffer addestrabili (\phi^U e \phi^R)
    phi_U = {k: v.clone().detach().requires_grad_(True) for k, v in phi_t.items()}
    phi_R = {k: v.clone().detach().requires_grad_(True) for k, v in phi_t.items()}
    
    # 3. OPTIMIZER SETUP (8-bit)
    optim_U = bnb.optim.PagedAdamW8bit(phi_U.values(), lr=learning_rate, weight_decay=weight_decay)
    optim_R = bnb.optim.PagedAdamW8bit(phi_R.values(), lr=learning_rate, weight_decay=weight_decay)
    
    # Contatore per l'accumulo interno
    acc_step_counter = 0
    
    print(f"   🚀 Inizio Loop di Addestramento ({len(client_dataloader)} batch)...")
    
    # 4. TRAINING LOOP
    for step, batch in enumerate(client_dataloader):
        
        batch = {k: v.to(model.device) for k, v in batch.items()}
        risk_scores = batch.pop("risk_score").float()
        
        w_U = 1.0 - risk_scores
        w_R = risk_scores
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # --- FORWARD PASS & SHIFT ---
            outputs = model(**batch)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            
            # --- LOSS COMPUTATION ---
            loss_fct = CrossEntropyLoss(reduction='none')
            loss_tokens = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_tokens = loss_tokens.view(shift_labels.size(0), -1) 
            
            mask = (shift_labels != -100).float()
            per_sample_loss = (loss_tokens * mask).sum(dim=1) / mask.sum(dim=1) 
            
            # Scaling per accumulo
            lossU = (per_sample_loss * w_U).mean() / gradient_accumulation_steps
            lossR = (per_sample_loss * w_R).mean() / gradient_accumulation_steps

        display_loss_U = lossU.item() * gradient_accumulation_steps
        display_loss_R = lossR.item() * gradient_accumulation_steps
        progress_bar.set_postfix({"loss_U": f"{display_loss_U:.3f}", "loss_R": f"{display_loss_R:.3f}"})
        
        # --- DOPPIO BACKWARD ---
        # A. Utilità
        model.zero_grad() 
        lossU.backward(retain_graph=True) 
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if phi_U[name].grad is None:
                    phi_U[name].grad = param.grad.clone()
                else:
                    phi_U[name].grad += param.grad.clone()
                    
        # B. Rischio
        model.zero_grad() 
        lossR.backward()  
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if phi_R[name].grad is None:
                    phi_R[name].grad = param.grad.clone()
                else:
                    phi_R[name].grad += param.grad.clone()
                    
        # --- AGGIORNAMENTO PESI (Step) ---
        acc_step_counter += 1
        if acc_step_counter % gradient_accumulation_steps == 0:
            
            # Aggiorniamo il LR per questo vero step di ottimizzazione
            current_lr = get_cosine_lr(global_step_counter, total_fl_steps, initial_lr=learning_rate)
            update_optimizers_lr(optim_U, optim_R, current_lr)
            
            optim_U.step()
            optim_R.step()
            
            optim_U.zero_grad()
            optim_R.zero_grad()
            
            # Avanziamo il contatore globale solo quando facciamo uno step vero
            global_step_counter += 1
            
        # Sabotaggio del modello base
        model.zero_grad()


    
    # 4. CALCOLO DEL DELTA (\Delta = \phi_finale - \phi_t)
    # =====================================================================
    print("   Calcolo dei Delta finali...")

    # Calcoliamo \Delta e lo spostiamo su CPU per non intasare la VRAM
    delta_U = {}
    delta_R = {}
    for k in phi_t.keys():
        cpu_phi_t = phi_t[k].detach().cpu()
        delta_U[k] = phi_U[k].detach().cpu() - cpu_phi_t
        delta_R[k] = phi_R[k].detach().cpu() - cpu_phi_t

    # 5. PULIZIA AGGRESSIVA
    print(f"   🧹 Pulizia memoria")
    if hasattr(model, "unload"):
        base_model = model.unload()
    
    if hasattr(base_model, "peft_config"):
        del base_model.peft_config
    if hasattr(base_model, "base_model_prepare_inputs_for_generation"):
        del base_model.base_model_prepare_inputs_for_generation
        
    del model, optim_U, optim_R, phi_t, phi_U, phi_R, progress_bar
    gc.collect()
    torch.cuda.empty_cache()
    
    return delta_U, delta_R, global_step_counter, base_model