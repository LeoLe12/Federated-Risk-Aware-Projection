from typing import List, Dict, Tuple
import torch
from torch import Tensor


## -- FUNZIONE DI FEDAVG
def fed_avg_math(weights_list: List[Dict[str, torch.Tensor]],
                 coefficients: List[float] = None) -> Dict[str, torch.Tensor]:

    num_clients = len(weights_list)
    if coefficients is None:
        coefficients = [1.0 / num_clients] * num_clients

    # Normalizza i coefficienti
    total_w = sum(coefficients)
    norm_coeffs = [c / total_w for c in coefficients]

    print(f"  FedAvg Coefficients: {[round(x, 3) for x in norm_coeffs]}")

    aggregated_weights = {}
    ref_keys = weights_list[0].keys()

    for key in ref_keys:
        # 1. Creiamo l'accumulatore vuoto usando lo STESSO DTYPE del primo client
        #    Se è float32, resta float32. Se è float16, resta float16.
        first_tensor = weights_list[0][key]
        weighted_sum = torch.zeros_like(first_tensor, device="cpu")

        for i, client_w in enumerate(weights_list):
            if key not in client_w: continue

            val = client_w[key].to("cpu")

            # Somma pesata
            weighted_sum += val * norm_coeffs[i]

        aggregated_weights[key] = weighted_sum

    return aggregated_weights



def peft_inner_product(dict_a: dict, dict_b: dict) -> float:
    """
    Calcola il prodotto interno di Frobenius tra due dizionari di pesi PEFT.

    Args:
        dict_a: Il primo aggiornamento (es. Delta^U_k,t)
        dict_b: Il secondo aggiornamento (es. G^R_t)
        le matrici devono essere delle stesse dimensioni.

    Returns:
        Il prodotto interno scalare, definito come
        <A, B> = Σ_i <A_i, B_i>_F
        ovvero la somma dei prodotti componente per componente corrispettivi di 2 matrici
    """
    inner_prod = 0.0
    for key in dict_a.keys():
        # Verifichiamo che la chiave esista in entrambi
        if key in dict_b:
            # torch.sum(A * B) è equivalente al prodotto di Frobenius tr(A^T B)
            inner_prod += torch.sum(dict_a[key] * dict_b[key]).item()

    return inner_prod


def project_onto(dict_v: dict, dict_g: dict) -> dict:
    """
    Calcola la proiezione ortogonale di V sulla direzione G: Proj_G(V)

    Args:
        dict_v: Il dizionario dei pesi da proiettare (es. il delta di un client).
        dict_g: Il dizionario dei pesi che fa da base per la proiezione (es. direzione globale di rischio).

    Returns:
        Un nuovo dizionario con i pesi proiettati, o un dizionario di zeri se la norma di dict_g è <= 0.
    """

    # 0. Check struttura tensori, devono essere uguali per la nostra funzione
    if set(dict_v.keys()) != set(dict_g.keys()):
        raise ValueError("PEFT tensors must have identical keys for projection.")

    # 1. Calcoliamo la norma al quadrato di G: <G, G>
    norm_sq_g = peft_inner_product(dict_g, dict_g)

    # 2. La condizione corretta del paper: se <G, G> non è > 0, restituiamo 0
    if norm_sq_g <= 0:
        return {k: torch.zeros_like(v) for k, v in dict_v.items()}

    # 3. Calcoliamo l'allineamento <V, G>
    inner_v_g = peft_inner_product(dict_v, dict_g)

    # 4. Calcoliamo il coefficiente scalare: <V, G> / <G, G>
    scalar_coef = inner_v_g / norm_sq_g

    # 5. Moltiplichiamo ogni tensore di G per lo scalare calcolato
    projected_dict = {}
    for key in dict_g.keys():
        if key in dict_v:
            projected_dict[key] = dict_g[key] * scalar_coef

    return projected_dict


def apply_risk_projection(delta_k: dict, G_R: dict, penalty_factor: float, G_U_for_gate: dict = None) -> dict:
    """
    Calcola l'aggiornamento sterilizzato e applica l'utility gate.

    Args:
        delta_k: Il delta originale del client (Delta_{k,t}^U o Delta_{k,t}^R).
        G_R: Direzione globale di rischio (G_t^R)
        G_U_for_gate: Direzione globale di utilità (G_t^U) (opzionale).
        penalty_factor: Il fattore di soppressione (lambda_t).

    Returns:
        Il dizionario dei pesi finali (U_{k,t} o R_{k,t}).
    """

    # 1. Calcolo della proiezione di Delta sulla direzione di rischio G_R
    proj = project_onto(delta_k, G_R)

    # 2. Sottrazione della proiezione
    # tilde{Delta} = Delta - penalty_factor * Proj_{G^R}(Delta)
    delta_tilde = {}
    for key in delta_k.keys():
        if key in proj:
            delta_tilde[key] = delta_k[key] - penalty_factor * proj[key]
        else:
            delta_tilde[key] = delta_k[key].clone()


    ## ==========================================
    # 3. UTILITY GATE (Applicato solo se G_U_for_gate è fornito)
    #lo applichiamo solo all'Utility Salvage from threat data e non al risk-suppressed update
    if G_U_for_gate is not None:
        # Calcoliamo l'allineamento con la direzione di utilità globale
        alignment_with_utility = peft_inner_product(delta_tilde, G_U_for_gate)

        # Se l'allineamento è <= 0, scartiamo l'aggiornamento (restituendo zeri)
        if alignment_with_utility <= 0:
            return {k: torch.zeros_like(v) for k, v in delta_tilde.items()}



    # Se il gate è superato (o commentato), restituiamo il delta pulito
    return delta_tilde




def final_aggregation_and_update(current_model: dict, U_list: list, R_list: list, a_list: list, gamma_t: float) -> tuple:
    """
    Esegue l'aggregazione finale (Riga 18) e aggiorna i pesi globali (Riga 19).

    Args:
        current_model: I pesi globali PEFT (LoRA) all'inizio del round t (phi_t nel paper).
        U_list: Lista dei dizionari U_{k,t} (aggiornamenti utili filtrati) per ogni client.
        R_list: Lista dei dizionari R_{k,t} (aggiornamenti recuperati filtrati) per ogni client.
        a_list: Lista dei pesi di aggregazione a_{k,t} per ogni client (es. n_k / N).
        gamma_t: Il coefficiente di salvataggio (gamma_t).

    Returns:
        next_model: Il nuovo dizionario dei pesi globali per il round t+1 (phi_{t+1} nel paper).
        Delta_t: Il delta globale applicato (utile per log/debug).
    """

    # Inizializziamo i dizionari per le somme
    sum_U = {k: torch.zeros_like(v) for k, v in current_model.items()}
    sum_R = {k: torch.zeros_like(v) for k, v in current_model.items()}

    # 1. Calcoliamo la somma pesata degli U_{k,t} e R_{k,t}
    for U_k, R_k, a_k in zip(U_list, R_list, a_list):
        for key in current_model.keys():
            if key in U_k:
                sum_U[key] += a_k * U_k[key]
            if key in R_k:
                sum_R[key] += a_k * R_k[key]

    # 2. Calcoliamo il Delta globale (Riga 18 dell'Algoritmo 1)
    # Delta_t = sum_k a_{k,t} U_{k,t} + gamma_t sum_k a_{k,t} R_{k,t}
    Delta_t = {}
    for key in current_model.keys():
        Delta_t[key] = sum_U[key] + gamma_t * sum_R[key]

    # 3. Aggiorniamo i pesi globali (Riga 19 dell'Algoritmo 1)
    # phi_{t+1} = phi_t + Delta_t
    next_model = {}
    for key in current_model.keys():
        next_model[key] = current_model[key] + Delta_t[key]

    return next_model, Delta_t





## --- FUNZIONE PER CHIMARE L'INTERO FLUSSO DI AGGREGAZIONE

def server_round_rap_fl(
    current_global_model: dict,
    list_delta_U: list,
    list_delta_R: list,
    client_weights: list,
    lambda_t: float,
    mu_t: float,
    gamma_t: float
) -> tuple:
    """
    Esegue un intero round di aggregazione RAP-FL lato server.

    Args:
        current_global_model: I pesi LoRA globali all'inizio del round (\phi_t).
        list_delta_U: Lista contenente i Delta^U inviati dai client.
        list_delta_R: Lista contenente i Delta^R inviati dai client.
        client_weights: Lista dei coefficienti di aggregazione (es. n_k / N).
        lambda_t: Fattore di penalità per la sterilizzazione dell'Utilità.
        mu_t: Fattore di penalità per la sterilizzazione del Rischio.
        gamma_t: Coefficiente di salvataggio (Utility Salvage) per il Rischio.

    Returns:
        new_global_model: Il modello globale aggiornato (\phi_{t+1}).
        global_delta: Lo spostamento netto applicato al modello (\Delta_t).
    """

    print("\n--- INIZIO AGGREGAZIONE SERVER RAP-FL ---")

    # ==========================================
    # STEP 1: Calcolo delle Direzioni Globali (Le "Bussole")
    # ==========================================
    print("1. Calcolo delle direzioni globali G^U e G^R...")

    # G_t^U = Somma pesata di tutti i Delta^U dei client
    G_U = fed_avg_math(weights_list=list_delta_U, coefficients=client_weights)

    # G_t^R = Somma pesata di tutti i Delta^R dei client
    G_R = fed_avg_math(weights_list=list_delta_R, coefficients=client_weights)


    # ==========================================
    # STEP 2: Proiezione e Sterilizzazione (Il Filtro)
    # ==========================================
    print("2. Sterilizzazione degli aggiornamenti locali...")

    clean_U_list = [] # Conterrà i \tilde{\Delta}_{k,t}^U
    clean_R_list = [] # Conterrà i \tilde{\Delta}_{k,t}^R

    # Iteriamo su ogni singolo client per pulire i suoi delta specifici
    for i in range(len(list_delta_U)):
        delta_u_client = list_delta_U[i]
        delta_r_client = list_delta_R[i]

        # A. Sterilizziamo l'utilità del client sottraendo la sua proiezione sul Rischio Globale (G_R)
        tilde_u = apply_risk_projection(
            delta_k=delta_u_client,
            G_R=G_R, # Proiettiamo sul RISCHIO
            penalty_factor=lambda_t,
            G_U_for_gate=None  # <- Gate disattivato
        )
        clean_U_list.append(tilde_u)

        # B. Sterilizziamo il rischio del client sottraendo la sua proiezione sull'Utilità Globale (G_U)
        # Questo serve per "salvare" le componenti buone rimaste intrappolate nel buffer di rischio
        tilde_r = apply_risk_projection(
            delta_k=delta_r_client,
            G_R=G_R, # Proiettiamo sul RISCHIO perchè dobbiamo sempre rimuovere quella componente
            penalty_factor=mu_t,
            G_U_for_gate=G_U   # <- Gate ATTIVATO!
        )
        clean_R_list.append(tilde_r)


    # ==========================================
    # STEP 3: Aggregazione Finale e Aggiornamento
    # ==========================================
    print("3. Generazione del Super-Delta e aggiornamento dei pesi globali...")

    # Usiamo i delta puliti (clean_U_list e clean_R_list) per creare \phi_{t+1}
    new_global_model, global_delta = final_aggregation_and_update(
        current_model=current_global_model,
        U_list=clean_U_list,
        R_list=clean_R_list,
        a_list=client_weights,
        gamma_t=gamma_t
    )

    print("--- ROUND SERVER COMPLETATO CON SUCCESSO ---\n")

    return new_global_model, global_delta