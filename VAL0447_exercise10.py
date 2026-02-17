import numpy as np
import pandas as pd
import random
import os
from copy import deepcopy

def sphere(params):
    return np.sum(np.array(params)**2)
def schwefel(params):
    params = np.array(params)
    d = len(params)
    return 418.9829 * d - np.sum(params * np.sin(np.sqrt(np.abs(params))))
def rosenbrock(params):
    params = np.array(params)
    return np.sum(100.0 * (params[1:] - params[:-1]**2.0)**2.0 + (1 - params[:-1])**2.0)
def rastrigin(params):
    params = np.array(params)
    return 10 * len(params) + np.sum(params**2 - 10 * np.cos(2 * np.pi * params))
def griewank(params):
    params = np.array(params)
    sum_sq = np.sum(params**2)
    prod_cos = np.prod(np.cos(params / np.sqrt(np.arange(1, len(params) + 1))))
    return sum_sq / 4000 - prod_cos + 1
def levy(params):
    params = np.array(params)
    w = 1 + (params - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2_sum = np.sum((w[:-1]-1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1]-1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2_sum + term3
def michalewicz(params, m=10):
    params = np.array(params)
    d_indices = np.arange(1, len(params) + 1)
    return -np.sum(np.sin(params) * (np.sin(d_indices * params**2 / np.pi))**(2 * m))
def zakharov(params):
    params = np.array(params)
    d_indices = np.arange(1, len(params) + 1)
    sum1 = np.sum(params**2)
    sum2 = np.sum(0.5 * d_indices * params)
    return sum1 + sum2**2 + sum2**4
def ackley(params, a=20, b=0.2, c=2 * np.pi):
    params = np.array(params)
    d = len(params)
    sum1 = -a * np.exp(-b * np.sqrt(np.sum(params**2) / d))
    sum2 = -np.exp(np.sum(np.cos(c * params)) / d)
    return sum1 + sum2 + a + np.e


DIMS = 30
POP_SIZE = 30
MAX_OFE = 3000
NUM_EXPERIMENTS = 30

def ensure_bounds(vec, search_range):
    """Zajistí, že parametry nepřekročí definované hranice."""
    lower_bound, upper_bound = search_range
    return np.clip(vec, lower_bound, upper_bound)


#Teaching-Learning Based Optimization
def tlbo_run(obj_func, num_dimensions, search_range, max_ofe, pop_size):
    
    # Inicializace počítadla vyhodnocení účelové funkce (Objective Function Evaluations)
    ofe_count = 0
    lower, upper = search_range

    # Inicializace populace
    population_params = np.random.uniform(lower, upper, (pop_size, num_dimensions))
    
    # Ohodnocení (zjištění "známek") každého studenta pomocí účelové funkce
    population_f = np.array([obj_func(p) for p in population_params])
    
    # Za počáteční inicializaci jsme spotřebovali 'pop_size' vyhodnocení
    ofe_count += pop_size 

    # Hlavní cyklus algoritmu: běží, dokud nevyčerpáme rozpočet 'max_ofe'
    while ofe_count < max_ofe:
        
        # Fáze Učitele (Teacher Phase)
        # Cíl: Posunout celou třídu směrem k nejlepšímu studentovi (učiteli)
        
        # Najdeme index nejlepšího studenta (toho s nejnižší fitness - minimalizujeme)
        teacher_idx = np.argmin(population_f)
        # Uložíme si jeho parametry (pozici) - toto je Učitel
        teacher_params = population_params[teacher_idx]
        
        #Vypočítáme průměrnou pozici všech studentů
        mean_params = np.mean(population_params, axis=0)

        # Projdeme každého studenta 'i' ve třídě
        for i in range(pop_size):
            if ofe_count >= max_ofe: break
            
            # Náhodný 'Teaching Factor' (TF), který určuje sílu vlivu průměru 
            # Může být 1 nebo 2
            TF = random.randint(1, 2) 
            # Náhodný vektor 'r' (pro každý rozměr) mezi 0 a 1
            r = np.random.rand(num_dimensions)
            
            # Výpočet vektoru posunu dle vzorce TLBO
            # Student 'i' se posouvá od průměru třídy (TF * mean_params) směrem k učiteli (teacher_params)
            difference = r * (teacher_params - TF * mean_params)
            
            # Výpočet nové, potenciálně lepší pozice studenta 
            new_params = population_params[i] + difference
            new_params = ensure_bounds(new_params, search_range)
            
            # Ohodnocení nové pozice studenta
            new_f = obj_func(new_params)
            # Započítání vyhodnocení
            ofe_count += 1

            # Greedy výběr
            # Pokud je nová pozice studenta lepší (nižší fitness) než jeho stará
            if new_f < population_f[i]:
                # přijme ji. Aktualizujeme jeho pozici i fitness.
                population_params[i] = new_params
                population_f[i] = new_f
                
        # Kontrola OFE i po vnitřní smyčce (pro přerušení 'while' cyklu)
        if ofe_count >= max_ofe: break 

        # Fáze Studenta (Learner Phase)
        # Cíl: Studenti se učí jeden od druhého náhodnou interakcí
        
        # Projdeme (opět) každého studenta 'i'
        for i in range(pop_size):
            if ofe_count >= max_ofe: break
            
            # Náhodně vybereme jiného studenta 'j' pro interakci
            # (Zajistíme, aby 'j' nebylo stejné jako 'i')
            j = random.choice([k for k in range(pop_size) if k != i])
            
            # Náhodný vektor 'r' (pro každý rozměr) mezi 0 a 1
            r = np.random.rand(num_dimensions) 
            
            # Porovnání studentů 'i' a 'j'
            if population_f[i] < population_f[j]:
                # Pokud je 'i' lepší než 'j', posune se 'i' SMĚREM OD 'j'
                # (Cílem je prozkoumat jiný prostor)
                new_params = population_params[i] + r * (population_params[i] - population_params[j])
            else:
                # Pokud je 'j' lepší než 'i', posune se 'i' SMĚREM K 'j'
                # (Cílem je poučit se od lepšího spolužáka)
                new_params = population_params[i] + r * (population_params[j] - population_params[i])
            
            # Ošetření hranic pro novou pozici
            new_params = ensure_bounds(new_params, search_range)
            # Ohodnocení nové pozice
            new_f = obj_func(new_params)
            # Započítání vyhodnocení
            ofe_count += 1

            # Pokud je nová pozice lepší než ta stará
            if new_f < population_f[i]:
                # student ji přijme.
                population_params[i] = new_params
                population_f[i] = new_f
                
        if ofe_count >= max_ofe: break
            
    # Po vyčerpání OFE vrátíme nejlepší fitness (nejnižší hodnotu) z celé populace
    return np.min(population_f)

def de_run(obj_func, num_dimensions, search_range, max_ofe, pop_size, F=0.5, CR=0.5):
    ofe_count = 0
    lower, upper = search_range

    population_params = np.random.uniform(lower, upper, (pop_size, num_dimensions))
    population_f = np.full(pop_size, np.inf)
    for i in range(pop_size):
        if ofe_count >= max_ofe: break
        population_f[i] = obj_func(population_params[i])
        ofe_count += 1
    if ofe_count == 0: return np.inf

    gBest_f = np.min(population_f)

    while ofe_count < max_ofe:
        new_pop_params = population_params.copy()
        new_pop_f = population_f.copy()
        
        for i in range(pop_size):
            if ofe_count >= max_ofe: break

            candidate_indices = list(range(pop_size))
            candidate_indices.remove(i)
            idx_a, idx_b, idx_c = random.sample(candidate_indices, 3)

            params_a = population_params[idx_a]
            params_b = population_params[idx_b]
            params_c = population_params[idx_c]

            mutant_vector = params_c + F * (params_a - params_b)
            mutant_vector = ensure_bounds(mutant_vector, search_range)

            trial_vector = np.zeros(num_dimensions)
            j_rnd = np.random.randint(0, num_dimensions)

            for j in range(num_dimensions):
                if np.random.uniform() < CR or j == j_rnd:
                    trial_vector[j] = mutant_vector[j]
                else:
                    trial_vector[j] = population_params[i][j]
            
            f_u = obj_func(trial_vector)
            ofe_count += 1

            if f_u <= population_f[i]:
                new_pop_params[i] = trial_vector
                new_pop_f[i] = f_u
                if f_u < gBest_f:
                    gBest_f = f_u
        
        population_params = new_pop_params
        population_f = new_pop_f
            
    return gBest_f

### Particle Swarm Optimization (PSO) ###
def pso_run(obj_func, num_dimensions, search_range, max_ofe, pop_size, c1=2.0, c2=2.0):
    ofe_count = 0
    lower, upper = search_range
    
    params = np.random.uniform(lower, upper, (pop_size, num_dimensions))
    velocities = np.zeros((pop_size, num_dimensions))
    
    # Vyhodnocení s kontrolou OFE
    f_values = np.full(pop_size, np.inf)
    for i in range(pop_size):
        if ofe_count >= max_ofe: break
        f_values[i] = obj_func(params[i])
        ofe_count += 1
    if ofe_count == 0: return np.inf
    
    pBest_params = params.copy()
    pBest_f = f_values.copy()
    
    gBest_idx = np.argmin(f_values)
    gBest_params = params[gBest_idx].copy()
    gBest_f = f_values[gBest_idx]

    while ofe_count < max_ofe:
        r1 = np.random.rand(pop_size, num_dimensions)
        r2 = np.random.rand(pop_size, num_dimensions)
        
        cognitive = c1 * r1 * (pBest_params - params)
        social = c2 * r2 * (gBest_params - params)
        
        velocities = 0.7 * velocities + cognitive + social
        params += velocities
        params = ensure_bounds(params, search_range)
        
        # Vyhodnocení s přesnou kontrolou OFE
        for i in range(pop_size):
            if ofe_count >= max_ofe: break
            f_values[i] = obj_func(params[i])
            ofe_count += 1
            
            # Aktualizace pBest
            if f_values[i] < pBest_f[i]:
                pBest_f[i] = f_values[i]
                pBest_params[i] = params[i].copy()
                
                # Aktualizace gBest
                if pBest_f[i] < gBest_f:
                    gBest_f = pBest_f[i]
                    gBest_params = pBest_params[i].copy()
                    
        if ofe_count >= max_ofe: break
            
    return gBest_f

### Self-Organizing Migrating Algorithm (SOMA) ###
def soma_run(obj_func, num_dimensions, search_range, max_ofe, pop_size, path_length=3.0, step=0.11, prt=0.3):
    ofe_count = 0
    lower, upper = search_range

    # Inicializace populace
    population_params = np.random.uniform(lower, upper, (pop_size, num_dimensions))
    population_f = np.full(pop_size, np.inf)
    for i in range(pop_size):
        if ofe_count >= max_ofe: break
        population_f[i] = obj_func(population_params[i])
        ofe_count += 1
    if ofe_count == 0: return np.inf

    # Hlavní smyčka (migrace)
    while ofe_count < max_ofe:
        
        # Najdeme aktuálního leadera
        leader_idx = np.argmin(population_f)
        leader_params = population_params[leader_idx].copy()
        
        # Každý jedinec (kromě leadera) migruje
        for i in range(pop_size):
            if ofe_count >= max_ofe: break
            if i == leader_idx:
                continue

            # Nejlepší pozice nalezená na "cestě" tohoto jedince
            best_path_params = population_params[i].copy()
            best_path_f = population_f[i]
            
            # Vektor perturbace
            prt_vec = np.random.rand(num_dimensions) < prt
            
            current_step = step
            while current_step <= path_length:
                if ofe_count >= max_ofe: break

                # Maska pro PRT
                prt_mask = np.where(prt_vec, 1, 0)
                
                # Nová pozice na cestě k leaderovi
                new_params = population_params[i] + (leader_params - population_params[i]) * current_step * prt_mask
                new_params = ensure_bounds(new_params, search_range)
                
                new_f = obj_func(new_params)
                ofe_count += 1
                
                # Aktualizace nejlepší pozice na této cestě
                if new_f < best_path_f:
                    best_path_f = new_f
                    best_path_params = new_params.copy()
                    
                current_step += step

            # Po skončení cesty (path_length) aktualizujeme jedince v populaci
            if best_path_f < population_f[i]:
                population_f[i] = best_path_f
                population_params[i] = best_path_params

        if ofe_count >= max_ofe: break
            
    return np.min(population_f)


### Firefly Algorithm (FA) - Správná verze z Cv. 9 ###
# Tato třída je vyžadována pro firefly_algorithm_run níže
class Firefly:
    """Reprezentuje jedince (světlušku) v populaci."""
    def __init__(self, num_dimensions, search_range):
        lower, upper = search_range
        self.params = np.random.uniform(lower, upper, num_dimensions)
        self.f = np.inf 

def firefly_algorithm_run(obj_func, num_dimensions, search_range, max_ofe, pop_size, alpha=0.3, beta0=1.0):
    """
    Implementace FA z Cvičení 9 (generační, s pohybem nejlepšího),
    upravená pro řízení pomocí max_ofe.
    """
    ofe_count = 0
    lower, upper = search_range

    # Vytvoříme populaci světlušek a spočítáme OFE
    fireflies = [Firefly(num_dimensions, search_range) for _ in range(pop_size)]
    for ff in fireflies:
        if ofe_count >= max_ofe: break
        ff.f = obj_func(ff.params)
        ofe_count += 1
    if ofe_count == 0: return np.inf

    gBest_firefly = min(fireflies, key=lambda ff: ff.f)
    gBest_f = gBest_firefly.f 
    
    while ofe_count < max_ofe:
        
        alpha_damp = 0.99
        alpha *= alpha_damp
        
        fireflies_copy = deepcopy(fireflies)

        # Pohyb světlušek (porovnání každé s každou)
        for i in range(pop_size):
            for j in range(pop_size):
                if i == j: continue 
                
                if fireflies_copy[j].f < fireflies_copy[i].f:
                    r_euklid = np.linalg.norm(fireflies[i].params - fireflies[j].params)
                    r = max(r_euklid, 1e-10) 
                    beta = beta0 / (1 + r) 
                    
                    attraction_term = beta * (fireflies_copy[j].params - fireflies_copy[i].params)
                    random_term = alpha * np.random.randn(num_dimensions)
                    
                    fireflies[i].params += attraction_term + random_term
                    fireflies[i].params = ensure_bounds(fireflies[i].params, search_range)
                    # Pozice se vyhodnotí až hromadně níže
        
        # Přepočítej fitness pro všechny světlušky (generační přístup)
        for ff in fireflies:
            if ofe_count >= max_ofe: break
            ff.f = obj_func(ff.params) 
            ofe_count += 1
        if ofe_count >= max_ofe: break
            
        current_best_idx = min(range(pop_size), key=lambda k: fireflies[k].f)
        
        # Náhodný pohyb nejlepší světlušky
        random_move_best = alpha * np.random.randn(num_dimensions)
        potential_new_pos = fireflies[current_best_idx].params + random_move_best
        potential_new_pos = ensure_bounds(potential_new_pos, search_range)
        
        if ofe_count >= max_ofe: break
        potential_new_f = obj_func(potential_new_pos)
        ofe_count += 1

        # Přijetí, pouze pokud je lepší
        if potential_new_f < fireflies[current_best_idx].f:
            fireflies[current_best_idx].params = potential_new_pos
            fireflies[current_best_idx].f = potential_new_f
            
        # Aktualizace globálně nejlepšího
        if fireflies[current_best_idx].f < gBest_f:
            gBest_f = fireflies[current_best_idx].f

    return gBest_f

if __name__ == "__main__":
    
    # Seznam testovacích funkcí
    function_data = [
        {"func": sphere, "name": "Sphere", "range": [-100, 100]},
        {"func": schwefel, "name": "Schwefel", "range": [-500, 500]},
        {"func": rosenbrock, "name": "Rosenbrock", "range": [-30, 30]},
        {"func": rastrigin, "name": "Rastigin", "range": [-5.12, 5.12]},
        {"func": griewank, "name": "Griewank", "range": [-600, 600]},
        {"func": levy, "name": "Levy", "range": [-10, 10]},
        {"func": michalewicz, "name": "Michalewicz", "range": [0, np.pi]},
        {"func": zakharov, "name": "Zakharov", "range": [-5, 10]},
        {"func": ackley, "name": "Ackley", "range": [-32.768, 32.768]}
    ]

    # Název souboru pro ukládání
    output_filename = 'All_results.xlsx'
    
    excel_writer = pd.ExcelWriter(output_filename, engine='openpyxl')
    
    # Slovník algoritmů pro snadné spouštění
    algorithms = {
            "DE": de_run,
            "PSO": pso_run,
            "SOMA": soma_run,
            "FA": firefly_algorithm_run,
            "TLBO": tlbo_run
        }
    
    # Hlavní smyčka přes všechny testovací funkce
    for data in function_data:
        func_name = data["name"]
        print(f"--- Probíhá testování pro funkci: {func_name} ---")
        
        results_data = {alg_name: [] for alg_name in algorithms.keys()}
        
        for i in range(NUM_EXPERIMENTS):
            print(f"   Experiment {i+1}/{NUM_EXPERIMENTS}...")
            for alg_name, alg_func in algorithms.items():
                best_val = alg_func(
                    obj_func=data["func"],
                    num_dimensions=DIMS,
                    search_range=data["range"],
                    max_ofe=MAX_OFE,
                    pop_size=POP_SIZE
                )
                results_data[alg_name].append(best_val)
        
        df = pd.DataFrame(results_data)
        
        mean_vals = df.mean()
        std_vals = df.std()
        
        summary_df = pd.DataFrame(
            [f"{mean:.4e} ± {std:.4e}" for mean, std in zip(mean_vals, std_vals)],
            index=df.columns,
            columns=["Mean ± Std. Dev."]
        ).T
        
        df.index = [f"Experiment {i+1}" for i in range(NUM_EXPERIMENTS)]
        
        final_df = pd.concat([df, summary_df])

        final_df.to_excel(excel_writer, sheet_name=func_name)

    excel_writer.close()
    
    print(f"\n--- Experimenty dokončeny. Výsledky byly uloženy do souboru '{os.path.abspath(output_filename)}' ---")