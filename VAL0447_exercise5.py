import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy
import random


# ==============================================================================
# 2. DEFINICE OPTIMALIZAČNÍCH FUNKCÍ
# ==============================================================================

def sphere(params):
    """f(x) = sum(x_i^2)"""
    return np.sum(np.array(params)**2)

def schwefel(params):
    """f(x) = 418.9829d - sum(x_i*sin(sqrt(|x_i|)))"""
    params = np.array(params)
    d = len(params)
    sum_term = np.sum(params * np.sin(np.sqrt(np.abs(params))))
    return 418.9829 * d - sum_term

def rosenbrock(params, a=1):
    """f(x) = sum[100(x_{i+1}-x_i^2)^2 + (x_i-a)^2]"""
    params = np.array(params)
    total_sum = 0
    for i in range(len(params) - 1):
        term1 = 100 * (params[i+1] - params[i]**2)**2
        term2 = (params[i] - a)**2
        total_sum += term1 + term2
    return total_sum

def rastrigin(params, A=10):
    """f(x) = Ad + sum[x_i^2 - A*cos(2*pi*x_i)]"""
    params = np.array(params)
    d = len(params)
    sum_term = np.sum(params**2 - A * np.cos(2 * np.pi * params))
    return A * d + sum_term

def griewank(params):
    """f(x) = sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i))) + 1"""
    params = np.array(params)
    sum_sq = np.sum(params**2)
    prod_cos = np.prod(np.cos(params / np.sqrt(np.arange(1, len(params) + 1))))
    return sum_sq / 4000 - prod_cos + 1

def levy(params):
    """f(x) = sin²(πw₁) + Σ[(wᵢ−1)²(1+10sin²(πwᵢ+1))] + (w_d−1)²(1+sin²(2πw_d)), w = 1 + (x−1)/4"""
    params = np.array(params)
    w = 1 + (params - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2_sum = np.sum((w[:-1]-1)**2 * (1+10*np.sin(np.pi*w[:-1]+1)**2))
    term3 = (w[-1]-1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    return term1 + term2_sum + term3

def michalewicz(params, m=10):
    """f(x) = -Σ[sin(xᵢ) * (sin(i*xᵢ²/π))^(2m)]"""
    params = np.array(params)
    d_indices = np.arange(1, len(params) + 1)
    return -np.sum(np.sin(params) * (np.sin(d_indices * params**2 / np.pi))**(2 * m))

def zakharov(params):
    """f(x) = Σ(xᵢ²) + (Σ(0.5*i*xᵢ))² + (Σ(0.5*i*xᵢ))⁴"""
    params = np.array(params)
    d_indices = np.arange(1, len(params) + 1)
    sum_x_sq = np.sum(params**2)
    sum_weighted_x = np.sum(0.5 * d_indices * params)
    return sum_x_sq + sum_weighted_x**2 + sum_weighted_x**4

def ackley(params, a=20, b=0.2, c=2 * np.pi):
    """f(x) = -a·exp(-b·√(1/d·Σxᵢ²)) - exp(1/d·Σcos(c·xᵢ)) + a + e"""
    params = np.array(params)
    d = len(params)
    sum1 = np.sum(params**2)
    sum2 = np.sum(np.cos(c * params))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.e

 
NP = 100       # NP: Počet jedinců v populaci
F = 0.5        # F: Mutační konstanta
CR = 0.5       # CR: Pravděpodobnost křížení
G_maxim = 50   # G_maxim: Maximální počet generací


# ALGORITMUS DIFERENCIÁLNÍ EVOLUCE

def ensure_bounds(vec, search_range):
    """Zajistí, že parametry při mutaci nepřekročí definované hranice."""
    lower_bound, upper_bound = search_range
    return np.clip(vec, lower_bound, upper_bound)

def find_best(pop):
    """Najde nejlepšího jedince v populaci."""
    best_individual = min(pop, key=lambda ind: ind[1])
    return best_individual[0].copy(), best_individual[1]

def mutate(population, target_idx, mutation_factor, search_range, population_size):
    """Provede mutaci DE/rand/1/bin."""

    # Vybereme všechny možné indexy kromě aktuálního jedince (target_idx)
    candidate_indices = list(range(population_size))
    candidate_indices.remove(target_idx)

    # Náhodně vybereme 3 různé jedince z populace (r1, r2, r3)
    idx_a, idx_b, idx_c = random.sample(candidate_indices, 3)

    # Získáme jejich souřadnice
    params_a, _ = population[idx_a]
    params_b, _ = population[idx_b]
    params_c, _ = population[idx_c]

    mutant_vector = params_c + mutation_factor * (params_a - params_b)

    # Zajistíme, aby mutant zůstal v povoleném rozsahu
    return ensure_bounds(mutant_vector, search_range)


def crossover(x_i_params, mutation_vector, CR, num_dimensions):
    """Provede binomiální křížení."""

    trial_vector = np.zeros(num_dimensions)

    # Náhodně vybereme složku, která se vždy vezme z mutantního vektoru
    j_rnd = np.random.randint(0, num_dimensions)

    # Pro každou složku rozhodneme, zda vezmeme z mutantního vektoru nebo z původního jedince
    for j in range(num_dimensions):
        if np.random.uniform() < CR or j == j_rnd:
            # vezmeme z mutantního vektoru
            trial_vector[j] = mutation_vector[j]
        else:
            # vezmeme z původního jedince
            trial_vector[j] = x_i_params[j]

    return trial_vector


def differential_evolution_run(obj_func, num_dimensions, search_range, max_iterations, pop_size, F, CR):
    """Hlavní implementace Diferenciální Evoluce (DE/rand/1/bin)."""

    lower_bound, upper_bound = search_range

    # Inicializace populace
    pop = []
    for _ in range(pop_size):
        params = np.random.uniform(lower_bound, upper_bound, num_dimensions)
        fitness = obj_func(params)
        pop.append([params, fitness])

    # Najdeme nejlepšího jedince na startu
    best_params, best_value = find_best(pop)

    # Uložíme historii populace pro vizualizaci
    pop_history = [[ind[0] for ind in pop]]

    # Hlavní cyklus generací
    for _ in range(max_iterations):
        # deepcopy: nová generace začíná jako kopie té předchozí
        new_pop = deepcopy(pop)

        # projdeme všechny jedince v populaci
        for i in range(pop_size):
            x_i_params, f_x_i = pop[i]

            # Mutace → vytvoříme mutantní vektor
            mutation_vector = mutate(pop, i, F, search_range, pop_size)

            # Křížení → vytvoříme trial vektor kombinací původního a mutantního
            trial_vector = crossover(x_i_params, mutation_vector, CR, num_dimensions)

            f_u = obj_func(trial_vector)

            # pokud je trial lepší než původní jedinec, nahradíme ho
            if f_u <= f_x_i:
                new_pop[i] = [trial_vector.copy(), f_u]

        # Aktualizace populace
        pop = new_pop

        # Aktualizace nejlepšího řešení
        current_best_params, current_best_value = find_best(pop)
        if current_best_value < best_value:
            best_params = current_best_params
            best_value = current_best_value

        # Uložení historie generace pro vizualizaci
        pop_history.append([ind[0] for ind in pop])

    # Vracíme nejlepší řešení a historii
    return best_params.tolist(), best_value, pop_history

# VIZUALIZACE A ANIMACE

def animate_de_search(func, title, search_range, pop_history, interval=100):
    """Vytvoří a zobrazí animaci hledání optimálního řešení pomocí DE."""
    x_range = y_range = search_range
    x = np.linspace(x_range[0], x_range[1], 150)
    y = np.linspace(y_range[0], y_range[1], 150)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([X[i,j], Y[i,j]]) for j in range(150)] for i in range(150)])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    initial_pop = np.array(pop_history[0])
    scat = ax.scatter(initial_pop[:, 0], initial_pop[:, 1], c='red', s=25, edgecolor='white', linewidth=0.5, zorder=10)
    gen_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12,
                       bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    def update(frame):
        current_pop = np.array(pop_history[frame])
        scat.set_offsets(current_pop)
        gen_text.set_text(f'Generace: {frame}/{len(pop_history)-1}')
        return scat, gen_text

    ani = FuncAnimation(fig, update, frames=len(pop_history),
                        interval=interval, blit=True, repeat=True)
    ax.set_title(title)
    plt.show()


# HLAVNÍ SPOUŠTĚCÍ BLOK

if __name__ == "__main__":
    DIMENSIONS = 2  # Pro vizualizaci používáme 2D

    # Seznam optimalizačních problémů k vyřešení
    function_data = [
        {"func": sphere, "name": "Sphere", "range": [-5.12, 5.12]},
        {"func": ackley, "name": "Ackley", "range": [-32.768, 32.768]},
        {"func": rastrigin, "name": "Rastrigin", "range": [-5.12, 5.12]},
        {"func": schwefel, "name": "Schwefel", "range": [-500, 500]},
        {"func": rosenbrock, "name": "Rosenbrock", "range": [-5, 5]},
    ]

    for data in function_data:
        print(f"\n--- Spouštím optimalizaci pro funkci: {data['name']} ---")

        best_de, val_de, history = differential_evolution_run(
            obj_func=data["func"],
            num_dimensions=DIMENSIONS,
            search_range=data["range"],
            max_iterations=G_maxim,
            pop_size=NP,
            F=F,
            CR=CR
        )

        print(f"Výsledek: Nejlepší řešení: x={best_de[0]:.4f}, y={best_de[1]:.4f}, Hodnota: {val_de:.4f}")

        animate_de_search(
            func=data["func"],
            title=f"Animace DE pro funkci: {data['name']}",
            search_range=data["range"],
            pop_history=history,
            interval=150  # vyšší = pomalejší animace
        )