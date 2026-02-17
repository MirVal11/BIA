import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random
from copy import deepcopy

# =G=============================================================================
# 1. FUNKCE PRO OPTIMALIZACI
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


# Řídící parametry SOMA
SOMA_POP_SIZE = 20      # pop_size
SOMA_PRT = 0.4          # PRT
SOMA_PATH_LENGTH = 3.0  # PathLength
SOMA_STEP = 0.11        # Step
SOMA_MAX_MIGRATIONS = 100 # M_max


class Individual:
    """Reprezentuje jedince (řešení) v populaci SOMA."""
    def __init__(self, num_dimensions, search_range, obj_func):
        lower, upper = search_range
        self.params = np.random.uniform(lower, upper, num_dimensions)
        self.f = obj_func(self.params)

def ensure_bounds(vec, search_range):
    """Zajistí, že parametry nepřekročí definované hranice."""
    lower_bound, upper_bound = search_range[0], search_range[1]
    return np.clip(vec, lower_bound, upper_bound)


def soma_alltoone_run(obj_func, num_dimensions, search_range, max_migrations, pop_size, prt, path_length, step):

    # Inicializace populace
    population = [Individual(num_dimensions, search_range, obj_func) for _ in range(pop_size)]
    
    # Historie pozic celé populace pro animaci
    full_population_history = []
    # Uložíme počáteční stav
    full_population_history.append(np.array([ind.params.copy() for ind in population]))
    
    for m in range(max_migrations):
        
        # "Lídr" pro tuto migrační smyčku. (nejlepší řešení)
        leader_index = np.argmin([ind.f for ind in population])
        
        # Uložíme si kopii pozice (parametrů) lídra.
        leader_params = population[leader_index].params.copy()
        
        # Projdeme každého jedince v populaci.
        for i in range(pop_size):
            
            # Získáme aktuálního jedince 'i'.
            current_individual = population[i]
            
            # Uložíme si jeho startovní pozici pro tuto migraci.
            x_i_start = current_individual.params.copy() 
            
            # Kontrola, zda jedinec není Lídr        
            if i == leader_index:
                # Pokud je aktuální jedinec sám Lídr, nemá kam migrovat.
                pass 
            else:
                # Tento blok se provede pro všechny jedince, kteří NEJSOU lídři.
                
                #Vytvoření PRT Vektoru

                # Vytvoříme náhodný vektor čísel [0, 1] o velikosti 'num_dimensions'.
                # Porovnáním s 'prt' vytvoříme binární masku (True/False).
                # True (1) znamená, že daná dimenze bude použita pro pohyb.
                # False (0) znamená, že v této dimenzi se jedinec nepohne.
                # 0-1
                T_mask = np.random.uniform(0, 1, num_dimensions) < prt
                
                # Výpočet směrového vektoru D
                
                # Vypočítáme "směrový" vektor D.
                # D = (Cíl - Start) * Maska
                # D = (leader_params - x_i_start) * T_mask
                # Výsledkem je vektor, který ukazuje od jedince k lídrovi,
                # ale pouze v dimenzích, kde T_mask byla True.
                D = (leader_params - x_i_start) * T_mask
                
                # Prohledávání cesty
                
                # Inicializujeme 't' (čas/vzdálenost) na základní 'step'.
                t = step
                
                # Inicializujeme nejlepší pozici nalezenou na startovní pozici jedince.
                best_position_on_path = x_i_start.copy()
                
                # Stejně tak inicializujeme nejlepší fitness
                best_f_on_path = current_individual.f
                
                while t <= path_length:
                    
                    # Vypočítáme novou "zkušební" pozici (trial position).
                    # x_trial = Start + (Směr * vzdálenost)
                    x_trial = x_i_start + D * t
                    
                    # Zajistíme, že nová pozice nepřekročila hranice 'search_range'.
                    x_trial = ensure_bounds(x_trial, search_range)
                    
                    # Vyhodnotíme fitness (hodnotu) této zkušební pozice.
                    f_trial = obj_func(x_trial)
                    
                    # Je tato zkušební pozice lepší než druhá
                    if f_trial < best_f_on_path:
                        # Pokud ano, aktualizujeme nejlepší fitness na cestě
                        best_f_on_path = f_trial
                        # a uložíme si tuto lepší pozici.
                        best_position_on_path = x_trial.copy()
                    
                    # Posuneme 't' o další krok.
                    t += step
                
                # Dívám se zda je lepší
                if best_f_on_path < current_individual.f:
                    
                    # Pokud ano, jedinec se "přesune" na tuto novou, lepší pozici.
                    current_individual.params = best_position_on_path
                    
                    # A aktualizuje se jeho fitness hodnota.
                    current_individual.f = best_f_on_path
                
        
        # Uložíme celou populace po této migrační fázi pro animaci.
        full_population_history.append(np.array([ind.params.copy() for ind in population]))
    
    final_leader_index = np.argmin([ind.f for ind in population])
    final_leader_params = population[final_leader_index].params.tolist()
    final_leader_f = population[final_leader_index].f
    
    # Vrátíme nejlepší řešení, jeho hodnotu a kompletní historii pro animaci.
    return final_leader_params, final_leader_f, full_population_history


def animate_soma_contour(func, title, x_range, y_range, full_population_history, cmap='viridis'):
    """
    Vykreslí 2D konturní (heatmap) graf funkce a animuje pohyb celé populace SOMA.
    """
    
    # --- Příprava pro heatmapu ---
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([X[i,j], Y[i,j]]) for j in range(100)] for i in range(100)])
    # -----------------------------

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # --- Vykreslení heatmapy ---
    contour = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
    fig.colorbar(contour, label='f(x,y) value')
    # ---------------------------
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Inicializace teček pro celou populaci
    initial_positions = full_population_history[0]
    # Změna barvy na 'black' pro lepší kontrast
    particles_plot, = ax.plot(initial_positions[:, 0], initial_positions[:, 1], 
                              'o', color='black', markersize=5, alpha=0.7, label='SOMA Population')
    
    # Text pro zobrazení aktuální migrace
    migration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white',
                             fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    def update(frame):
        """Aktualizační funkce pro animaci."""
        current_positions = full_population_history[frame]
        
        # Aktualizujeme pozice teček
        particles_plot.set_data(current_positions[:, 0], current_positions[:, 1])
    

        migration_text.set_text(f'Migrace: {frame} / {len(full_population_history)-1}')
        ax.set_title(f"SOMA Pohyb populace (Contour Plot): {title}")
        
        # Vrátíme pouze upravené artists
        return particles_plot, migration_text

    # Zde interval=100 znamená 100ms na snímek (rychlejší animace)
    anim = animation.FuncAnimation(fig, update, frames=len(full_population_history), 
                                   interval=100, blit=True, repeat=False)
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    dimensions = 2
    
    # Seznam optimalizačních testovacích funkcí a jejich rozsahů
    function_data = [
        {"func": sphere, "name": "Sphere", "range": [-5.12, 5.12]},
        {"func": schwefel, "name": "Schwefel", "range": [-500, 500]},
        {"func": rosenbrock, "name": "Rosenbrock", "range": [-2.048, 2.048]},
        {"func": rastrigin, "name": "Rastrigin", "range": [-5.12, 5.12]},
        {"func": griewank, "name": "Griewank", "range": [-5, 5]},
        {"func": levy, "name": "Levy", "range": [-10, 10]},
        {"func": michalewicz, "name": "Michalewicz", "range": [0, np.pi]},
        {"func": zakharov, "name": "Zakharov", "range": [-10, 10]},
        {"func": ackley, "name": "Ackley", "range": [-32.768, 32.768]}
    ]
    
    # Spuštění SOMA pro všechny funkce
    for data in function_data:
        
        # 1. Optimalizace SOMA AllToOne
        best_soma, val_soma, full_population_history = soma_alltoone_run(
            data["func"], 
            dimensions, 
            data["range"], 
            max_migrations=SOMA_MAX_MIGRATIONS,
            pop_size=SOMA_POP_SIZE, 
            prt=SOMA_PRT, 
            path_length=SOMA_PATH_LENGTH,
            step=SOMA_STEP
        )

        print(f"SOMA AllToOne Optimization ({data['name']}): Best solution: x={best_soma[0]:.4f}, y={best_soma[1]:.4f}, Value: {val_soma:.4f}")

        animate_soma_contour(
            data["func"],
            f"{data['name']} (Best value: {val_soma:.4f})",
            data["range"],
            data["range"],
            full_population_history=full_population_history
        )