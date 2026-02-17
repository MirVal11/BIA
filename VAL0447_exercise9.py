import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random
from copy import deepcopy

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

class Firefly:
    """Reprezentuje jedince (světlušku) v populaci."""
    def __init__(self, num_dimensions, search_range, obj_func):
        lower, upper = search_range
        self.params = np.random.uniform(lower, upper, num_dimensions)
        self.f = obj_func(self.params)

def ensure_bounds(vec, search_range):
    """Zajistí, že parametry nepřekročí definované hranice."""
    lower_bound, upper_bound = search_range[0], search_range[1]
    return np.clip(vec, lower_bound, upper_bound)


def firefly_algorithm_run(obj_func, num_dimensions, search_range, max_iterations, pop_size=30, alpha=0.3, beta0=1.0):    
    # Vytvoříme populaci světlušek
    fireflies = [Firefly(num_dimensions, search_range, obj_func) for _ in range(pop_size)]
    
    # Najdeme a uložíme si nejlepší světlušku (gBest) z počáteční populace.
    gBest_firefly = min(fireflies, key=lambda ff: ff.f)
    gBest_params = gBest_firefly.params.copy() # Kopie pozice nejlepšího
    gBest_f = gBest_firefly.f                 # Fitness nejlepšího
    
    # Seznam pro ukládání populace pro animaci.
    full_population_history = []
    full_population_history.append(np.array([ff.params.copy() for ff in fireflies])) 
    
    # Běží 'max_iterations' krát.
    for m in range(max_iterations):
        
        # Postupné tlumení (snižování) parametru 'alpha' (náhodný pohyb).
        # Pomáhá algoritmu konvergovat ke konci (méně náhodnosti).
        alpha_damp = 0.99
        alpha *= alpha_damp
        
        # Vytvoříme 'hlubokou' kopii populace.
        fireflies_copy = deepcopy(fireflies)

        # Pohyb světlušek (porovnání každé s každou)
        for i in range(pop_size):
            for j in range(pop_size):
                
                # Světluška se neporovnává sama se sebou.
                if i == j:
                    continue 

                # Princip FA: Pohybujeme se POUZE k jasnějším světluškám.
                if fireflies_copy[j].f < fireflies_copy[i].f:
                    # Světluška 'j' je jasnější (lepší) než 'i'.
                    # Musíme pohnout světluškou 'i' (v originálním seznamu 'fireflies') směrem k 'j'.
                    
                    # Vypočítáme Euklidovskou vzdálenost 'r'
                    r_euklid = np.linalg.norm(fireflies[i].params - fireflies[j].params)
                    # Ošetření dělení nulou, pokud by byly náhodou na stejném místě.
                    r = max(r_euklid, 1e-10) 
                    
                    # Vypočítáme atraktivitu 'beta'
                    beta = beta0 / (1 + r) 
                    
                    # Složka pohybu daná přitažlivostí k 'j'.
                    attraction_term = beta * (fireflies_copy[j].params - fireflies_copy[i].params)
                    # Složka náhodného pohybu (Gaussovo/Normální rozdělení).
                    random_term = alpha * np.random.randn(num_dimensions)
                    
                    # Aktualizujeme pozici světlušky 'i'.
                    fireflies[i].params += attraction_term + random_term
                    # Zajistíme, že světluška 'i' zůstala v povolených hranicích.
                    fireflies[i].params = ensure_bounds(fireflies[i].params, search_range)
        
        # Přepočítej fitness a proveď náhodný pohyb nejlepší
        
        # Po všech pohybech musíme přepočítat
        # fitness (jas) pro všechny světlušky, které se pohnuly.
        for ff in fireflies:
            ff.f = obj_func(ff.params) 
            
        # Najdeme index aktuálně nejlepší světlušky po této iteraci.
        current_best_idx = min(range(pop_size), key=lambda k: fireflies[k].f)
        
        # Pravidlo FA: Nejlepší světluška provádí náhodný pohyb (random walk).
        random_move_best = alpha * np.random.randn(num_dimensions)
        potential_new_pos = fireflies[current_best_idx].params + random_move_best
        potential_new_pos = ensure_bounds(potential_new_pos, search_range)
        # Zjistíme fitness této nové náhodné pozice.
        potential_new_f = obj_func(potential_new_pos)

        # Nejlepší světluška se přesune, POUZE pokud je nová pozice lepší. 
        if potential_new_f < fireflies[current_best_idx].f:
            fireflies[current_best_idx].params = potential_new_pos
            fireflies[current_best_idx].f = potential_new_f
            

        # Aktualizujeme globálně nejlepší řešení (gBest), pokud je to nutné.
        if fireflies[current_best_idx].f < gBest_f:
            gBest_f = fireflies[current_best_idx].f
            gBest_params = fireflies[current_best_idx].params.copy()
            
        # Uložíme "snímek" celé populace po této iteraci pro animaci.
        full_population_history.append(np.array([ff.params.copy() for ff in fireflies]))

    # Po skončení všech iterací vrátíme finální řešení a historii.
    return gBest_params.tolist(), gBest_f, full_population_history


def animate_fa_contour(func, title, x_range, y_range, full_population_history, cmap='viridis'):
    """
    Vykreslí 2D konturní (heatmap) graf funkce a animuje pohyb celé populace FA.
    Nezobrazuje gBest path ani finální řešení.
    """
    
    # Příprava pro heatmapu
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([X[i,j], Y[i,j]]) for j in range(100)] for i in range(100)])

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Vykreslení heatmapy
    contour = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
    fig.colorbar(contour, label='f(x,y) value')
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Inicializace teček ("čárek") pro celou populaci
    initial_positions = full_population_history[0]
    particles_plot, = ax.plot(initial_positions[:, 0], initial_positions[:, 1], 
                              'o', color='black', markersize=5, alpha=0.7, label='Firefly Population')
    
    # Text pro zobrazení aktuální iterace
    iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white',
                             fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    def update(frame):
        """Aktualizační funkce pro animaci."""
        current_positions = full_population_history[frame]
        
        # Aktualizujeme pozice teček
        particles_plot.set_data(current_positions[:, 0], current_positions[:, 1])
        
        iteration_text.set_text(f'Iterace: {frame} / {len(full_population_history)-1}')
        ax.set_title(f"Firefly Algorithm Pohyb populace: {title}")
        
        return particles_plot, iteration_text

    anim = animation.FuncAnimation(fig, update, frames=len(full_population_history), 
                                   interval=100, blit=True, repeat=False)
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    dimensions = 2
    
    FA_POP_SIZE = 30 # velikost
    FA_ALPHA = 0.1 # nahodny koeficient
    FA_BETA0 = 2.0 # intenzita svetla
    FA_MAX_ITERATIONS = 200
    
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
    
    # Spuštění FA pro všechny funkce
    for data in function_data:
        
        # 1. Optimalizace Algoritmem světlušek (FA)
        best_fa, val_fa, full_fa_history = firefly_algorithm_run(
            data["func"], 
            dimensions, 
            data["range"], 
            max_iterations=FA_MAX_ITERATIONS,
            pop_size=FA_POP_SIZE, 
            alpha=FA_ALPHA, 
            beta0=FA_BETA0
        )

        print(f"Firefly Algorithm ({data['name']}): Best solution: x={best_fa[0]:.4f}, y={best_fa[1]:.4f}, Value: {val_fa:.4f}")
        
        animate_fa_contour(
            data["func"],
            f"{data['name']} (Best value: {val_fa:.4f})",
            data["range"],
            data["range"],
            full_population_history=full_fa_history
        )