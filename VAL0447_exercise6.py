import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random
from copy import deepcopy

# ==============================================================================
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


# Řídící parametry PSO dle zadání
PSO_POP_SIZE = 15       # pop_size
PSO_C1 = 2.0            # c1
PSO_C2 = 2.0            # c2
PSO_MAX_ITERATIONS = 50 # M_max

WS = 0.9 # w_s
WE = 0.4 # w_e

class Particle:
    """Reprezentuje jedince (částici) v roji."""
    def __init__(self, num_dimensions, search_range, obj_func):
        lower, upper = search_range
        self.params = np.random.uniform(lower, upper, num_dimensions)
        self.v = np.zeros(num_dimensions)
        self.f = obj_func(self.params)
        self.pBest_params = self.params.copy()
        self.pBest_f = self.f

def ensure_bounds(vec, search_range):
    """Zajistí, že parametry nepřekročí definované hranice."""
    lower_bound, upper_bound = search_range[0], search_range[1]
    return np.clip(vec, lower_bound, upper_bound)

def ensure_velocity_bounds(v, v_limit):
    """Zajistí, že rychlost nepřekročí v_limit."""
    v_max = v_limit
    v_min = -v_limit
    return np.clip(v, v_min, v_max)

def calculate_w(current_iteration, max_iterations, ws=0.9, we=0.4):
    """Vypočítá inerciální koeficient w, který lineárně klesá."""
    if max_iterations == 0:
        return ws
    return ws - (ws - we) * current_iteration / max_iterations

# ALGORITMUS PSO

def pso_run(obj_func, num_dimensions, search_range, max_iterations, pop_size=15, c1=2.0, c2=2.0, ws=0.9, we=0.4):
    """
    Implementace PSO.
    Vrací historii pozic VŠECH částic pro animaci.
    """
    
    # Inicializace roje částic (každá částice je náhodně rozmístěná v prostoru řešení )(15)
    swarm = [Particle(num_dimensions, search_range, obj_func) for _ in range(pop_size)]
    
    # Inicializace globálního nejlepšího řešení (gBest)
    gBest_params = None
    gBest_f = float('inf')
    
    # Najdeme první nejlepší částici (ta s nejnižší hodnotou funkce)
    for p in swarm:
        if p.f < gBest_f:
            gBest_f = p.f
            gBest_params = p.params.copy()
            
    # Uložíme si historii
    swarm_history = []
    swarm_history.append(np.array([p.params for p in swarm]))
    
    # Nastavíme maximální povolenou rychlost částic 0.2 * 5 - (-5)
    v_limit = 0.2 * (search_range[1] - search_range[0]) 

    # Max iterations
    for m in range(max_iterations):
        
        # postupně klesá z ws → we, aby se zpomaloval ke konci
        w = calculate_w(m, max_iterations, ws, we)
        
        # Pro každou částici v roji:
        for p in swarm:
            # Generujeme náhodné faktory pro kognitivní a sociální složku
            r1 = np.random.uniform(0, 1, num_dimensions)
            r2 = np.random.uniform(0, 1, num_dimensions)
            
            # Kognitivní složka = snaha přiblížit se vlastnímu nejlepšímu řešení (pBest)
            cognitive_component = r1 * c1 * (p.pBest_params - p.params)
            
            # Sociální složka = snaha přiblížit se nejlepšímu řešení v celém roji (gBest)
            social_component = r2 * c2 * (gBest_params - p.params)
            
            # Aktualizace rychlosti částice (kombinace předchozí rychlosti, osobního a globálního vlivu)
            p.v = w * p.v + cognitive_component + social_component
            
            # Omezíme rychlost, aby částice nelétala mimo definovaný rozsah
            p.v = ensure_velocity_bounds(p.v, v_limit)

            # Posuneme částici podle nové rychlosti
            p.params = p.params + p.v
            
            # Ujistíme se, že částice zůstává uvnitř povoleného prostoru
            p.params = ensure_bounds(p.params, search_range)
            
            # Vyhodnotíme novou hodnotu cílové funkce
            new_f = obj_func(p.params)
            p.f = new_f
            
            # Pokud si částice zlepšila svůj osobní výsledek → aktualizujeme pBest
            if new_f < p.pBest_f:
                p.pBest_f = new_f
                p.pBest_params = p.params.copy()
                
            # Pokud se tímto zlepšilo i globální nejlepší řešení → aktualizujeme gBest
            if p.pBest_f < gBest_f:
                gBest_f = p.pBest_f
                gBest_params = p.pBest_params.copy()
        
        # Uložíme pozice všech částic po dané iteraci (pro animaci)
        swarm_history.append(np.array([p.params for p in swarm]))
    
    # Vrátíme nejlepší nalezené řešení, jeho hodnotu a celou historii pohybu roje
    return gBest_params.tolist(), gBest_f, swarm_history

# VIZUALIZACE

def animate_pso_contour(func, title, x_range, y_range, swarm_history, cmap='viridis'):
    """
    Vykreslí 2D konturní (heatmap) graf funkce a animuje pohyb všech částic.
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([X[i,j], Y[i,j]]) for j in range(100)] for i in range(100)])

    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
    fig.colorbar(contour, label='f(x,y) value')
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    initial_positions = swarm_history[0]
    particles_plot, = ax.plot(initial_positions[:, 0], initial_positions[:, 1], 
                              'o', color='black', markersize=5)
    
    generation_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white',
                              fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    def update(frame):
        """Aktualizační funkce pro animaci."""
        positions = swarm_history[frame]
        particles_plot.set_data(positions[:, 0], positions[:, 1])
        generation_text.set_text(f'Generace: {frame} / {len(swarm_history)-1}')
        ax.set_title(f"PSO Pohyb roje: {title}")
        return particles_plot, generation_text

    # Zde interval=100 znamená 100ms na snímek (rychlejší animace)
    anim = animation.FuncAnimation(fig, update, frames=len(swarm_history), 
                                   interval=100, blit=True, repeat=False)
    
    # Zobrazení animace
    plt.show()

# 5. HLAVNÍ SPOUŠTĚCÍ BLOK
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
    
    for data in function_data:
        
        best_pso, val_pso, full_history = pso_run(
            data["func"], 
            dimensions, 
            data["range"], 
            max_iterations=PSO_MAX_ITERATIONS,
            pop_size=PSO_POP_SIZE, 
            c1=PSO_C1, 
            c2=PSO_C2,
            ws=WS,
            we=WE
        )

        print(f"  Nejlepší řešení: x={best_pso[0]:.4f}, y={best_pso[1]:.4f}")
        print(f"  Hodnota funkce: {val_pso:.4f}")

        # 2. Vizualizace PSO (Animace pohybu roje)
        animate_pso_contour(
            data["func"],
            f"{data['name']} (Nejlepší hodnota: {val_pso:.4f})",
            data["range"],
            data["range"],
            swarm_history=full_history
        )