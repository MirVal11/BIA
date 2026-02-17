import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation 
from matplotlib.path import Path 
from matplotlib.markers import MarkerStyle
import time 

def get_marker_path(marker_symbol):
    """Vrátí Path objekt pro daný symbol markeru pomocí MarkerStyle."""
    ms = MarkerStyle(marker_symbol)
    return ms.get_path().transformed(ms.get_transform())

def sphere(params):
    """f(x) = Σ(xᵢ²)"""
    return np.sum(np.array(params)**2)

def schwefel(params):
    """f(x) = 418.9829d - Σ[xᵢ sin(√(∣xᵢ∣))]"""
    params = np.array(params)
    d = len(params)
    sum_term = np.sum(params * np.sin(np.sqrt(np.abs(params))))
    return 418.9829 * d - sum_term

def rosenbrock(params, a=1):
    """f(x) = Σ[100(xᵢ₊₁−xᵢ²)² + (xᵢ−a)²]"""
    params = np.array(params)
    total_sum = 0
    for i in range(len(params) - 1):
        term1 = 100 * (params[i+1] - params[i]**2)**2
        term2 = (params[i] - a)**2
        total_sum += term1 + term2
    return total_sum

def rastrigin(params, A=10):
    """f(x) = 10d + Σ[xᵢ² - 10cos(2πxᵢ)]"""
    params = np.array(params)
    d = len(params)
    sum_term = np.sum(params**2 - A * np.cos(2 * np.pi * params))
    return A * d + sum_term

def griewank(params):
    """f(x) = Σ(xᵢ²/4000) - Π[cos(xᵢ/√i)] + 1"""
    params = np.array(params)
    sum_sq = np.sum(params**2)
    prod_cos = np.prod(np.cos(params / np.sqrt(np.arange(1, len(params) + 1))))
    return sum_sq / 4000 - prod_cos + 1

def levy(params):
    """f(x) = sin²(πw₁) + ... + (w_d−1)²(1+sin²(2πw_d)), wᵢ = 1 + (xᵢ−1)/4"""
    params = np.array(params)
    w = 1 + (params - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2_sum = np.sum((w[:-1]-1)**2 * (1+10*np.sin(np.pi*w[:-1]+1)**2))
    term3 = (w[-1]-1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    return term1 + term2_sum + term3

def michalewicz(params, m=10):
    """f(x) = -Σ[sin(xᵢ) sin²ᵐ(i xᵢ²/π)]"""
    params = np.array(params)
    d_indices = np.arange(1, len(params) + 1)
    return -np.sum(np.sin(params) * (np.sin(d_indices * params**2 / np.pi))**(2 * m))

def zakharov(params):
    """f(x) = Σ(xᵢ²) + (Σ(0.5 i xᵢ))² + (Σ(0.5 i xᵢ))⁴"""
    params = np.array(params)
    d_indices = np.arange(1, len(params) + 1)
    sum_x_sq = np.sum(params**2)
    sum_weighted_x = np.sum(0.5 * d_indices * params)
    return sum_x_sq + sum_weighted_x**2 + sum_weighted_x**4

def ackley(params, a=20, b=0.2, c=2 * np.pi):
    """f(x) = -a exp(...) + a + exp(1)"""
    params = np.array(params)
    d = len(params)
    sum1 = np.sum(params**2)
    sum2 = np.sum(np.cos(c * params))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.e

def simulated_annealing_run(obj_func, num_dimensions, search_range, 
                            T_0=100, T_min=0.5, alpha=0.95, step_std=0.5, max_steps = 100):
    
    # Inicializace teploty
    T = T_0
    
    # Náhodné počáteční řešení v daném rozsahu
    current_solution = np.random.uniform(search_range[0], search_range[1], num_dimensions)
    current_value = obj_func(current_solution)
    
    # Nejlepší nalezené řešení
    best_overall_solution = current_solution.copy()
    best_overall_value = current_value
    
    # Historie všech kroků
    path_history = [current_solution.copy()]
    step_count = 0
    
    # Hlavní cyklus SA – pokračuje dokud teplota neklesne pod minimum
    # nebo dokud není dosažen maximální počet kroků
    while T > T_min and step_count < max_steps:
        
        # Generování sousedního řešení
        neighbor = current_solution + np.random.normal(0, step_std, num_dimensions)
        
        # Oříznutí řešení do povoleného intervalu
        neighbor = np.clip(neighbor, search_range[0], search_range[1])
        neighbor_value = obj_func(neighbor)
        
        # Rozdíl kvality
        delta_E = neighbor_value - current_value 
        
        # Pokud je nové řešení lepší, přijmeme ho
        if delta_E < 0:
            current_solution = neighbor
            current_value = neighbor_value
        else:
            # Pokud je horší, přijmeme ho jen s určitou pravděpodobností
            r = np.random.uniform(0, 1)
            if r < np.exp(-delta_E / T):
                current_solution = neighbor
                current_value = neighbor_value
                
        # Aktualizace nejlepšího řešení
        if current_value < best_overall_value:
            best_overall_value = current_value
            best_overall_solution = current_solution.copy()
            
        # Uložení průběhu
        path_history.append(current_solution.copy())

        # Ochlazovací schéma – postupné snižování teploty
        T = T * alpha
        step_count += 1

    # Návrat nejlepšího řešení, jeho hodnoty a celé cesty
    return best_overall_solution, best_overall_value, path_history


def plot_heatmap_animation(func, title, x_range, y_range, cmap='jet', path=None):
    if not path or len(path) < 2:
        return

    path = np.array(path)
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)
    
    points = np.vstack([X.ravel(), Y.ravel()]).T
    Z_values = np.array([func(p) for p in points])
    Z = Z_values.reshape(X.shape)
    Z_clipped = np.clip(Z, np.min(Z), np.percentile(Z, 95)) 

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(Z_clipped, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap=cmap, aspect='auto')
    
    start_point = path[0]
    ax.scatter(start_point[0], start_point[1], color='lime', s=70, marker='o', label='Start', edgecolor='black', zorder=15)
    
    line, = ax.plot([], [], color='white', linewidth=1.5, linestyle='--', alpha=0.8, zorder=10)
    current_dot = ax.scatter([], [], color='cyan', s=30, marker='o', edgecolor='black', zorder=15)
    
    ax.set_title(title)
    ax.set_xlabel('X-osa')
    ax.set_ylabel('Y-osa')

    def animate(i):
        line.set_data(path[:i+1, 0], path[:i+1, 1])
        
        # Aktualizace tečky
        current_dot.set_offsets(path[i])
        
        # Změna barvy, velikosti a markeru pro finální bod
        if i == len(path) - 1:
            current_dot.set_color('cyan')
            current_dot.set_sizes([100])
            current_dot.set_paths([get_marker_path('o')]) 
        else:
            if i == len(path) - 2:
                current_dot.set_paths([get_marker_path('o')])
        return line, current_dot

    # Inicializace animace
    anim = FuncAnimation(fig, animate, frames=len(path), interval=2, blit=True)
    
    plt.show()
    
    try:
        anim.event_source.stop()
        time.sleep(0.1)
    except:
        pass
    plt.close(fig) 

def plot_function_3d_animation(func, title, x_range, y_range, cmap='viridis', path=None):
    if not path or len(path) < 2:
        return

    path = np.array(path)
    path_z = np.array([func(p) for p in path])
    
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    
    points = np.vstack([X.ravel(), Y.ravel()]).T
    Z_values = np.array([func(p) for p in points])
    Z = Z_values.reshape(X.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.7)

    # Vykreslení statického startovního bodu
    start_point = path[0]
    start_z = path_z[0]
    ax.scatter(start_point[0], start_point[1], start_z, color='lime', s=100, marker='o', label='Start', edgecolor='black', zorder=15)
    
    # Prvky, které se budou v animaci aktualizovat
    line, = ax.plot([], [], [], color='red', linewidth=2, zorder=10)
    # current_dot je PathCollection
    current_dot = ax.scatter([], [], [], color='cyan', s=40, marker='o', edgecolor='black', zorder=15)
    
    ax.set_title(title)
    ax.set_xlabel('X-osa')
    ax.set_ylabel('Y-osa')
    ax.set_zlabel('f(x,y)')
    
    # Omezení os pro stabilní zobrazení
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(np.min(Z_values), np.max(Z_values))

    def animate(i):
        # Aktualizace čáry cesty
        line.set_data(path[:i+1, 0], path[:i+1, 1])
        line.set_3d_properties(path_z[:i+1])
        
        # Aktualizace tečky
        current_dot._offsets3d = (path[i:i+1, 0], path[i:i+1, 1], path_z[i:i+1])
        
        # Změna barvy a velikosti tečky pro finální bod
        if i == len(path) - 1:
            current_dot.set_facecolors('cyan')
            current_dot.set_sizes([40])
            current_dot.set_paths([get_marker_path('o')]) 
        else:
            current_dot.set_facecolors('cyan')
            current_dot.set_sizes([40])
            if i == len(path) - 2:
                current_dot.set_paths([get_marker_path('o')])
            
        return line, current_dot

    anim = FuncAnimation(fig, animate, frames=len(path), interval=2, blit=False)
    
    plt.show()
    
    try:
        anim.event_source.stop()
        time.sleep(0.1)
    except:
        pass
    plt.close(fig)


if __name__ == "__main__":
    dimensions = 2
    
    # Seznam optimalizačních testovacích funkcí a jejich rozsahů
    function_data = [
        {"func": sphere, "name": "Sphere", "range": [-5.12, 5.12]},
        {"func": schwefel, "name": "Schwefel", "range": [-500, 500]},
        {"func": rosenbrock, "name": "Rosenbrock", "range": [-2.048, 2.048]},
        {"func": rastrigin, "name": "Rastrigin", "range": [-5.12, 5.12]},
        {"func": griewank, "name": "Griewank", "range": [-5, 5]}, #zmena
        {"func": levy, "name": "Levy", "range": [-10, 10]},
        {"func": michalewicz, "name": "Michalewicz", "range": [0, np.pi]},
        {"func": zakharov, "name": "Zakharov", "range": [-10, 10]}, #zmena
        {"func": ackley, "name": "Ackley", "range": [-32.768, 32.768]}
    ]
    
    print("\nSpouštění Simulated Annealing")
    
    # Parametry pro SA
    T0 = 100
    Tmin = 0.5
    alpha = 0.95
    
    for data in function_data:
        # 1. Spuštění Simulated Annealing
        best_sa, val_sa, path_sa = simulated_annealing_run(
            obj_func=data["func"], 
            num_dimensions=dimensions, 
            search_range=data["range"],
            T_0=T0,
            T_min=Tmin,
            alpha=alpha,
            step_std=(data["range"][1] - data["range"][0]) * 0.05 
        )

        print(f"\nSA ({data['name']}) dokončeno po {len(path_sa)-1} krocích. Nejlepší hodnota: {val_sa:.4f}")
        print("Spouštím 2D animaci Heatmapy")

        # 2. Vizualizace 1: Heatmap (2D)
        plot_heatmap_animation(
            data["func"], 
            f"SA Heatmap na {data['name']}", 
            data["range"], 
            data["range"],
            path=path_sa
        )
        
        print("Spouštím 3D animaci povrchového grafu")

        # 3. Vizualizace 2: 3D Povrchový graf
        plot_function_3d_animation(
            data["func"], 
            f"SA 3D Pohled na {data['name']}", 
            data["range"], 
            data["range"],
            path=path_sa
        )