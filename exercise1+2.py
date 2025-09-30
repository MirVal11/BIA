import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#functions
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

def blind_search_run(obj_func, num_dimensions, search_range, max_iterations):
    # Nejlepší nalezené řešení (zatím nic) a jeho hodnota (nekonečno = nejhorší start)
    best_solution, best_value = None, float('inf')
    # Ukládá všechny náhodně generované kandidáty (hodí se pro vykreslení)
    all_solutions = []
    
    # Opakovaně generuje náhodná řešení v daném rozsahu
    for i in range(max_iterations):
        candidate = [np.random.uniform(search_range[0], search_range[1])
                     for i in range(num_dimensions)]
        all_solutions.append(candidate)
        value = obj_func(candidate)

        # Pokud je nové řešení lepší, nahradí dosavadní nejlepší
        if value < best_value:
            best_solution, best_value = candidate, value

    # Vrací nejlepší řešení, jeho hodnotu a všechny vyzkoušené body
    return best_solution, best_value, all_solutions


def hill_climbing_run(obj_func, num_dimensions, search_range, max_iterations, num_neighbors=1, step_std=0.1):
    # Náhodně zvolené počáteční řešení
    current_solution = np.random.uniform(search_range[0], search_range[1], num_dimensions)
    current_value = obj_func(current_solution)
    # Historie navštívených řešení pro vykreslení cesty
    path_history = [current_solution.copy()]
    
    for _ in range(max_iterations):
        best_neighbor = None
        best_neighbor_value = float('inf')

        # Generuje několik sousedů (10) kolem aktuálního bodu
        for _ in range(num_neighbors):
            neighbor = current_solution + np.random.normal(0, step_std, num_dimensions)
            # Omezí řešení na dovolený rozsah
            neighbor = np.clip(neighbor, search_range[0], search_range[1])
            neighbor_value = obj_func(neighbor)
            
            # Vybere nejlepšího souseda
            if neighbor_value < best_neighbor_value:
                best_neighbor = neighbor
                best_neighbor_value = neighbor_value

        # Pokud je nejlepší soused lepší než aktuální řešení, posune se tam
        if best_neighbor_value < current_value:
            current_solution = best_neighbor
            current_value = best_neighbor_value
            path_history.append(current_solution.copy())
        else:
            # Pokud se nenašel lepší soused, algoritmus končí (uvízl v lokálním minimu)
            break
    
    # Vrací nejlepší nalezené řešení, jeho hodnotu a celou cestu
    return current_solution, current_value, path_history


# Visualization
def plot_function(func, title, x_range, y_range, cmap='viridis', solutions=None, path=None):
    # Vytvoření mřížky hodnot
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([xi, yi]) for xi in x] for yi in y])

    # Nastavení grafu
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.7)

    # Pokud byly předány řešení v blind search, vykreslí se
    if solutions:
        has_labels = any('label' in sol and sol['label'] for sol in solutions)
        for sol in solutions:
            px, py = sol['point'][0], sol['point'][1]
            pz = func([px, py])
            ax.scatter(px, py, pz,
                       color=sol.get('color', 'red'),
                       s=sol.get('size', 120),
                       marker=sol.get('style', 'o'),
                       label=sol.get('label', None) if has_labels else None,
                       edgecolor='black',
                       zorder=5)

    # Pokud byla předána cesta v hill climbing, vykreslí ji červeně
    if path:
        path = np.array(path)
        path_z = np.array([func(p) for p in path])
        ax.plot(path[:, 0], path[:, 1], path_z, color='red', linewidth=2, label='Hill Climbing Path')
        ax.scatter(path[-1, 0], path[-1, 1], path_z[-1], color='red', s=150, marker='*', label='Hill Climbing Final Solution', zorder=10)

    # Popisky a legenda
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('f(x,y)')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    plt.show()


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
        {"func": zakharov, "name": "Zakharov", "range": [-10, -10]}, #zmena
        {"func": ackley, "name": "Ackley", "range": [-32.768, 32.768]}
    ]

    # Blind Search pro všechny funkce
    for data in function_data:
        # Spuštění blind search
        best_blind, val_blind, all_blind_points = blind_search_run(data["func"], dimensions, data["range"], max_iterations=500)
        print(f"Blind Search ({data['name']}): Best solution: {best_blind}, Value: {val_blind:.4f}")
        # Vykreslení celé plochy funkce + náhodně zkoušené body + nejlepší nalezený bod
        plot_function(
            data["func"],
            f"Blind Search on {data['name']} Function",
            data["range"],
            data["range"],
            solutions=[{'point': p, 'style': '.', 'color': 'black', 'size': 10} for p in all_blind_points] +
                      [{'point': best_blind, 'style': 'D', 'color': 'green', 'label': 'Best Solution', 'size': 120}]
        )
    
    # Hill Climbing pro všechny funkce
    for data in function_data:
        # Spuštění hill climbing
        best_hc, val_hc, path_hc = hill_climbing_run(data["func"], dimensions, data["range"], max_iterations=100, num_neighbors=10, step_std=0.5)
        print(f"Hill Climbing ({data['name']}): Best solution: {best_hc}, Value: {val_hc:.4f}")
        # Vykreslení celé plochy funkce + cesty hill climbing
        plot_function(
            data["func"],
            f"Hill Climbing on {data['name']} Function",
            data["range"],
            data["range"],
            path=path_hc
        )
