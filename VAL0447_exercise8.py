import numpy as np
import matplotlib.pyplot as plt
import random

# Parametry algoritmu
N_CITIES = 30  # Počet měst
N_ANTS = 30    # Počet mravenců
N_ITERATIONS = 150 # Počet iterací (migračních cyklů)

# Parametry ovlivňující chování mravenců
ALPHA = 1.0  # Důležitost feromonové stopy
BETA = 5.0   # Důležitost heuristické informace (vzdálenosti)
RHO = 0.5    # Rychlost odpařování feromonu
Q = 100.0    # Množství feromonu, které mravenec zanechá

def calculate_distance_matrix(cities):
    """Vypočítá matici vzdáleností mezi všemi městy."""
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = np.linalg.norm(cities[i] - cities[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix

def calculate_path_length(path, dist_matrix):
    """Vypočítá celkovou délku dané trasy."""
    length = 0
    for i in range(len(path) - 1):
        length += dist_matrix[path[i], path[i+1]]
    length += dist_matrix[path[-1], path[0]] # Návrat do startovního města
    return length

def ant_colony_optimization():
    # Generování náhodných pozic měst
    cities = np.random.rand(N_CITIES, 2) * 100
    # Vzdálenost mezi městy
    dist_matrix = calculate_distance_matrix(cities)

    # Odhad (1 / vzdálenost) dvojice bodů
    # Malá hodnota, aby se zamezilo dělení nulou inverse
    eta = 1.0 / (dist_matrix + 1e-10)

    # Inicializace feromonové matice
    pheromone_matrix = np.ones((N_CITIES, N_CITIES))

    # Proměnné pro sledování nejlepší nalezené trasy
    global_best_path = []
    global_best_path_length = float('inf')
    
    # Inicializace vizualizace
    fig, ax = initialize_visualization(cities)

    for iteration in range(N_ITERATIONS):
        print(f"Iterace {iteration+1}/{N_ITERATIONS}")

        # Seznam všech tras a jejich délek, které mravenci nalezli v této iteraci
        all_ant_paths = []
        
        # Každému mravenci se přiřadí jiné startovní město
        start_cities = list(range(N_CITIES))   # vytvoření seznamu měst
        random.shuffle(start_cities)           # náhodné promíchání pořadí
        start_cities = start_cities[:N_ANTS]   # vybere se prvních N_ANTS měst jako startovní pozice

        # Konstrukce tras – každý mravenec projde všechna města
        for k in range(N_ANTS):
            current_city = start_cities[k]          # aktuální město, odkud mravenec začíná
            path = [current_city]                   # seznam měst, která mravenec navštívil
            unvisited_cities = set(range(N_CITIES)) # všechna města
            unvisited_cities.remove(current_city)   # odstraní startovní město (už navštíveno)
            
            # Dokud existují nenavštívená města
            while unvisited_cities:
                probabilities = []  # pravděpodobnosti přechodu do dalších měst
                
                # Pro všechna nenavštívená města vypočítáme "atraktivitu"
                for next_city in unvisited_cities:
                    # Feromonová stopa – čím víc feromonu, tím atraktivnější cesta
                    tau_alpha = pheromone_matrix[current_city, next_city] ** ALPHA
                    # Čím kratší vzdálenost, tím větší přitažlivost
                    eta_beta = eta[current_city, next_city] ** BETA
                    # Výsledná atraktivita cesty (kombinace feromonu a vzdálenosti)
                    probabilities.append(tau_alpha * eta_beta)
                
                # Normalizace pravděpodobností (aby jejich součet byl 1) suma
                sum_probs = sum(probabilities)
                probabilities = [p / sum_probs for p in probabilities]

                # Výběr dalšího města podle vypočtených pravděpodobností
                next_city = random.choices(list(unvisited_cities), weights=probabilities, k=1)[0]
                
                # Mravenec se přesune do dalšího města
                path.append(next_city)
                unvisited_cities.remove(next_city)
                current_city = next_city  # aktualizace aktuálního města
                
            # Po dokončení cesty (navštívení všech měst) uložíme trasu a její délku
            all_ant_paths.append((path, calculate_path_length(path, dist_matrix)))

        # Aktualizace feromonů
        # a) Odpařování feromonu – všechny cesty ztrácí část své feromonové stopy
        pheromone_matrix *= (1 - RHO)

        # b) Přidání nového feromonu podle nalezených tras
        for path, path_length in all_ant_paths:
            pheromone_deposit = Q / path_length  # množství feromonu ∝ kvalitě (kratší = více)
            for i in range(N_CITIES - 1):
                # Přidání feromonu na cestu mezi městy i a i+1
                pheromone_matrix[path[i], path[i+1]] += pheromone_deposit
                pheromone_matrix[path[i+1], path[i]] += pheromone_deposit  # symetrická matice
            # Přidání feromonu i na cestu zpět do výchozího města (uzavření okruhu)
            pheromone_matrix[path[-1], path[0]] += pheromone_deposit
            pheromone_matrix[path[0], path[-1]] += pheromone_deposit

        # Určení nejlepší trasy v aktuální iteraci
        current_best_path, current_best_length = min(all_ant_paths, key=lambda x: x[1])

        # Pokud je tato trasa lepší než globálně nejlepší, aktualizujeme ji
        if current_best_length < global_best_path_length:
            global_best_path = current_best_path
            global_best_path_length = current_best_length
            
            # Aktualizace vizualizace pouze při nalezení lepší trasy
            update_visualization(ax, cities, global_best_path, iteration, global_best_path_length)


    # Zobrazení finálního výsledku
    print("\nOptimalizace dokončena.")
    print(f"Nejlepší nalezená trasa: {global_best_path}")
    print(f"Délka nejlepší trasy: {global_best_path_length:.2f}")

    # Finalní graf
    ax.set_title(f"Finální nejlepší trasa | Délka: {global_best_path_length:.2f}")
    if len(ax.lines) > 1: ax.lines[1].set_label(f'Finální trasa')
    plt.ioff()
    plt.legend()
    plt.show()

def initialize_visualization(cities):
    """Inicializuje graf pro vizualizaci."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Průběh hledání nejlepší trasy (ACO)")
    ax.set_xlabel("X souřadnice")
    ax.set_ylabel("Y souřadnice")
    # Vykreslení měst jako bodů
    ax.plot(cities[:, 0], cities[:, 1], 'bo', markersize=8, label='Města')
    plt.legend()
    plt.ion() # Zapnutí interaktivního režimu
    plt.show()
    return fig, ax

def update_visualization(ax, cities, best_path, iteration, best_path_length):
    """Aktualizuje vizualizaci o nově nalezenou nejlepší trasu."""
    # Smazání předchozí trasy
    if len(ax.lines) > 1:
        ax.lines[1].remove()

    # Vykreslení nové nejlepší trasy
    path_coords = np.array([cities[i] for i in best_path + [best_path[0]]])
    ax.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=1.5, label=f'Nejlepší trasa po iteraci {iteration}')

    # Aktualizace titulku s délkou trasy
    ax.set_title(f"Iterace: {iteration+1}/{N_ITERATIONS} | Nejlepší délka: {best_path_length:.2f}")
    plt.pause(0.01)


if __name__ == '__main__':
    ant_colony_optimization()