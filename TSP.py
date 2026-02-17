import numpy as np
import matplotlib.pyplot as plt
import random

# PARAMETRY
D = 30        # Počet měst
NP = 50       # Velikost populace
G = 200       # Počet generací
MUT_PROB = 0.4  # Pravděpodobnost mutace

#np.random.seed(33)
#random.seed(33)

# INICIALIZACE MĚST
cities = np.random.rand(D, 2) * 100  # Náhodné souřadnice měst
population = [random.sample(range(D), D) for _ in range(NP)]
best_distances = []


plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

# HLAVNÍ CYKLUS GA
for gen in range(G):

    # Vyhodnocení populace - spočítení délku trasy pro každého jedince
    evaluated = []
    #všechny trasy
    for ind in population:
        total_distance = 0

        for i in range(D):
            city_a = cities[ind[i]]
            city_b = cities[ind[(i + 1) % D]] # zpět
            distance = np.linalg.norm(city_a - city_b)
            total_distance += distance

        evaluated.append((ind, total_distance))

    # Nejlepší jedinec
    best_ind, best_dist = min(evaluated, key=lambda x: x[1])
    best_distances.append(best_dist)

    # Nová populace (kopie)
    new_population = population.copy()

    # Evoluce populace
    for j in range(len(population)):
        parent_a = population[j]

        # Vyber náhodně druhého rodiče, který není stejný jako první
        possible_parents = []
        for p in population:
            if p != parent_a:
                possible_parents.append(p)
        parent_b = random.choice(possible_parents)

        cut = random.randint(1, D - 2)
        offspring = []

        # První část potomka z rodiče A
        for k in range(cut):
            offspring.append(parent_a[k])

        # Druhá část - přidáme města z B, která ještě nejsou v potomkovi
        for city in parent_b:
            if city not in offspring:
                offspring.append(city)

        # Mutace
        if random.random() < MUT_PROB:
            i1, i2 = random.sample(range(D), 2)
            temp = offspring[i1]
            offspring[i1] = offspring[i2]
            offspring[i2] = temp

        # Nahrazení horšího rodiče
        offspring_length = 0
        for i in range(D):
            city_a = cities[offspring[i]]
            city_b = cities[offspring[(i + 1) % D]]
            offspring_length += np.linalg.norm(city_a - city_b)

        parent_length = 0
        for i in range(D):
            city_a = cities[parent_a[i]]
            city_b = cities[parent_a[(i + 1) % D]]
            parent_length += np.linalg.norm(city_a - city_b)

        if offspring_length < parent_length:
            new_population[j] = offspring

    population = new_population

    # Vykreselní
    ax.clear()
    route = np.array([cities[i] for i in best_ind + [best_ind[0]]])
    ax.plot(route[:, 0], route[:, 1], 'r-')
    ax.plot(cities[:, 0], cities[:, 1], 'ro')
    ax.set_title(f"Generace {gen+1}, délka: {best_dist:.2f}")
    plt.pause(0.01)

plt.ioff()

# VÝSLEDKY
print(f"Nejkratší nalezená trasa: {min(best_distances):.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Nejlepší trasa
route = np.array([cities[i] for i in best_ind + [best_ind[0]]])
ax1.plot(route[:, 0], route[:, 1], 'r-')
ax1.plot(cities[:, 0], cities[:, 1], 'ro')
ax1.set_title(f'Nejlepší trasa: {best_dist:.2f}')
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

# Vývoj fitness
ax2.plot(best_distances)
ax2.set_title("Vývoj délky nejlepší trasy")
ax2.set_xlabel("Generace")
ax2.set_ylabel("Celková vzdálenost")

plt.show()