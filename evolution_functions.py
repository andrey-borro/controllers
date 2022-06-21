
import numpy as np
from numpy.lib.index_tricks import nd_grid

def create_child(parents: np.ndarray, mr1: float, mr2: float) -> np.ndarray:
    size = parents.shape[1]
    child = np.copy(parents[0])
    cross_mask = np.arange(size)[np.random.rand(size) < 0.5]
    point_mutation_mask = np.arange(size)[np.random.rand(size) < mr1]
    child[cross_mask] = parents[1][cross_mask]
    child[point_mutation_mask] = np.random.randn(size)[point_mutation_mask]
    child = child + (mr2 * np.random.randn(size))
    return child.reshape(1,size)
    
def evolve_population(population: np.ndarray, 
                        fitness_list: np.ndarray, 
                        *,
                        new_pop_size: int = None,
                        mutation_rate1: float = 0.02, 
                        mutation_rate2: float = 0.02) -> np.ndarray:
    """Evolves a population according to 2 mutation factors.
        Inputs: 
            population - a 2D numpy array of population members (rows) and their genes [floats]
            fitness_list - a 1D numpy array of their [non-negative!] corresponding fitnesses
            mutation_rate_1 - the factor of the additive gaussian mutation during reproduction
            mutation_rate_2 - the chance of a gene being randomised during reproduction
        Returns:
            A new 2D array with new genes for the population
    """
    old_pop_size = population.shape[0]
    new_pop_size = old_pop_size if new_pop_size is None else new_pop_size
    new_population = np.empty((0, population.shape[1]))
    sampling_distribution = fitness_list/sum(fitness_list)
    for _ in range(new_pop_size):
        #we sample parents proportionately to their fitness
        sample_indices = np.random.choice(old_pop_size, 2, p=sampling_distribution)
        child = create_child(population[sample_indices], mutation_rate1, mutation_rate2)
        new_population = np.append(new_population, child, axis=0)
    return new_population
