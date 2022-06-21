import numpy as np
import os
import copy
from typing import Callable
from abc import ABC, abstractmethod

from numpy.core.fromnumeric import sort

import evolution_functions

class BaseController: 
    """Intended for use as a superclass."""
    def __init__(self, controls: list[str]) -> None:
        """Inputs: Controls - a list of string names for the controls being passed to the controller"""
        self.controls = controls

    def return_controls(self) -> list[int]:
        """Returns an ordered integer list of all inputs (controls) for the game to execute. Controllers must override this method."""
        print('The return_controls method has not been overriden for the controller calling this.')
        return []

class GeneticTrainable: #Abstract Class
    #should we make update_genes etc into abstract methods?
    @staticmethod
    def print_scores(info, scores):
        print(f'[{info}] Max. Score: {max(scores):.2f} | Min. Score: {min(scores) :.2f} | Avg. Score: {sum(scores) / len(scores) :.2f}')

    @classmethod
    def train_genetic_base(
            cls,
            game_function: Callable[[BaseController], list[float]], 
            controllers: list['GeneticTrainable'],
            num_training_epochs: int = 30, 
            *,
            mutation_rate_1: float = 0.02,
            mutation_rate_2: float = 0.02,
            score_transform: Callable[[float], float]  = None,
            intermediate_save_filename: str = None) -> None:

        if not controllers:
            raise ValueError('Controller list is empty.')
        
        current_best_score = 0
        scores = game_function(copy.deepcopy(controllers))
        cls.print_scores(f'Initial Run', scores)
        for i in range(num_training_epochs):
            scores = cls.training_epoch(game_function, controllers, scores, mutation_rate_1, mutation_rate_2, score_transform)
            cls.print_scores(f'Epoch {i}', scores)
            if intermediate_save_filename and max(scores) > current_best_score:
                current_best = sorted(zip(controllers, scores), key=lambda x: x[1], reverse=True)[0]
                print(f'[Epoch {i}] Saving new best score of {current_best[1]:.2f}')
                current_best[0].save(intermediate_save_filename)
                current_best_score = current_best[1]
        
        #Final runthrough
        scores = cls.training_epoch(game_function, controllers, scores, mutation_rate_1, mutation_rate_2, score_transform)
        cls.print_scores(f'Epoch {num_training_epochs}', scores)

        return sorted(zip(controllers, scores), key=lambda x: x[1], reverse=True)
            
    
    @classmethod
    def training_epoch(
            cls,
            game_function: Callable[[BaseController], list[float]], 
            controllers: list['GeneticTrainable'],
            scores: list[float],
            mr1: float,
            mr2: float,
            score_transform: Callable[[float], float]) -> list[float]:
        
        fitness_list = np.array([score_transform(x) for x in scores] if score_transform else scores)

        layer_sizes = controllers[0].network.layer_sizes
        gene_width = sum(a*b for a,b in zip(layer_sizes[1:], layer_sizes[:-1])) + sum(layer_sizes[1:])
        population = np.zeros((len(controllers), gene_width)) #less scary than empty

        #check that all controllers are of the correct type and size
        for index, controller in enumerate(controllers):
            if not isinstance(controller, cls):
                raise TypeError(
                    f'All controllers being trained must be of type {cls.__name__} or a subclass. At least one controller is of type: {type(controller).__name__}'
                )
            if not controller.network.layer_sizes == layer_sizes:
                raise ValueError(f'All {cls.__name__} controllers being trained must have identically shaped networks.')

            population[index] = controller.convert_to_float_array()
        
        new_pop = evolution_functions.evolve_population(population, fitness_list, mutation_rate1=mr1, mutation_rate2=mr2, new_pop_size=len(population) - 1)
        np.vstack((new_pop, sorted(zip(population, scores), key=lambda x: x[1], reverse=True)[0][0]))

        #now we unpack the float lists back into controllers 
        for controller, genes in zip(controllers, new_pop):
            controller.update_genes(genes)
        
        scores = game_function(copy.deepcopy(controllers)) #make a copy to ensure they aren't being tampered with inside the game itself
        return scores
        
class NeuralNetwork:
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0/1.0 + np.exp(-z)

    @staticmethod
    def load(filename: str) -> 'NeuralNetwork':
        npzfile = np.load(f'{filename}.npz', allow_pickle=True)
        network = NeuralNetwork(empty_warning=False)
        network.biases = [np.array(bias_vector) for bias_vector in npzfile['biases']]
        network.weights = [np.array(weight_matrix) for weight_matrix in npzfile['weights']]
        network.layer_sizes = [w.shape[1] for w in network.weights] + [network.weights[-1].shape[0]]
        network.num_layers = len(network.layer_sizes)
        return network

    def __init__(self, layer_sizes: list[int] = None, empty_warning: bool = True) -> None:
        if not layer_sizes and empty_warning: #sometimes we just want to set these by hand
            print('WARNING: No layers specified, intialising placeholder network object.')
        else:
            self.layer_sizes = layer_sizes if layer_sizes else []
            self.num_layers = len(self.layer_sizes)
            self.biases = [np.random.randn(y,1) for y in self.layer_sizes[1:]]
            self.weights = [np.random.randn(y,x) for x,y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
          
    def feed_forward(self, a: np.ndarray) -> np.ndarray:
        for b, w in zip(self.biases, self.weights):
            a = NeuralNetwork.sigmoid(np.dot(w, a) + b)
        return a
    
    def save(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(f'{filename}.npz', biases=self.biases, weights=self.weights)


class NeuralNetController(BaseController, GeneticTrainable, ABC):
    @classmethod
    def load(cls, filename: str, controls: list[str]) -> 'NeuralNetController':
        neural_net = NeuralNetwork.load(filename=filename)
        controller = cls(controls, neural_net.layer_sizes, neural_net=neural_net)
        return controller

    def __init__(self, controls: list[str], layer_sizes: list[int], *, neural_net: NeuralNetwork = None) -> None:
        """Creates a new NeuralNetController object with a randomly intialised NN."""
        super().__init__(controls)
        self.network = neural_net if neural_net else NeuralNetwork(layer_sizes=layer_sizes)

        if not len(self.controls) == self.network.layer_sizes[-1]:
            raise ValueError('Inconsistent number of controls to output neurons')
    
    @abstractmethod
    def get_inputs(self, game_state: dict) -> list[float]:
        pass
    
    def return_controls(self, game_state: dict) -> list[int]:
        """
        Returns an ordered list of all inputs (controls) for the game to execute.
        The input list must be of the same size as the input layer of the underlying neural net. 
        """
        inputs = self.get_inputs(game_state)
        if not len(inputs) == self.network.layer_sizes[0]:
            raise ValueError('Inconsistent input size to input neurons')
        
        inputs = np.array(inputs).reshape(len(inputs), 1)
        index = np.argmax(self.network.feed_forward(inputs))
        return [self.controls[index]]
    
    def convert_to_float_array(self) -> np.ndarray:
        flat_weights = np.concatenate([w.flatten() for w in self.network.weights])
        flat_biases = np.concatenate([b.flatten() for b in self.network.biases])
        return np.concatenate([flat_weights, flat_biases])

    def update_genes(self, genes: np.ndarray) -> None:
        offset = 0
        self.network.weights = []
        for a,b in zip(self.network.layer_sizes[1:], self.network.layer_sizes[:-1]):
            self.network.weights.append(genes[offset: offset + a*b].reshape((a,b))) # ORDER MATTERS!
            offset += a*b

        self.network.biases = []
        for a in self.network.layer_sizes[1:]:
            self.network.biases.append(genes[offset: offset + a].reshape(a,1))
            offset += a
    
    def save(self, filename: str) -> None:
        self.network.save(filename=filename)

class SensoryMotorController(BaseController, GeneticTrainable, ABC):
    def __init__(self, controls: list[str], function_point_count: int, function_points: list[tuple[float]] = None) -> None:
        super().__init__(controls)
        if not function_points is None:
            self.function_points = function_points
        else:
            step_size = 1.0 / (function_point_count - 1)
            self.function_points = [(i * step_size, np.random.rand()) for i in range(function_point_count)]

    def return_controls(self) -> list[int]:
        #whats the best way to do this??
        pass

    def convert_to_float_array(self):
        pass

    def update_genes(self):
        pass

    
    
    
  