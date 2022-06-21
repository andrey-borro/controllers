from controllers import NeuralNetController, SensoryMotorController
from refactored_flappy_square import NeuralFlappyController, launch_game, game_loop
import numpy as np

# NeuralNetwork(empty_warning=False)
# nn = NeuralNetwork.load('networks/lstest')
# print(nn.layer_sizes) 

 
 
# gc = NeuralNetController([45, 69, 78], NeuralNetwork([3,2,3]))
# print(gc.return_controls(np.array([0.2, 0.3, 0.4]).reshape(3,1))) 

# controllers = [NeuralFlappyController(['JUMP', 'NOT_JUMP'], [4,1,2]) for _ in range(50)]
# NeuralFlappyController.train_genetic_base( game_loop, 
#                                         controllers, 
#                                         num_training_epochs=30,
#                                         mutation_rate_1 = 0.1,
#                                         mutation_rate_2 = 0.1,
#                                         intermediate_save_filename='networks/test2'
#                                     )

# nfc = NeuralFlappyController.load('networks/test1', ['JUMP', 'NOT_JUMP'])
# launch_game([nfc])

# smc = SensoryMotorController(['JUMP', 'NOT_JUMP'], 8)