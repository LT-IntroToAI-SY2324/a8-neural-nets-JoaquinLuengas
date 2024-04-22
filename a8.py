from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

# # Training Data
# print("\n\nTraining XOR\n\n")
# xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

# xorn = NeuralNet(2, 1, 1)
# xorn.train(xor_training_data)
# print(xorn.test_with_expected(xor_training_data))
# print(xorn.get_ho_weights())

xor_data = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

# print(xor_data)
xor_nn = NeuralNet(2, 2, 1)
xor_nn.train(xor_data, iters = 10000, print_interval = 1000)
print(xor_nn.test_with_expected(xor_data))
print(xor_nn.evaluate([1, 1]))