from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

# # Training Data
# print("\n\nTraining XOR\n\n")
# xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

# xorn = NeuralNet(2, 1, 1)
# xorn.train(xor_training_data)
# print(xorn.test_with_expected(xor_training_data))
# print(xorn.get_ho_weights())

# xor_data = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]

# # print(xor_data)
# xor_nn = NeuralNet(2, 2, 1) # Number of inputs, hidden nodes, outputs
# # Original number of hidden nodes: 2
# xor_nn.train(xor_data, iters = 10000, print_interval = 1000)
# print(xor_nn.test_with_expected(xor_data))
# print(xor_nn.evaluate([1, 1]))

voter_data = [
    ([0.9, 0.6, 0.8, 0.3, 0.1], [1]),
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1]),
    ([0.7, 0.2, 0.4, 0.6, 0.3], [1]), 
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8], [0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6], [0])
]

voter_nn = NeuralNet(5, 1, 1)# Original Hidden Nodes:1

voter_nn.train(voter_data)

print(voter_nn.test_with_expected(voter_data))

print(" ")

print(voter_nn.test((
    [1.0, 1.0, 1.0, 0.1, 0.1],
    [0.5, 0.2, 0.1, 0.7, 0.7],
    [0.8, 0.3, 0.3, 0.3, 0.8], 
    [0.8, 0.3, 0.3, 0.8, 0.3],
    [0.9, 0.8, 0.8, 0.3, 0.6]
)))

print(" ")

print(voter_nn.evaluate([1, 1, 1, 1, 0.1]))