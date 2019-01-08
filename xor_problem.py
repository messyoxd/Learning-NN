from NeuralNetwork import *
import random
training_data = [

    [
        [0,1],
        [1]
    ],
    [
        [1,0],
        [1]
    ],
    [
        [0,0],
        [0]
    ],
    [
        [1,1],
        [0]
    ]

]

if __name__ == "__main__":

    nn = NeuralNetwork(2,2,1)

    for i in range(0,50000):
        data = random.choice(training_data)
        nn.train(data[0], data[1])

    print(nn.feedfoward([1,0])[0])
    print(nn.feedfoward([0,1])[0])
    print(nn.feedfoward([0,0])[0])
    print(nn.feedfoward([1,1])[0])
