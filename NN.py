import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.bias = np.ones((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias
    
    def lin_rect_act(self):
        self.output_act = np.maximum(0, self.output)
