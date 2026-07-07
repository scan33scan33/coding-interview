import numpy as np
import math


class LinearLayer:
    def __init__(self, input_size, output_size):
        # w x + b
        self.w = np.random.randn(input_size, output_size) * math.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))

        self.activation = None
        self.grad_w = None
        self.grad_b = None

    def forward(self, x):
        # x is (batch_size, input_size)
        self.x = x
        return np.matmul(x, self.w) + self.b

    def backward(self, grad):
        # grad is (batch_size, output_size)
        # y = w x + b
        # dL / dx = dL/dy * dy/dx = grad * w 
        # dL / dw = grad x 
        # dL / db = grad ones
        self.grad_w = np.matmul(self.x.T, grad)
        self.grad_b = np.sum(grad, axis=0, keepdims=1)

        return np.matmul(grad, self.w.T)

    def update_grad(self, learning_rate):
        self.w -= learning_rate * self.grad_w
        self.b -= learning_rate * self.grad_b

HIDDEN_DIM = 2
LEARNING_RATE = 0.001


class MLP:
    def __init__(self, num_layers=3, hidden_dim=HIDDEN_DIM):
        self.hidden_layers = [LinearLayer(hidden_dim, hidden_dim) for i in range(num_layers)]
        self.output_layer = LinearLayer(hidden_dim, 1)
        self.hidden_activations = [None] * num_layers

    def forward(self, x):
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i].forward(x)
            x = np.maximum(x, 0)
            self.hidden_activations[i] = x
        return np.reshape(self.output_layer.forward(x), [-1])

    def backward(self, grad):
        grad = self.output_layer.backward(grad)
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            # Mask by ReLU'(z_i) BEFORE the linear backward: the mask belongs to
            # this layer's pre-activation, and the linear backward must see the
            # masked gradient so grad_w picks up ReLU'. Applying the mask after
            # backward() masks the wrong tensor (dL/da_{i-1}) and leaves grad_w
            # unmasked. Verified against finite differences.
            grad = grad * (self.hidden_activations[i] > 0)
            grad = self.hidden_layers[i].backward(grad)


    def update_grad(self):
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].update_grad(LEARNING_RATE)
        self.output_layer.update_grad(LEARNING_RATE)
        


class MSELoss:
    @staticmethod
    def loss(pred_y, true_y):
        diff = pred_y - true_y
        return diff * diff

    @staticmethod
    def grad(pred_y, true_y):
        return 2 * (pred_y - true_y)



x = np.array([[-1., -1.], [-1., 1], [1., 1.], [1., -1.]])
y = np.array([1, 1, 0, 0])


model = MLP()
for iter in range(100):
    pred_y = model.forward(x)
    grad = MSELoss.grad(pred_y, y)
    print(f"loss: {np.sum(MSELoss.loss(pred_y, y))}, grad: {grad} ")
    model.backward(np.expand_dims(grad, 1))
    model.update_grad()
