from torch import nn
import torch


class ComplexMLP(nn.Module):
    """
        Define a five-layer MLP network for training a completely clean model.

        Parameters:

        dim_in: int - The dimension of the input layer (784)
        hidden1: int - The original dimension of the first hidden layer (1024)
        hidden2: int - The original dimension of the second hidden layer (512)
        hidden3: int - The original dimension of the third hidden layer (256)
        hidden4: int - The original dimension of the fourth hidden layer (128)
        hidden5: int - The original dimension of the fifth hidden layer (64)
        dim_out: int - The dimension of the output layer (10)
    """
    def __init__(self, dim_in, hidden1, hidden2, hidden3, hidden4, hidden5, dim_out):
        super(ComplexMLP, self).__init__()
        self.fc_layer1 = nn.Linear(dim_in, hidden1)
        self.fc_layer2 = nn.Linear(hidden1, hidden2)
        self.fc_layer3 = nn.Linear(hidden2, hidden3)
        self.fc_layer4 = nn.Linear(hidden3, hidden4)
        self.fc_layer5 = nn.Linear(hidden4, hidden5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden5, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])  # Flatten the input, each image of F-MNIST is 28x28=784 pixels
        x = self.relu(self.fc_layer1(x))
        x = self.relu(self.fc_layer2(x))
        x = self.relu(self.fc_layer3(x))
        x = self.relu(self.fc_layer4(x))
        x = self.relu(self.fc_layer5(x))
        x = self.output(x)
        return x

class ExtendedMLP(nn.Module):

    def __init__(self, original_model, dim_in, hidden1, hidden2, hidden3, hidden4, hidden5, dim_out):
        super(ExtendedMLP, self).__init__()

        # Define the five fully connected layers of the new model, adding new neurons to each layer, e.g., 10 neurons,
        # keeping the input and output dimensions unchanged.
        self.fc_layer1 = nn.Linear(dim_in, hidden1 + 100)
        self.fc_layer2 = nn.Linear(hidden1 + 100, hidden2 + 50)
        self.fc_layer3 = nn.Linear(hidden2 + 50, hidden3 + 20)
        self.fc_layer4 = nn.Linear(hidden3 + 20, hidden4 + 10)
        self.fc_layer5 = nn.Linear(hidden4 + 10, hidden5 + 5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden5 + 5, dim_out)

        with torch.no_grad():
            # Copy the weights and biases of the first layer neurons from the original model to corresponding positions
            # in the first layer of the new model.
            self.fc_layer1.weight[:hidden1, :] = original_model.fc_layer1.weight
            self.fc_layer1.bias[:hidden1] = original_model.fc_layer1.bias

            # Copy the weights and biases of the second layer neurons from the original model to corresponding positions
            # in the second layer of the new model.
            self.fc_layer2.weight[:hidden2, :hidden1] = original_model.fc_layer2.weight
            self.fc_layer2.bias[:hidden2] = original_model.fc_layer2.bias

            # Disconnect the new neurons in fc_layer2 from the original model's neurons;
            # Set their initial weights to zero and freeze them during training,
            # so they remain zero throughout the training process and after completion.
            self.fc_layer2.weight[hidden2:, :hidden1] = torch.zeros_like(self.fc_layer2.weight[hidden2:, :hidden1])
            self.fc_layer2.weight[:hidden2, hidden1:] = torch.zeros_like(self.fc_layer2.weight[:hidden2, hidden1:])

            # Copy the weights and biases of the third layer neurons from the original model to corresponding positions
            # in the third layer of the new model.
            self.fc_layer3.weight[:hidden3, :hidden2] = original_model.fc_layer3.weight
            self.fc_layer3.bias[:hidden3] = original_model.fc_layer3.bias
            # Disconnect the new neurons in fc_layer3 from the original model's neurons;
            # Set their initial weights to zero and freeze them during training,
            # so they remain zero throughout the training process and after completion.
            self.fc_layer3.weight[hidden3:, :hidden2] = torch.zeros_like(self.fc_layer3.weight[hidden3:, :hidden2])
            self.fc_layer3.weight[:hidden3, hidden2:] = torch.zeros_like(self.fc_layer3.weight[:hidden3, hidden2:])

            # Perform the same operations for fc_layer4.
            self.fc_layer4.weight[:hidden4, :hidden3] = original_model.fc_layer4.weight
            self.fc_layer4.bias[:hidden4] = original_model.fc_layer4.bias
            self.fc_layer4.weight[hidden4:, :hidden3] = torch.zeros_like(self.fc_layer4.weight[hidden4:, :hidden3])
            self.fc_layer4.weight[:hidden4, hidden3:] = torch.zeros_like(self.fc_layer4.weight[:hidden4, hidden3:])

            # Perform the same operations for fc_layer5.
            self.fc_layer5.weight[:hidden5, :hidden4] = original_model.fc_layer5.weight
            self.fc_layer5.bias[:hidden5] = original_model.fc_layer5.bias
            self.fc_layer5.weight[hidden5:, :hidden4] = torch.zeros_like(self.fc_layer5.weight[hidden5:, :hidden4])
            self.fc_layer5.weight[:hidden5, hidden4:] = torch.zeros_like(self.fc_layer5.weight[:hidden5, hidden4:])

            # Copy the weights and biases of the output layer neurons from the original model to corresponding positions
            # in the output layer of the new model.
            self.output.weight[:, :hidden5] = original_model.output.weight
            self.output.bias = original_model.output.bias

        # print("!\n")
        # print(self.fc_layer2.weight.data[hidden2:, :hidden1])
        # print("!!\n")
        # print(original_model.fc_layer1.weight)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.relu(self.fc_layer1(x))
        x = self.relu(self.fc_layer2(x))
        x = self.relu(self.fc_layer3(x))
        x = self.relu(self.fc_layer4(x))
        x = self.relu(self.fc_layer5(x))
        x = self.output(x)
        return x