import torch
import torch.nn as nn
from torchinfo import summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mlp(nn.Module):
    def __init__(self):

        super().__init__()

        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(3, 128),      #1
            nn.Tanh(),
            nn.Linear(128, 128),    #2
            nn.Tanh(),
            nn.Linear(128, 128),    #3
            nn.Tanh(),
            nn.Linear(128, 128),    #4
            nn.Tanh(),
            nn.Linear(128, 128),    #5
            nn.Tanh(),
            nn.Linear(128, 3),      #6
        )

    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits

#    def wheights_init():


#Execute as main for testing
if __name__ == "__main__":

    print("Training device is: ", DEVICE)

    model = Mlp()
    summary(model, input_size=(1, 1, 3))
