import torch
import torch.nn as nn
from torchinfo import summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def xavier_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)

class Mlp(nn.Module):
    def __init__(self):

        super().__init__()

        self.flatten = nn.Flatten()
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
            nn.Linear(128, 1),      #6
        )

        self.linear_tanh_stack.apply(xavier_init)
        self.linear_tanh_stack = self.linear_tanh_stack.to(torch.float64)

    def forward(self, x):
        x = x.flatten()
        logits = self.linear_tanh_stack(x)
        return logits

#    def wheights_init():


#Execute as main for testing
if __name__ == "__main__":

    print("Training device is: ", DEVICE)

    model = Mlp().to(torch.float64)
    test_tensor = torch.cat([torch.zeros(1,1,dtype=torch.float64),
        torch.zeros(1,1,dtype=torch.float64),
        torch.zeros(1,1,dtype=torch.float64)])
    print(test_tensor)
    summary(model, input_data=test_tensor
    )

    #Test gradients
    t = torch.tensor([[0.1]], requires_grad=True, dtype=torch.float64)
    x = torch.tensor([[0.2]], requires_grad=True, dtype=torch.float64)
    v = torch.tensor([[0.3]], requires_grad=True, dtype=torch.float64)

    model.to(DEVICE)
    input = torch.cat([t, x, v], dim=-1).to(DEVICE)
    f = model(input)
    f_delta_t = torch.autograd.grad(f, t, create_graph=True)[0]
    f_delta_x = torch.autograd.grad(f, x, create_graph=True)[0]
    f_delta_v = torch.autograd.grad(f, v, create_graph=True)[0]

    #Check individual values
    print("f = ", f.item())
    print("f_t = ", f_delta_t)
    print("f_x = ", f_delta_x)
    print("f_v = ", f_delta_v)
