import torch

def kaiming_reinitialize(model: torch.nn.Module):
    """
    Reinitializes all the layers in the given model using kaiming initialization.
    :param model: the model to be reinitialized
    """
    model.apply(kaiming_init_resnet_module)
    
def kaiming_init_resnet_module(nn_module: torch.nn.Module):
    """
    Initializes the parameters of a resnet module in the following way:
        - Conv2d: weights are initialized using xavier normal initialization and bias are initialized to zero
        - Linear: same as Conv2d
        - BatchNorm2d: bias are initialized to 0, weights are initialized to 1
    :param nn_module: an instance ot torch.nn.Module to be initialized
    """

    if isinstance(nn_module, torch.nn.Conv2d) or isinstance(nn_module, torch.nn.Linear):
        if isinstance(nn_module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(nn_module.weight, nonlinearity="relu")
        else:   # the only linear layer in a resnet is the output layer
            torch.nn.init.kaiming_normal_(nn_module.weight, nonlinearity="linear")
        if nn_module.bias is not None:
            torch.nn.init.constant_(nn_module.bias, 0.0)

    if isinstance(nn_module, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(nn_module.weight, 1.0)
        torch.nn.init.constant_(nn_module.bias, 0.0)