class Module:
    """torchok.nn.Module basis class.

    Attributes:
        _parameters (list): stores parameters of current Module.
        submodules (dict): dictionary of subsequential modules.

    Example:
        >>> class Linear(nn.Module)
    """
    def __init__(self):
        self._parameters = []
        self.submodules = {}

    def parameters(self):
        params = list(self._parameters)
        for module in self.submodules.values():
            params.extend(module.parameters())
        return params
    
    def add_parameter(self, param):
        self._parameters.append(param)

    # Essential!
    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.add_module(name, value)
        elif hasattr(value, 'requires_grad'):
            self.add_parameter(value)
        object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward() not implemented yet")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
