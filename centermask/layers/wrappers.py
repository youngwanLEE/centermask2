# Author, Sangrok Lee, github.com/lsrock1

import torch


# from facebook detectron2
class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class MaxPool2d(torch.nn.MaxPool2d):
    """
    A wrapper around :class:`torch.nn.MaxPool2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._make_iteratable()

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], x.shape[1]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            
            return empty

        x = super().forward(x)
        
        return x

    def _make_iteratable(self):
        if not isinstance(self.padding, list):
            self.padding = [self.padding, self.padding]

        if not isinstance(self.dilation, list):
            self.dilation = [self.dilation, self.dilation]

        if not isinstance(self.kernel_size, list):
            self.kernel_size = [self.kernel_size, self.kernel_size]

        if not isinstance(self.stride, list):
            self.stride = [self.stride, self.stride]


class Linear(torch.nn.Linear):
    """
    A wrapper around :class:`torch.nn.Linear` to support empty inputs and more features.
    Because of https://github.com/pytorch/pytorch/issues/34202
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if x.numel() == 0:
            output_shape = [x.shape[0], self.weight.shape[0]]
            
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        return x


def Max(x):
    """
    A wrapper around torch.max in Spatial Attention Module (SAM) to support empty inputs and more features.
    """
    if x.numel() == 0:
        output_shape = [x.shape[0], 1, x.shape[2], x.shape[3]]
        empty = _NewEmptyTensorOp.apply(x, output_shape)
        return empty
    return torch.max(x, dim=1, keepdim=True)[0]