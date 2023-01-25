from .gradients_core import GradFunc
from .gradients_core import apply_dim_transformations
import torch
from torch import nn
import numpy as np

import torch
from torch.nn.modules.utils import _single, _pair, _triple


def _grad_input_padding(grad_output, input_size, stride, padding, kernel_size, dilation=None):
    if dilation is None:
        # For backward compatibility
        warnings.warn("_grad_input_padding 'dilation' argument not provided. Default of 1 is used.")
        dilation = [1] * len(stride)

    input_size = list(input_size)
    k = grad_output.dim() - 2

    if len(input_size) == k + 2:
        input_size = input_size[-k:]
    if len(input_size) != k:
        raise ValueError("input_size must have {} elements (got {})"
                         .format(k + 2, len(input_size)))

    def dim_size(d):
        return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] + 1
                + dilation[d] * (kernel_size[d] - 1))

    min_sizes = [dim_size(d) for d in range(k)]
    max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
    for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
        if size < min_size or size > max_size:
            raise ValueError(
                ("requested an input grad size of {}, but valid sizes range "
                 "from {} to {} (for a grad_output of {})").format(
                     input_size, min_sizes, max_sizes,
                     grad_output.size()[2:]))

    return tuple(input_size[d] - min_sizes[d] for d in range(k))

def conv_to_autograd(elem):
    from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
    from syft.execution.placeholder import PlaceHolder

    return AutogradTensor().on(PlaceHolder().on(elem, wrap=False), wrap=False)

def conv2d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):
    r"""
    Computes the gradient of conv2d with respect to the weight of the convolution.
    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
    Examples::
        >>> input = torch.randn(1,1,3,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1,2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv2d_weight(input, weight.shape, grad_output)
    """
    from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
    from syft.execution.placeholder import PlaceHolder
    
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    in_channels = input.size(1)
    # in_channels = input.shape[1]
    out_channels = grad_output.size(1)
    min_batch = input.size(0)

    # print("in_channels", in_channels, " groups ", groups, " in_channels // groups", in_channels // groups)
    grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1,1)
    # print("grad_output.size(0)",grad_output.size(0),"grad_output.size(1)",grad_output.size(1), "grad_output.size(2)",grad_output.size(2),"grad_output.size(3)",grad_output.size(3))
    grad_output = grad_output.contiguous().view(
        grad_output.size(0) * grad_output.size(1), 1, grad_output.size(2),
        grad_output.size(3))

    
    input = input.contiguous().view(1, input.size(0) * input.size(1),
                                    input.size(2), input.size(3))

    grad_weight = torch.conv2d(input, grad_output, None, dilation, padding,
                               stride, in_channels * min_batch)

    grad_weight = grad_weight.contiguous().view(
        min_batch, grad_weight.size(1) // min_batch, grad_weight.size(2),
        grad_weight.size(3))
    # print("------------------------conv2d_weight: end-------------------")
    return grad_weight.sum(dim=0).contiguous().view(
        in_channels // groups, out_channels,
        grad_weight.size(2), grad_weight.size(3)).transpose(0, 1).narrow(
            2, 0, weight_size[2]).narrow(3, 0, weight_size[3])


def conv2d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1):
    r"""
    Computes the gradient of conv2d with respect to the input of the convolution.
    This is same as the 2D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.
    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
    Examples::
        >>> input = torch.randn(1,1,3,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1,2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv2d_input(input.shape, weight, grad_output)
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    kernel_size = (weight.size(2), weight.size(3))

    if input_size is None:
        raise ValueError("grad.conv2d_input requires specifying an input_size")

    grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
                                             padding, kernel_size, dilation)
    return torch.conv_transpose2d(
        grad_output, weight, None, stride, padding, grad_input_padding, groups,
        dilation)



# from nn import grad

class CloneBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad.clone()
        return (grad_self_,)

class AddBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self = grad.copy()
        grad_other = grad.copy() if isinstance(self.self_, type(self.other)) else None

        if not isinstance(self.other.child, int):
            if self.self_.shape != self.other.shape:
                grad_self, grad_other = apply_dim_transformations(
                    grad_self, grad_other, self.self_.shape, self.other.shape
                )


        return (grad_self, grad_other)


class SubBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self = grad.copy()
        grad_other = grad * -1 if isinstance(self.self_, type(self.other)) else None

        if not isinstance(self.other.child, int):
            if self.self_.shape != self.other.shape:
                grad_self, grad_other = apply_dim_transformations(
                    grad_self, grad_other, self.self_.shape, self.other.shape
                )
        return (grad_self, grad_other)

class MmBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self_ = grad @ self.other.t()
        grad_other = self.self_.t() @ grad if isinstance(self.self_, type(self.other)) else None
        return (grad_self_, grad_other)



class SumBackward(GradFunc):
    """Tensor Sum backward gradient class"""

    def __init__(self, self_, **kwargs):
        super().__init__(self, self_)
        self.self_ = self_
        self.kwargs = kwargs

    def gradient(self, grad):
        if grad.shape != self.self_.shape:
            grad = grad.reshape([-1, 1])

        return ((self.self_ * 0 + 1) * grad,)


class MeanBackward(GradFunc):
    """Tensor Mean backward gradient class"""

    def __init__(self, self_, dim=None):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        if grad.shape != self.self_.shape:
            grad = grad.reshape([-1, 1])
        numel = self.self_.numel()
        return ((self.self_ * 0 + 1) * grad / numel,)


class ReshapeBackward(GradFunc):
    """Tensor reshape backward gradient class"""

    def __init__(self, self_, *dims):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        if grad.shape != self.self_.shape:
            grad = grad.reshape(self.self_.shape)
        return ((self.self_ * 0 + 1) * grad,)


class AsinBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * (-self.self_ * self.self_ + 1).rsqrt()
        return (grad_self_,)



class LogBackward(GradFunc):
    """Log backward gradient class"""

    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad / self.self_
        return (grad_self_,)

class SoftmaxBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_ - self_.max(dim = 1, keepdim = True)[0]

    def gradient(self, grad):
        
        inputExp = self.self_.clone().exp()
        expSums = inputExp.clone().sum(dim = -1, keepdim = True)

        grad_inputExp = grad / expSums
       
        grad_expSums = grad * (inputExp / (expSums ** 2) * -1)

        grad_expSums_cpy = grad_expSums.sum(dim = -1, keepdim = True)


        if grad_expSums_cpy.shape != inputExp.shape:
          grad_expSums_cpy = grad_expSums_cpy.reshape(-1, 1)

        
        grad_expSums = ((inputExp * 0) + 1) * grad_expSums_cpy
        grad_expSums = inputExp * grad_expSums

        grad_inputExp = grad_inputExp * self.self_.exp()

        grad_self_ = grad_expSums + grad_inputExp

    
        return (grad_self_,)


class ExpBackward(GradFunc):
    """Exp backward gradient class"""

    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * self.self_.exp()

        return (grad_self_,)


class MulBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self_ = grad * self.other
        grad_other = grad * self.self_ if isinstance(self.self_, type(self.other)) else None
        return (grad_self_, grad_other)


class NegBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        

        grad_self_ = grad * -1
        return (grad_self_,)


class DivBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        # assert isinstance(self.other, int)
        grad_self_ = grad / self.other
        grad_other = grad * (self.self_ / (self.other ** 2) * -1)

        return (grad_self_, grad_other,)
    
# class PermuteBackward(GradFunc):
#     def __init__(self, self_, arg1, arg2, arg3):
#         super().__init__(self, self_, arg1, arg2, arg3)
#         self.self_ = self_
#         self.arg1 = arg1
#         self.arg2 = arg2
#         self.arg3 = arg3

#     def gradient(self, grad):
#         return (grad.permute(self.arg1, self.arg2, self.arg3),)

# class ContiguousBackward(GradFunc):
#     def __init__(self, self_):
#         super().__init__(self, self_)
#         self.self_ = self_

#     def gradient(self, grad):

#         return (grad,)

class FlattenBackward(GradFunc):
    def __init__(self, self_, start_dim = 0, end_dim = -1):
        super().__init__(self, self_)
        self.self_ = self_
        self.start_dim = start_dim
        self.end_dim = end_dim

    def gradient(self, grad):
        grad_ = grad.reshape(self.self_.shape)

        return (grad_,)

class FloatBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        return (grad,)

class DoubleBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        return (grad,)

def conv_to_size(tensor, shape):
    return [tensor.size(i) for i in range(len(shape))]

class Conv2dBackward(GradFunc):
    def __init__(self, self_, weight, bias, *params):
        super().__init__(self, self_, weight, bias, *params)
        self.self_ = self_
        self.weight = weight
        self.bias = bias
        self.params = params
        # print("-----------conve2d constructor self_.shape", self_.shape)

    def gradient(self, grad):
        from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
        from syft.execution.placeholder import PlaceHolder

        grad_self_ = conv2d_input(self.self_.shape, self.weight, grad, *self.params)
        grad_weight = conv2d_weight(self.self_, self.weight.shape, grad, *self.params)
        grad_bias = grad.sum(dim=0).sum(dim=1).sum(dim=1)
        return (grad_self_, grad_weight, grad_bias) 

class Max_pool2dBackward(GradFunc):
    def __init__(self, self_, *params):
        super().__init__(self, self_, *params)
        self.self_ = self_.copy()
        self.params = params

        kernel_size, stride, padding, dilation = self.params[:4]
        return_indices = True
        ceil_mode = self.params[-1]

        _, self.indices = torch.nn.functional.max_pool2d_with_indices(self.self_, kernel_size, stride, padding, dilation, ceil_mode=ceil_mode, return_indices = return_indices)


    def gradient(self, grad):
        

        kernel_size, stride, padding, dilation = self.params[:4]
        
        grad_self_ = torch.nn.functional.max_unpool2d(grad, self.indices, kernel_size, stride, padding)


        return  (grad_self_,) 
        

class PadBackward(GradFunc):
    def __init__(self, self_, pad, mode, value):
        super().__init__(self, self_, pad, mode, value)
        self.self_ = self_
        self.pad = pad
        self.value = value
        self.mode = mode

    def gradient(self, grad):
        grad_self = grad.copy()

        if len(grad.shape) == 2:
            grad_self = grad_self[ self.pad[2] : self.pad[3], self.pad[0] : self.pad[1]]
        

        return (grad_self)

class MaxBackward(GradFunc):
    def __init__(self, self_, **kwargs):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        if grad.shape != self.self_.shape:
            grad = grad.reshape([-1, 1])
        
        return ((self.self_ * 0 + 1) * grad,)


class PowBackward(GradFunc):
    def __init__(self, self_, power):
        super().__init__(self, self_, power)
        self.self_ = self_
        self.power = power

    def gradient(self, grad):
        power = self.power
        return (power * self.self_ ** (power - 1) * grad,)


class MatmulBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self_ = grad @ self.other.t()
        # grad_other = self.self_.t() @ grad if isinstance(self.self_, type(self.other)) else None
        grad_other = grad.t() @ self.self_ if isinstance(self.self_, type(self.other)) else None
        return (grad_self_, grad_other.t())


class TBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        return (grad.t(),)


class SigmoidBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * self.self_.sigmoid() * (1 - self.self_.sigmoid())
        return (grad_self_,)


class SinBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * self.self_.cos()
        return (grad_self_,)


class SinhBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * self.self_.cosh()
        return (grad_self_,)


# class SqrtBackward(GradFunc):
#     def __init__(self, self_):
#         super().__init__(self, self_)
#         self.self_ = self_
#
#     def gradient(self, grad):
#         TODO: Broken as of Garbage Collection for `AutoGradTensor` (#3387)
#         grad_self_ = grad / (2 * self.result)
#         return (grad_self_,)


class TanhBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * (1 - self.self_.tanh() ** 2)
        return (grad_self_,)


class ReluBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        zero = self.self_ * 0
        gt_zero = self.self_ > zero

        return (gt_zero * grad,)