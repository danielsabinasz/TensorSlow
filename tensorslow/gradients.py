import numpy as np
from tensorslow.operations import *

# A dictionary that will map operations to gradient functions
_gradient_registry = {}


class RegisterGradient:
    """A decorator for registering the gradient function for an op type.
    """

    def __init__(self, op_type):
        """Creates a new decorator with `op_type` as the Operation type.
        Args:
          op_type: The name of an operation
        """
        try:
            self._op_type = eval(op_type)
        except:
            self._op_type = eval(op_type)

    def __call__(self, f):
        """Registers the function `f` as gradient function for `op_type`."""
        _gradient_registry[self._op_type] = f
        return f


@RegisterGradient("add")
def _add_gradient(op, grad):
    """Computes the gradients for `add`.

    Args:
      op: The `add` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `add` op.

    Returns:
      Gradients with respect to the input of `add`.
    """
    a = op.inputs[0]
    b = op.inputs[1]

    grad_wrt_a = grad
    while np.ndim(grad_wrt_a) > len(a.shape):
        grad_wrt_a = np.sum(grad_wrt_a, axis=0)
    for axis, size in enumerate(a.shape):
        if size == 1:
            grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)

    grad_wrt_b = grad
    while np.ndim(grad_wrt_b) > len(b.shape):
        grad_wrt_b = np.sum(grad_wrt_b, axis=0)
    for axis, size in enumerate(b.shape):
        if size == 1:
            grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)

    return [grad_wrt_a, grad_wrt_b]


@RegisterGradient("matmul")
def _matmul_gradient(op, grad):
    """Computes the gradients for `matmul`.

    Args:
      op: The `matmul` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `matmul` op.

    Returns:
      Gradients with respect to the input of `matmul`.
    """

    A = op.inputs[0]
    B = op.inputs[1]

    return [grad.dot(B.T), A.T.dot(grad)]


@RegisterGradient("sigmoid")
def _sigmoid_gradient(op, grad):
    """Computes the gradients for `sigmoid`.

    Args:
      op: The `sigmoid` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `sigmoid` op.

    Returns:
      Gradients with respect to the input of `sigmoid`.
    """

    sigmoid = op.output

    return grad * sigmoid * (1 - sigmoid)


@RegisterGradient("softmax")
def _softmax_gradient(op, grad):
    """Computes the gradients for `softmax`.

    Args:
      op: The `softmax` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `softmax` op.

    Returns:
      Gradients with respect to the input of `softmax`.
    """

    softmax = op.output
    return (grad - np.reshape(
        np.sum(grad * softmax, 1),
        [-1, 1]
    )) * softmax



@RegisterGradient("log")
def _log_gradient(op, grad):
    """Computes the gradients for `log`.

    Args:
      op: The `log` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `log` op.

    Returns:
      Gradients with respect to the input of `log`.
    """
    x = op.inputs[0]
    return grad/x


@RegisterGradient("multiply")
def _multiply_gradient(op, grad):
    """Computes the gradients for `multiply`.

    Args:
      op: The `multiply` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `multiply` op.

    Returns:
      Gradients with respect to the input of `multiply`.
    """

    A = op.inputs[0]
    B = op.inputs[1]

    return [grad * B, grad * A]


@RegisterGradient("reduce_sum")
def _reduce_sum_gradient(op, grad):
    """Computes the gradients for `reduce_sum`.

    Args:
      op: The `reduce_sum` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `reduce_sum` op.

    Returns:
      Gradients with respect to the input of `reduce_sum`.
    """
    A = op.inputs[0]

    output_shape = np.array(A.shape)
    output_shape[op.axis] = 1
    tile_scaling = A.shape // output_shape
    grad = np.reshape(grad, output_shape)
    return np.tile(grad, tile_scaling)


@RegisterGradient("negative")
def _negative_gradient(op, grad):
    """Computes the gradients for `negative`.

    Args:
      op: The `negative` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `negative` op.

    Returns:
      Gradients with respect to the input of `negative`.
    """
    return -grad
