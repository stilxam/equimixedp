import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import PyTree, Float, Int, Bool, Array
from typing import Callable


def _cast_to_dtype(dtype: jnp.dtype) -> Callable:
    def _cast(x):
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return _cast


@eqx.filter_jit
def cast_tree(tree: PyTree, dtype: jnp.dtype) -> PyTree:
    """
    Casts all float arrays in a pytree to the specified dtype.

    :param tree: The pytree to cast.
    :type tree: PyTree
    :param dtype: The target dtype.
    :return: The casted pytree.
    :rtype: PyTree
    """
    return jax.tree.map(_cast_to_dtype(dtype), tree)


def cast_to_float16(tree: PyTree) -> PyTree:
    """Casts the pytree to float16."""
    return cast_tree(tree, jnp.float16)


def cast_to_bfloat16(tree: PyTree) -> PyTree:
    """Casts the pytree to bfloat16."""
    return cast_tree(tree, jnp.bfloat16)


def cast_to_float32(tree: PyTree) -> PyTree:
    """Casts the pytree to float32."""
    return cast_tree(tree, jnp.float32)


def cast_to_full_precision(tree: PyTree) -> PyTree:
    """Casts the pytree to full precision (float32)."""
    return cast_tree(tree, jnp.float32)


def cast_to_half_precision(tree: PyTree, half_dtype: jnp.dtype) -> PyTree:
    """
    Casts the pytree to half precision using the specified dtype.

    :param tree: The pytree to cast.
    :type tree: PyTree
    :param half_dtype: The half-precision dtype (e.g., jnp.float16 or jnp.bfloat16).
    :type half_dtype: jnp.dtype
    :return: The casted pytree.
    :rtype: PyTree
    """
    return cast_tree(tree, half_dtype)


def cast_function(input_dtype: jnp.dtype, output_dtype: jnp.dtype = None) -> Callable:
    """
    Decorator to cast function inputs and outputs to specified dtypes.

    :param input_dtype: Dtype for inputs.
    :param output_dtype: Dtype for outputs. Defaults to input_dtype.
    :return: The decorated function.
    :rtype: callable
    """
    def decorator(foo):
        if output_dtype is None:
            output_dtype_ = input_dtype
        else:
            output_dtype_ = output_dtype

        def wrapper(*args, **kwargs):
            args_cast = [cast_tree(arg, input_dtype) for arg in args]
            kwargs_cast = {
                key: cast_tree(value, input_dtype) for key, value in kwargs.items()
            }

            results = foo(*args_cast, **kwargs_cast)

            if isinstance(results, tuple):
                results_converted = tuple(cast_tree(res, output_dtype_) for res in results)
            elif eqx.is_array(results):
                results_converted = cast_tree(results, output_dtype_)
            else:
                results_converted = results
            return results_converted

        return wrapper
    return decorator


def force_full_precision(output_dtype: jnp.dtype = jnp.float32) -> Callable:
    """
    Decorator to force function to use full precision.

    :param output_dtype: Dtype for outputs. Defaults to float32.
    :return: The decorated function.
    :rtype: callable
    """
    return cast_function(jnp.float32, output_dtype)
