import jax
import jax.numpy as jnp
import equinox as eqx

import optax

from . import casting as cast
from . import loss_scaling as loss_scaling

from jaxtyping import PyTree, Bool, Array, Float, Int
from typing import Callable


@eqx.filter_jit
def select_tree(pred: Bool[Array, ""], a: PyTree, b: PyTree) -> PyTree:
    """
    Selects elements from one of two pytrees based on a scalar boolean predicate.

    This function traverses two input pytrees (`a` and `b`) and selects elements
    from either `a` or `b` based on the value of the scalar boolean `pred`. If
    `pred` is `True`, elements from `a` are selected; otherwise, elements from `b`
    are selected. Non-array elements in the pytrees are taken directly from `a`.

    :param pred: A scalar boolean array (`jnp.bool_`) that determines which pytree to select elements from.
    :type pred: jnp.ndarray
    :param a: The first pytree to select elements from.
    :type a: PyTree
    :param b: The second pytree to select elements from.
    :type b: PyTree
    :return: A new pytree with elements selected from `a` or `b` based on `pred`.
    :rtype: PyTree
    :raises AssertionError: If `pred` is not a scalar boolean array (`jnp.bool_`).
    """
    """Selects a pytree based on the given predicate."""
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"

    def _select_leaf(x1, x2):
        if eqx.is_array(x1):
            return jax.lax.select(pred, x1, x2)
        else:
            return x1

    return jax.tree.map(_select_leaf, a, b)


@eqx.filter_jit
def optimizer_update(
    model: PyTree,
    optimizer: optax.GradientTransformation,
    optimizer_state: PyTree,
    grads: PyTree,
    grads_finite: Bool[Array, ""],
):
    """
    Updates the model and optimizer state using the gradients, but only if gradients are finite.

    :param model: The current model.
    :type model: PyTree
    :param optimizer: The optimizer.
    :type optimizer: optax.GradientTransformation
    :param optimizer_state: The current optimizer state.
    :type optimizer_state: PyTree
    :param grads: The gradients.
    :type grads: PyTree
    :param grads_finite: Whether the gradients are finite.
    :type grads_finite: Bool
    :return: (new_model, new_optimizer_state)
    :rtype: tuple
    """
    # optimizer step
    updates, new_optimizer_state = optimizer.update(
        grads, optimizer_state, eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)

    # only apply updates to the model and optimizer state if gradients are finite
    new_model = select_tree(grads_finite, new_model, model)
    optimizer_state = select_tree(grads_finite, new_optimizer_state, optimizer_state)

    return new_model, optimizer_state

def _prepare_mixed_precision_call(
    func: Callable,
    scaling: loss_scaling.DynamicLossScaling,
    has_aux: bool,
    use_mixed_precision: bool,
    half_dtype: jnp.dtype,
    args: tuple,
    kwargs: dict,
) -> tuple[Callable, tuple, dict]:
    """
    Prepares a function call for mixed-precision execution by casting inputs and
    scaling the function if required.

    :param func: The function to be called.
    :type func: callable
    :param scaling: The loss scaling object.
    :type scaling: loss_scaling.DynamicLossScaling
    :param has_aux: Whether the function returns auxiliary data.
    :type has_aux: bool
    :param use_mixed_precision: If True, inputs are cast to half-precision and
        the function is scaled.
    :type use_mixed_precision: bool
    :param half_dtype: The half-precision data type to use.
    :type half_dtype: jnp.dtype
    :param args: Positional arguments for the function.
    :type args: tuple
    :param kwargs: Keyword arguments for the function.
    :type kwargs: dict
    :return: A tuple containing the (potentially scaled) function, and the
        (potentially cast) arguments.
    :rtype: tuple
    """
    if use_mixed_precision:
        args_cast = tuple([cast.cast_to_half_precision(x, half_dtype) for x in args])
        kwargs_cast = {
            k: cast.cast_to_half_precision(v, half_dtype) for k, v in kwargs.items()
        }
        func_scaled = loss_scaling.scaled(func, scaling, has_aux=has_aux)
    else:
        args_cast = args
        kwargs_cast = kwargs
        func_scaled = func
    return func_scaled, args_cast, kwargs_cast


def filter_grad(
    func: Callable,
    scaling: loss_scaling.DynamicLossScaling,
    has_aux: bool = False,
    use_mixed_precision: bool = True,
    half_dtype: jnp.dtype = jnp.bfloat16,
) -> Callable:
    """
    Filters the gradients of a function based on a predicate.

    This function computes the gradients of the given function `func` with respect
    to its arguments (`args` and `kwargs`). It then filters the gradients based on
    a predicate function that checks whether the gradients are finite. The filtered
    gradients are returned as a new pytree.

    :param func: The function to compute gradients for. This function must only use pytrees as parameters!
    :type func: callable
    :param scaling: The loss scaling object.
    :type scaling: loss_scaling.DynamicLossScaling
    :param has_aux: If True, the function is expected to return auxiliary values along with the gradients.
    :type has_aux: bool
    :param use_mixed_precision: If True, the function will be cast to half precision. Defaults to True.
    :type use_mixed_precision: bool
    :param half_dtype: The half-precision dtype. Defaults to jnp.bfloat16.
    :type half_dtype: jnp.dtype
    :return: A function that computes the filtered gradients of `func`. It returns the grad, the new loss scaling, and a boolean indicating whether the gradients are finite (and the aux-value if has_aux is true).
    :rtype: callable
    """

    def wrapper(*args, **kwargs):
        func_scaled, args_cast, kwargs_cast = _prepare_mixed_precision_call(
            func, scaling, has_aux, use_mixed_precision, half_dtype, args, kwargs
        )

        dfunc_scaled = eqx.filter_grad(func_scaled, has_aux=has_aux)

        if has_aux:
            grad, aux = dfunc_scaled(*args_cast, **kwargs_cast)
            if use_mixed_precision:
                grads_finite = loss_scaling.all_finite(grad)
                loss_scaling_new = scaling.adjust(grads_finite)
                grad = loss_scaling_new.unscale(grad)
            else:
                grads_finite = jnp.bool_(True)
                loss_scaling_new = scaling
            return loss_scaling_new, grads_finite, grad, aux
        else:
            grad = dfunc_scaled(*args_cast, **kwargs_cast)
            if use_mixed_precision:
                grad = cast.cast_to_full_precision(grad)
                grads_finite = loss_scaling.all_finite(grad)
                loss_scaling_new = scaling.adjust(grads_finite)
                grad = loss_scaling_new.unscale(grad)
            else:
                grads_finite = jnp.bool_(True)
                loss_scaling_new = scaling
            return loss_scaling_new, grads_finite, grad

    return wrapper


def calculate_scaled_grad(
    func: Callable,
    scaling: loss_scaling.DynamicLossScaling,
    has_aux: bool = False,
    use_mixed_precision: bool = True,
    half_dtype: jnp.dtype = jnp.bfloat16,
) -> Callable:
    """
    Calculates the scaled gradient of a function.

    :param func: The function to compute gradients for.
    :type func: callable
    :param scaling: The loss scaling object.
    :type scaling: loss_scaling.DynamicLossScaling
    :param has_aux: If True, the function is expected to return auxiliary values along with the gradients.
    :type has_aux: bool
    :param use_mixed_precision: If True, the function will be cast to half precision. Defaults to True.
    :type use_mixed_precision: bool
    :param half_dtype: The half-precision dtype. Defaults to jnp.bfloat16.
    :type half_dtype: jnp.dtype
    :return: A function that computes the scaled gradients of `func`.
    :rtype: callable
    """
    def wrapper(*args, **kwargs):
        func_scaled, args_cast, kwargs_cast = _prepare_mixed_precision_call(
            func, scaling, has_aux, use_mixed_precision, half_dtype, args, kwargs
        )

        dfunc_scaled = eqx.filter_value_and_grad(func_scaled, has_aux=has_aux)
        return dfunc_scaled(*args_cast, **kwargs_cast)

    return wrapper


def filter_value_and_grad(
    func: Callable,
    scaling: loss_scaling.DynamicLossScaling,
    has_aux: bool = False,
    use_mixed_precision: bool = True,
    half_dtype: jnp.dtype = jnp.bfloat16,
) -> Callable:
    """
    Wraps a function to compute its value and gradient with support for mixed precision
    and dynamic loss scaling.

    :param func: The function for which the value and gradient are to be computed.
    :type func: callable
    :param scaling: An instance of DynamicLossScaling to handle loss scaling and gradient unscaling.
    :type scaling: loss_scaling.DynamicLossScaling
    :param has_aux: Indicates whether the function `func` returns auxiliary outputs along with the main value. Defaults to False.
    :type has_aux: bool
    :param use_mixed_precision: If True, the function will be cast to half precision. Defaults to True.
    :type use_mixed_precision: bool
    :param half_dtype: The half-precision dtype. Defaults to jnp.bfloat16.
    :type half_dtype: jnp.dtype
    :return: A wrapped function that computes the value, gradient, and additional
        information:
            - If `has_aux` is True:
                ((value, aux), loss_scaling_new, grads_finite, grad)
            - If `has_aux` is False:
                (value, loss_scaling_new, grads_finite, grad)
        Where:
            - `value`: The computed value of the function.
            - `aux`: Auxiliary outputs returned by the function (if `has_aux` is True).
            - `loss_scaling_new`: The updated loss scaling object.
            - `grads_finite`: A boolean indicating whether all gradients are finite.
            - `grad`: The computed gradients, unscaled.
    :rtype: callable
    """

    def wrapper(*args, **kwargs):
        dfunc_scaled = calculate_scaled_grad(
            func,
            scaling=scaling,
            has_aux=has_aux,
            use_mixed_precision=use_mixed_precision,
            half_dtype=half_dtype,
        )

        results, grad = dfunc_scaled(*args, **kwargs)

        if use_mixed_precision:
            inv_loss_scaling = (1 / scaling.loss_scaling).astype(jnp.float32)

            def unscale_and_cast(x):
                if eqx.is_array(x) and x.dtype in (
                    jnp.float16,
                    jnp.bfloat16,
                    jnp.float32,
                ):
                    return (x * inv_loss_scaling).astype(jnp.float32)
                return x

            grad = jax.tree.map(unscale_and_cast, grad)
            results = jax.tree.map(unscale_and_cast, results)
            grads_finite = loss_scaling.all_finite(grad)
            loss_scaling_new = scaling.adjust(grads_finite)
        else:
            grads_finite = jnp.bool_(True)
            loss_scaling_new = scaling

        if has_aux:
            value, aux = results
            return (value, aux), loss_scaling_new, grads_finite, grad
        else:
            value = results
            return value, loss_scaling_new, grads_finite, grad

    return wrapper
