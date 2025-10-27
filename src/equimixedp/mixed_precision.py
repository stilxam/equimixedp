import jax.numpy as jnp
import equinox as eqx

from .dtypes import DtypeConfig
from .loss_scaling import DynamicLossScaling
from .gradients import filter_grad, optimizer_update

from jaxtyping import PyTree
from typing import Callable


class MixedPrecision(eqx.Module):
    """
    Main class for mixed precision training utilities.

    This class encapsulates dtype configuration and dynamic loss scaling,
    providing high-level methods for gradient computation and model updates.

    :param dtype_config: Configuration for data types.
    :type dtype_config: DtypeConfig
    :param loss_scaler: Dynamic loss scaling instance.
    :type loss_scaler: DynamicLossScaling
    """

    dtype_config: DtypeConfig
    loss_scaler: DynamicLossScaling

    def __init__(
        self,
        half_dtype: str | jnp.dtype = "bfloat16",
        initial_loss_scale: float = 2**15,
        min_loss_scale: float = 1.0,
        scale_factor: int = 2,
        scale_period: int = 2000,
    ):
        """
        Initializes the MixedPrecision instance.

        :param half_dtype: The half-precision dtype. Defaults to "bfloat16".
        :type half_dtype: str or jnp.dtype
        :param initial_loss_scale: Initial loss scaling factor. Defaults to 2**15.
        :type initial_loss_scale: float
        :param min_loss_scale: Minimum loss scaling factor. Defaults to 1.0.
        :type min_loss_scale: float
        :param scale_factor: Factor for adjusting loss scaling. Defaults to 2.
        :type scale_factor: int
        :param scale_period: Period for potential loss scaling increases. Defaults to 2000.
        :type scale_period: int
        """
        self.dtype_config = DtypeConfig(half_dtype)
        self.loss_scaler = DynamicLossScaling(
            loss_scaling=jnp.array(float(initial_loss_scale)),
            min_loss_scaling=jnp.array(float(min_loss_scale)),
            factor=scale_factor,
            period=scale_period,
        )

    def gradient_fn(self, func: Callable, has_aux: bool = False) -> Callable:
        """
        Returns a function that computes gradients with mixed precision and loss scaling.

        :param func: The loss function.
        :type func: Callable
        :param has_aux: Whether the function returns auxiliary data.
        :type has_aux: bool
        :return: A function that takes *args, **kwargs and returns
            (loss_scaler, grads_finite, grads) or (loss_scaler, grads_finite, grads, aux) if has_aux.
        :rtype: Callable
        """
        return filter_grad(
            func,
            self.loss_scaler,
            has_aux=has_aux,
            half_dtype=self.dtype_config.half_precision_dtype,
        )

    def update_model(
        self,
        model: PyTree,
        optimizer,
        optimizer_state: PyTree,
        grads: PyTree,
        grads_finite: Bool[Array, ""],
    ) -> tuple[PyTree, PyTree]:
        """
        Updates the model and optimizer state using the gradients, conditional on finiteness.

        :param model: The current model.
        :type model: PyTree
        :param optimizer: The optimizer (optax.GradientTransformation).
        :type optimizer: optax.GradientTransformation
        :param optimizer_state: The current optimizer state.
        :type optimizer_state: PyTree
        :param grads: The gradients.
        :type grads: PyTree
        :param grads_finite: Whether the gradients are finite.
        :type grads_finite: bool
        :return: (new_model, new_optimizer_state)
        :rtype: tuple
        """
        return optimizer_update(model, optimizer, optimizer_state, grads, grads_finite)

    def value_and_grad_fn(self, func: Callable, has_aux: bool = False) -> Callable:
        """
        Returns a function that computes the value and gradient with mixed precision and loss scaling.

        :param func: The loss function.
        :type func: Callable
        :param has_aux: Whether the function returns auxiliary data.
        :type has_aux: bool
        :return: A function that takes *args, **kwargs and returns
            (value, loss_scaling_new, grads_finite, grads) or ((value, aux), loss_scaling_new, grads_finite, grads) if has_aux.
        :rtype: Callable
        """
        from .gradients import filter_value_and_grad

        return filter_value_and_grad(
            func,
            self.loss_scaler,
            has_aux=has_aux,
            half_dtype=self.dtype_config.half_precision_dtype,
        )

    def set_half_dtype(self, dtype: str | jnp.dtype):
        """
        Sets the half-precision dtype.

        :param dtype: The new half-precision dtype.
        :type dtype: str or jnp.dtype
        """
        self.dtype_config.set_half_precision_dtype(dtype)
