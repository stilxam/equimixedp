import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array, PRNGKeyArray, Bool, PyTree
from typing import Callable
import optax


@eqx.filter_jit
def all_finite(tree: PyTree) -> Bool[Array, ""]:
    """
    Checks if all leaves in the pytree are finite.

    :param tree: The pytree to check.
    :type tree: PyTree
    :return: True if all leaves are finite, False otherwise.
    :rtype: Array
    """
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.array(True)
    else:
        leaves = map(jnp.isfinite, leaves)
        leaves = map(jnp.all, leaves)
        return jnp.stack(list(leaves)).all()


def scaled(
    func: Callable, scaling: "DynamicLossScaling", has_aux: bool = False
) -> Callable:
    """
    Scales the output of a function using dynamic loss scaling.

    This decorator wraps a given function such that its output is scaled using the
    provided dynamic loss scaling object. If the wrapped function returns auxiliary
    data (indicated by has_aux=True), only the primary value is scaled; otherwise, the
    sole returned value is scaled.

    :param func: The original function whose output is to be scaled.
    :type func: callable
    :param scaling: An object providing a `scale` method for scaling
        the function's output.
    :type scaling: DynamicLossScaling
    :param has_aux: Flag indicating whether the wrapped function returns
        a tuple (value, aux) where only the `value` should be scaled. Defaults to False.
    :type has_aux: bool
    :return: A new function that wraps the original function's behavior by applying
        the dynamic loss scaling to its result.
    :rtype: callable
    """

    def wrapper(*_args, **_kwargs):
        if has_aux:
            value, aux = func(*_args, **_kwargs)
            value = scaling.scale(value)
            return value, aux
        else:
            value = func(*_args, **_kwargs)
            value = scaling.scale(value)
            return value

    return wrapper


class DynamicLossScaling(eqx.Module):
    """
    Implements dynamic loss scaling for mixed precision training in JAX.

    The basic structure is taken from jmp.
    This class automatically adjusts the loss scaling factor during training to prevent
    numerical underflow/overflow when using reduced precision (e.g., float16). The scaling
    factor is increased periodically if gradients are finite, and decreased if non-finite
    gradients are detected, within specified bounds.

    :param loss_scaling: Current loss scaling factor.
    :type loss_scaling: jnp.ndarray
    :param min_loss_scaling: Minimum allowed loss scaling factor.
    :type min_loss_scaling: jnp.ndarray
    :param counter: Counter for tracking update periods.
    :type counter: jnp.ndarray
    :param factor: Multiplicative factor for adjusting loss scaling.
    :type factor: int
    :param period: Number of steps between potential increases of loss scaling.
    :type period: int
    """

    loss_scaling: Float[Array, ""]
    min_loss_scaling: Float[Array, ""]
    counter: Int[Array, ""]
    factor: int
    period: int

    def __init__(
        self,
        loss_scaling: Float[Array, ""],
        min_loss_scaling: Float[Array, ""],
        factor: int = 2,
        period: int = 2000,
        counter: Int[Array, ""] = None,
    ):
        """
        Initializes the DynamicLossScaling instance.

        :param loss_scaling: Initial loss scaling factor.
        :type loss_scaling: jnp.ndarray
        :param min_loss_scaling: Minimum loss scaling factor.
        :type min_loss_scaling: jnp.ndarray
        :param factor: Multiplicative factor for adjusting loss scaling. Defaults to 2.
        :type factor: int
        :param period: Number of steps between potential increases of loss scaling. Defaults to 2000.
        :type period: int
        :param counter: Initial counter value. Defaults to None.
        :type counter: jnp.ndarray, optional
        """
        assert loss_scaling.ndim == 0, "Expected scalar loss scaling"
        assert min_loss_scaling.ndim == 0, "Expected scalar minimum loss scaling"
        self.loss_scaling = loss_scaling
        self.min_loss_scaling = min_loss_scaling
        self.factor = factor
        self.period = period
        if counter is None:
            self.counter = jnp.zeros((), dtype=jnp.int32)
        else:
            self.counter = counter

    @eqx.filter_jit
    def scale(self, tree: PyTree) -> PyTree:
        """Scales each element in the input tree by the loss scaling factor.
        This method applies a multiplication operation to every leaf in the given pytree,
        using the loss scaling factor (converted to jnp.float16) stored in the instance.
        It returns a new pytree where each element has been scaled accordingly.

        :param tree: A pytree (e.g., nested lists, tuples, dicts) containing numerical values that represent the data to be scaled.
        :type tree: PyTree
        :return: A new pytree with each value multiplied by the loss scaling factor as a jnp.float16.
        :rtype: PyTree
        """
        return jax.tree.map(lambda x: x * self.loss_scaling.astype(jnp.float16), tree)

    @eqx.filter_jit
    def unscale(self, tree):
        """
        Unscales a pytree by multiplying each leaf element by the inverse of the loss scaling factor (in float32).

        :param tree: A pytree (nested structure of arrays, lists, tuples, dicts, etc.) where each leaf is a numeric array. These numerical values will be scaled by the computed inverse loss scaling factor.
        :type tree: PyTree
        :return: A new pytree with the same structure as the input, where each numeric leaf is multiplied by 1 / loss_scaling (as a float32).
        :rtype: PyTree
        """

        inv_loss_scaling = 1 / self.loss_scaling
        inv_loss_scaling = inv_loss_scaling.astype(
            jnp.float32
        )  # cast to float32, so the result is float32 (otherwise the whole scaling point would be senseless)
        return jax.tree.map(lambda x: x * inv_loss_scaling, tree)

    @eqx.filter_jit
    def adjust(self, grads_finite: Bool[Array, ""]) -> "DynamicLossScaling":
        """
        Adjust the loss scaling based on the finiteness of gradients and update the internal counter.
        It follows https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html and is directly adopted form JMP https://github.com/google-deepmind/jmp .

        :param grads_finite: A boolean scalar (0-dimensional) indicating whether all gradients are finite.
            Must satisfy grads_finite.ndim == 0.
        :type grads_finite: jnp.ndarray
        :return: A new instance of DynamicLossScaling. Use this and replace the current instance with it.
        :rtype: DynamicLossScaling
        """

        assert grads_finite.ndim == 0, "Expected boolean scalar"

        loss_scaling = jnp.where(
            grads_finite,
            jnp.where(
                self.counter == (self.period - 1),
                jnp.where(
                    jnp.isfinite(self.loss_scaling * self.factor),
                    self.loss_scaling * self.factor,
                    self.loss_scaling,
                ),
                self.loss_scaling,
            ),
            jnp.maximum(self.min_loss_scaling, self.loss_scaling / self.factor),
        )

        loss_scaling = jnp.clip(
            loss_scaling, min=self.min_loss_scaling, max=(2 - 2 ** (-10)) * 2**15
        )

        counter = ((self.counter + 1) % self.period) * grads_finite

        return DynamicLossScaling(
            loss_scaling=loss_scaling,
            counter=counter,
            period=self.period,
            factor=self.factor,
            min_loss_scaling=self.min_loss_scaling,
        )
