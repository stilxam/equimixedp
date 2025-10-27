import jax.numpy as jnp
import equinox as eqx


class DtypeConfig(eqx.Module):
    """
    Configuration for data types used in mixed precision training.

    This class manages the half-precision data type, allowing users to set and retrieve
    the current half-precision dtype (e.g., float16 or bfloat16).

    :ivar half_precision_dtype: The current half-precision data type.
    """

    half_precision_dtype: jnp.dtype

    def __init__(self, half_precision_dtype: str | jnp.dtype = "bfloat16"):
        """
        Initializes the DtypeConfig with the specified half-precision dtype.

        :param half_precision_dtype: The half-precision dtype.
            Can be a string ("float16" or "bfloat16") or a JAX dtype.
            Defaults to "bfloat16".
        :type half_precision_dtype: str or jnp.dtype
        """
        if isinstance(half_precision_dtype, str):
            if half_precision_dtype == "float16":
                self.half_precision_dtype = jnp.float16
            elif half_precision_dtype == "bfloat16":
                self.half_precision_dtype = jnp.bfloat16
            else:
                raise ValueError(
                    f"Unsupported dtype: {half_precision_dtype}. Use 'float16' or 'bfloat16'."
                )
        elif half_precision_dtype in (jnp.float16, jnp.bfloat16):
            self.half_precision_dtype = half_precision_dtype
        else:
            raise TypeError("Dtype must be a string or in (jnp.float16, jnp.bfloat16).")

    def set_half_precision_dtype(self, dtype: str | jnp.dtype):
        """
        Sets the half-precision dtype.

        :param dtype: The new half-precision dtype.
        :type dtype: str or jnp.dtype
        """
        if isinstance(dtype, str):
            if dtype == "float16":
                self.half_precision_dtype = jnp.float16
            elif dtype == "bfloat16":
                self.half_precision_dtype = jnp.bfloat16
            else:
                raise ValueError(
                    f"Unsupported dtype: {dtype}. Use 'float16' or 'bfloat16'."
                )
        elif dtype in (jnp.float16, jnp.bfloat16):
            self.half_precision_dtype = dtype
        else:
            raise TypeError("Dtype must be a string or in (jnp.float16, jnp.bfloat16).")

    def get_half_precision_dtype(self) -> jnp.dtype:
        """
        Returns the current half-precision dtype.

        :return: The current half-precision dtype.
        :rtype: jnp.dtype
        """
        return self.half_precision_dtype
