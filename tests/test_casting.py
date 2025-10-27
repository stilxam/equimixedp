import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from equimixedp import casting

class TestCasting:
    def test_cast_tree(self):
        tree = {
            "a": jnp.array([1.0, 2.0], dtype=jnp.float32),
            "b": {
                "c": jnp.array([3.0, 4.0], dtype=jnp.float32),
                "d": "string",
            },
            "e": 1,
        }

        # Test casting to float16
        tree_f16 = casting.cast_tree(tree, jnp.float16)
        assert tree_f16["a"].dtype == jnp.float16
        assert tree_f16["b"]["c"].dtype == jnp.float16
        assert tree_f16["b"]["d"] == "string"
        assert tree_f16["e"] == 1

        # Test casting to bfloat16
        tree_bf16 = casting.cast_tree(tree, jnp.bfloat16)
        assert tree_bf16["a"].dtype == jnp.bfloat16
        assert tree_bf16["b"]["c"].dtype == jnp.bfloat16

        # Test casting to float32
        tree_f32 = casting.cast_tree(tree_f16, jnp.float32)
        assert tree_f32["a"].dtype == jnp.float32
        assert tree_f32["b"]["c"].dtype == jnp.float32

    def test_cast_function(self):
        @casting.cast_function(input_dtype=jnp.float16, output_dtype=jnp.float32)
        def identity_fn(x):
            return x

        x = jnp.array([1.0, 2.0], dtype=jnp.float32)
        y = identity_fn(x)

        # The input to the function should be cast to float16
        # but we can't check that directly.
        # We can check the output is float32
        assert y.dtype == jnp.float32

    def test_force_full_precision(self):
        @casting.force_full_precision()
        def identity_fn(x):
            return x

        x = jnp.array([1.0, 2.0], dtype=jnp.float16)
        y = identity_fn(x)
        assert y.dtype == jnp.float32
