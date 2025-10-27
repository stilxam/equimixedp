# Equimixedp: Mixed-Precision Training for Equinox

`equimixedp` is a small library that provides utilities for mixed-precision training with [Equinox](https://github.com/patrick-kidger/equinox).

## Credits

This library is a rework of the implementation from [Data-Science-in-Mechanical-Engineering/mixed_precision_for_JAX](https://github.com/Data-Science-in-Mechanical-Engineering/mixed_precision_for_JAX), which itself is an adaptation of [google-deepmind/jmp](https://github.com/google-deepmind/jmp).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Usage

Here is a simple example of how to use `equimixedp`:

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from equimixedp import MixedPrecision

# Define a simple model
class MyModel(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, key):
        self.linear = eqx.nn.Linear(10, 10, key=key)

    def __call__(self, x):
        return self.linear(x)

# Initialize the model and optimizer
key = jax.random.PRNGKey(0)
model = MyModel(key)
optimizer = optax.adam(1e-3)
optimizer_state = optimizer.init(model)

# Initialize MixedPrecision
mixed_precision = MixedPrecision()

# Define the loss function
@eqx.filter_value_and_grad
def loss_fn(model, x, y):
    y_pred = model(x)
    return jnp.mean((y - y_pred) ** 2)

# Create a gradient function with mixed precision
grad_fn = mixed_precision.gradient_fn(loss_fn, has_aux=False)

# Generate some dummy data
x = jnp.ones((128, 10))
y = jnp.ones((128, 10))

# Training step
loss_scaler, grads_finite, grads = grad_fn(model, x, y)
model, optimizer_state = mixed_precision.update_model(
    model, optimizer, optimizer_state, grads, grads_finite
)
```
