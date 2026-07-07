"""Transformer-based classifier on MNIST using JAX.

Multi-head self-attention blocks with feedforward layers, trained with
cross-entropy loss on flattened MNIST images.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial

NUM_CLASSES = 10
D_MODEL = 112
NUM_HEADS = 4
LEARNING_RATE = 1e-3
BATCH_SIZE = 256  # FIX #3: was undefined. Larger batches keep the TPU busy.


def init_linear_layer(key, in_dim, out_dim):
  W = jax.random.normal(key, (in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
  b = jnp.zeros(out_dim)
  return {'W': W, 'b': b}


def init_transformer_ffn_block(keys, in_dim):
  up_layer = init_linear_layer(keys[0], in_dim, in_dim * 4)
  down_layer = init_linear_layer(keys[1], in_dim * 4, in_dim)
  return {'up_layer': up_layer, 'down_layer': down_layer}


def init_multihead_self_attention_block(keys, in_dim):
  key_layer = init_linear_layer(keys[0], in_dim, in_dim)
  query_layer = init_linear_layer(keys[1], in_dim, in_dim)
  value_layer = init_linear_layer(keys[2], in_dim, in_dim)
  output_layer = init_linear_layer(keys[3], in_dim, in_dim)
  return {'key_layer': key_layer, 'query_layer': query_layer,
          'value_layer': value_layer, 'output_layer': output_layer}


def init_layer_norm(in_dim):
  return {'beta': jnp.zeros(in_dim), 'gamma': jnp.ones(in_dim)}


def init_transformer_blocks_with_output(key, d_model, num_layers):
  keys = jax.random.split(key, num=num_layers * 6 + 1)
  transformer = []
  for i in range(num_layers):
    base = i * 6
    mha_block = init_multihead_self_attention_block(keys[base:base + 4], d_model)
    ln_1 = init_layer_norm(d_model)
    ffn_block = init_transformer_ffn_block(keys[base + 4:base + 6], d_model)
    ln_2 = init_layer_norm(d_model)
    transformer.append({'mha_block': mha_block, 'ln_1': ln_1,
                        'ffn_block': ffn_block, 'ln_2': ln_2})
  transformer.append(init_linear_layer(keys[-1], d_model, NUM_CLASSES))
  return transformer


def linear_forward(params, X):
  return X @ params['W'] + params['b']


def mha_forward(params, num_heads, X):
  batch_size, seq_len, d_model = X.shape
  query = linear_forward(params['query_layer'], X)
  key = linear_forward(params['key_layer'], X)
  value = linear_forward(params['value_layer'], X)

  d_head = d_model // num_heads
  query_split = query.reshape([batch_size, seq_len, num_heads, d_head]).swapaxes(1, 2)
  key_split = key.reshape([batch_size, seq_len, num_heads, d_head]).transpose([0, 2, 3, 1])
  value_split = value.reshape([batch_size, seq_len, num_heads, d_head]).swapaxes(1, 2)

  # FIX #2: scale scores BEFORE softmax, by sqrt(d_head) (per-head dim),
  # so attention weights sum to 1 and scores don't saturate.
  attention_scores = (query_split @ key_split) / jnp.sqrt(d_head)
  attention_probs = jax.nn.softmax(attention_scores, axis=-1)

  att_value = attention_probs @ value_split

  return linear_forward(params['output_layer'],
                        att_value.swapaxes(1, 2).reshape([batch_size, seq_len, d_model]))


def ffn_forward(params, X):
  X = X @ params['up_layer']['W'] + params['up_layer']['b']
  X = jax.nn.gelu(X)
  X = X @ params['down_layer']['W'] + params['down_layer']['b']
  return X


def ln_forward(params, X):
  mu = jnp.mean(X, axis=-1, keepdims=True)
  sigma = jnp.var(X, axis=-1, keepdims=True)
  return (X - mu) * jax.lax.rsqrt(sigma + 1e-9) * params['gamma'] + params['beta']


# SPEED #5: jit re-enabled. num_heads drives reshapes, so it must be static.
@partial(jax.jit, static_argnums=1)
def forward(params, num_heads, X):
  batch_size, d = X.shape
  seq_len = d // D_MODEL
  X = jnp.reshape(X, (batch_size, seq_len, D_MODEL))
  for i in range(len(params) - 1):
    X_out = mha_forward(params[i]['mha_block'], num_heads, X)
    X_res = ln_forward(params[i]['ln_1'], X_out + X)
    X_out = ffn_forward(params[i]['ffn_block'], X_res)
    X = ln_forward(params[i]['ln_2'], X_out + X_res)
  X = X[:, 0, :]
  return linear_forward(params[-1], X)


def loss_fn(params, num_heads, X, y):
  output = forward(params, num_heads, X)
  # FIX #1: paired fancy indexing — "for each row i, take column y[i]".
  # A lone [y] would gather whole ROWS by label value (silently wrong).
  per_example_loss = -jax.nn.log_softmax(output)[jnp.arange(y.shape[0]), y]
  return jnp.mean(per_example_loss)


# SPEED #8: donate params buffers — XLA reuses their memory for the update output.
@partial(jax.jit, donate_argnums=(0,))
def update(params, X, y):
  loss, grads = jax.value_and_grad(loss_fn)(params, NUM_HEADS, X, y)
  return jax.tree_util.tree_map(lambda v, g: v - LEARNING_RATE * g, params, grads), loss


@jax.jit
def accuracy(params, X, y):
  y_pred = forward(params, NUM_HEADS, X)
  return jnp.sum(jnp.argmax(y_pred, axis=-1) == y) / y.shape[0]


# ---------------------------------------------------------------------------
# Data. SPEED #7: MNIST fits in memory, so we pre-load to numpy arrays and
# slice batches directly — no per-batch DataLoader/collate overhead.
# COMPAT #4: .data / .targets (train_data / train_labels were removed).
# ---------------------------------------------------------------------------
from torchvision.datasets import MNIST

mnist_train = MNIST('/tmp/mnist/', download=True, train=True)
mnist_test = MNIST('/tmp/mnist/', download=True, train=False)

train_images = np.asarray(mnist_train.data.numpy().reshape(len(mnist_train), -1), dtype=np.float32)
train_labels = np.asarray(mnist_train.targets.numpy())
test_images = np.asarray(mnist_test.data.numpy().reshape(len(mnist_test), -1), dtype=np.float32)
test_labels = np.asarray(mnist_test.targets.numpy())

num_train = train_images.shape[0]
steps_per_epoch = num_train // BATCH_SIZE  # drop last partial batch:
# a different batch shape would trigger a fresh XLA compile of `update`.

key = jax.random.key(42)
params = init_transformer_blocks_with_output(key, d_model=D_MODEL, num_layers=3)

rng = np.random.default_rng(0)

for epoch in range(1000):
  start_time = time.time()
  perm = rng.permutation(num_train)
  epoch_loss = 0.0

  for step in range(steps_per_epoch):
    idx = perm[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
    params, loss = update(params, train_images[idx], train_labels[idx])
    # SPEED #6: don't print (i.e. sync) every step — log occasionally.
    if step % 100 == 0:
      print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {float(loss):.4f}")
    epoch_loss += loss  # stays on-device; only synced once per epoch below

  epoch_time = time.time() - start_time

  train_acc = accuracy(params, train_images, train_labels)
  test_acc = accuracy(params, test_images, test_labels)
  print("Epoch {} in {:0.2f} sec | mean loss {:.4f}".format(
      epoch, epoch_time, float(epoch_loss) / steps_per_epoch))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))
