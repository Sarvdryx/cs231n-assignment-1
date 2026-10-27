# NumPy refresher: shapes, broadcasting, `dot` and `@`

Short rules needed for the vectorized Softmax (and everything after it). Each rule has one tiny example.

Running example:

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])     # shape (2, 3): 2 rows, 3 columns
```

---

## 1. Shape is a tuple: why `(2,)` and not `(2)`

- `shape` is always a Python **tuple**. In Python, the **comma** makes a tuple, not the parentheses.
  - `(2)` is just the integer `2` in parentheses: `type((2))` → `int`
  - `(2,)` is a tuple with one element: `type((2,))` → `tuple`
- So a 1-element shape must be written with a trailing comma: `(2,)`.

## 2. `(2,)` vs `(2, 1)`

| Shape | `ndim` | What it is | Example |
|---|---|---|---|
| `(2,)` | 1 | a flat list of 2 numbers, no rows or columns | `[6, 15]` |
| `(2, 1)` | 2 | a matrix with 2 rows and 1 column (a column) | `[[6], [15]]` |

Key fact: in broadcasting, a 1-D array behaves like a **row** and lines up with the **last** axis. To make it act as a column, give it shape `(N, 1)`.

## 3. Broadcasting rules for `+ - * /` (element-wise operations)

Element-wise: each cell combines with the cell at the same position. Shapes do **not** need to be identical, only **compatible**.

**Rule:** compare shapes from **right to left**. For each pair of axes, it works if the lengths are **equal**, or **one of them is 1**, or **one side has no axis there**. An axis of length 1 (or a missing axis) is stretched (copied) to match.

| Operation | Right-to-left check | Result |
|---|---|---|
| `(2,3) / (2,3)` | 3–3 ✓, 2–2 ✓ | `(2,3)` |
| `(2,3) / (2,1)` | 3–1 ✓, 2–2 ✓ | `(2,3)`: each **row** divided by its own number |
| `(2,3) / (3,)` | 3–3 ✓, 2–none ✓ | `(2,3)`: each **column** divided by its own number |
| `(2,3) / 5` | scalar | `(2,3)`: everything divided by 5 |
| `(2,3) / (2,)` | 3–2 ✗ | **error** |

Examples:

```
A / [1, 2, 3]        # (3,) is a row, copied down to both rows
  = [[1/1, 2/2, 3/3],
     [4/1, 5/2, 6/3]]  = [[1, 1, 1], [4, 2.5, 2]]

A / [10, 100]        # (2,) is a row of length 2, but A has 3 columns
  → ValueError: operands could not be broadcast together with shapes (2,3) (2,)

A / [[10], [100]]    # (2, 1) is a column, copied across the 3 columns
  = [[1/10,  2/10,  3/10 ],
     [4/100, 5/100, 6/100]]
```

Trap: if the number of rows equals the number of columns (N = C), dividing by an `(N,)` vector does **not** error. It silently divides by column instead of by row. Wrong result, no warning.

## 4. `keepdims=True`

A reduction (`sum`, `max`, `mean`) along an axis removes that axis. `keepdims=True` keeps the **number of dimensions** the same; the reduced axis becomes length 1.

```python
A.sum(axis=1)                  # [6, 15]       shape (2,)    (a row)
A.sum(axis=1, keepdims=True)   # [[6], [15]]   shape (2, 1)  (a column)

A / A.sum(axis=1, keepdims=True)   # each row divided by its own sum ✓
```

This is the softmax normalisation for a whole batch: `p = e / e.sum(axis=1, keepdims=True)`. The same goes for the stability trick: `scores - scores.max(axis=1, keepdims=True)`.

Rule of thumb: when you reduce along rows and then combine the result with the original matrix, use `keepdims=True`.

## 5. Matrix product: `dot` and `@`

Not element-wise and no broadcasting of the inner axis. Multiply and **sum along one axis**.

**Rule:** the **last axis of the left** operand must equal the **first axis of the right** operand. That matching axis disappears; the remaining axes form the result.

| Operation | Match | Result | Meaning in Softmax |
|---|---|---|---|
| `(D,) @ (D,C)` | D–D ✓ | `(C,)` | scores of one image: `X[i] @ W` |
| `(N,D) @ (D,C)` | D–D ✓ | `(N,C)` | scores of the batch: `X @ W` |
| `(D,N) @ (N,C)` | N–N ✓ | `(D,C)` | gradient: `X.T @ dscores` |
| `(D,C) @ (D,)` | C–D ✗ | **error** (unless C = D) | |

Example:

```
[1, 2] @ [[1, 0, 2],      =  [1*1 + 2*3,  1*0 + 2*1,  1*2 + 2*0]  =  [7, 2, 2]
          [3, 1, 0]]
 (2,)      (2, 3)                                                   (3,)
```

Each output element is the dot product of the vector with one column: `scores[k] = x · W[:, k]`.

## 6. `dot` vs `@`

For 1-D and 2-D arrays (everything in Assignment 1), `a @ b`, `a.dot(b)` and `np.dot(a, b)` give **the same result**.

They differ in two cases:

| Case | `@` (matmul) | `np.dot` |
|---|---|---|
| Scalar | `2 @ v` → error | `np.dot(2, v)` → `2 * v` |
| 3-D and above, e.g. `(2,3,4)` with `(2,4,5)` | treats the first axis as a batch, multiplies pair by pair → `(2, 3, 5)` | multiplies every combination → `(2, 3, 2, 5)` |

Use `@` in new code: it reads like the math (`X @ W`), it is strict with scalars, and its batch behaviour matches PyTorch. Leave `.dot` in the CS231n starter code as it is.

## 7. Element-wise vs matrix product at a glance

| | `+ - * /` (element-wise) | `dot` / `@` |
|---|---|---|
| What happens | each cell with the cell at the same position | multiply, then sum along one axis |
| Which axes must match | all axes (equal, or 1) | only the inner axis |
| Does the matched axis survive | yes | no, it is summed away |

Habit: before each line of vectorized code, decide which rule it uses and write the shapes next to it.
