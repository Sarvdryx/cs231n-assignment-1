# 03 · Gradient of the Softmax loss (for `softmax_loss_naive`)

Sources: CS231n notes *Linear classification* (Softmax section) and *Optimization*. Follows section 7 of `02-linear-classification.md`.

Goal of this file: derive $\frac{\partial L}{\partial W}$ for the **full** loss (data loss + L2 regularisation), step by step, and map every symbol to a variable in `cs231n/classifiers/softmax.py`.

---

## 1. Setup in the assignment's layout

The code uses **row vectors**, so the shapes are transposed compared with the lecture notes ($Wx$).

| Symbol | Meaning | Shape | In the code |
|---|---|---|---|
| $N$ | number of examples in the minibatch | | `num_train` |
| $C$ | number of classes | | `num_classes` |
| $D$ | input dimension (3073 with the bias trick) | | `X.shape[1]` |
| $x_i$ | example $i$ (a row) | $(D,)$ | `X[i]` |
| $y_i$ | correct label of example $i$ | integer | `y[i]` |
| $W$ | weights; **column** $j$ is the template of class $j$ | $(D, C)$ | `W` |
| $W_{:,j}$ | column $j$ of $W$ | $(D,)$ | `W[:, j]` |
| $s = x_i W$ | scores of example $i$ | $(C,)$ | `scores` |
| $p_k = \frac{e^{s_k}}{\sum_j e^{s_j}}$ | softmax probability of class $k$ | $(C,)$ | `p` |
| $\lambda$ | regularisation strength | | `reg` |

Note: $s_k = x_i \cdot W_{:,k}$. Score $k$ depends **only on column $k$** of $W$. This one fact is what makes the gradient simple.

---

## 2. The full loss

$$
L(W) = \underbrace{\frac{1}{N}\sum_{i=1}^{N} L_i}_{\text{data loss}} \;+\; \underbrace{\lambda \sum_{d,c} W_{d,c}^2}_{\text{regularisation}},
\qquad
L_i = -\log p_{y_i} = -s_{y_i} + \log\sum_{j} e^{s_j}
$$

The derivative of a sum is the sum of derivatives, so

$$
\frac{\partial L}{\partial W} = \frac{1}{N}\sum_{i=1}^{N} \frac{\partial L_i}{\partial W} \;+\; \frac{\partial}{\partial W}\Big(\lambda \sum W^2\Big)
$$

We handle the two parts separately (sections 3–5 for $L_i$, section 6 for the regularisation).

### 2.1 Where $L_i = -s_{y_i} + \log\sum_j e^{s_j}$ comes from

Start from cross-entropy between the true distribution $t$ and the predicted distribution $p$ over the $C$ classes:

$$
H(t, p) = -\sum_{k=1}^{C} t_k \log p_k
$$

**Step A: the target is one-hot.** For example $i$, $t_k = 1$ if $k = y_i$ and $t_k = 0$ otherwise. Every term with $t_k = 0$ vanishes, only $k = y_i$ survives:

$$
L_i = H(t, p) = -1\cdot\log p_{y_i} = -\log p_{y_i}
$$

**Step B: substitute softmax.** $p_{y_i} = \dfrac{e^{s_{y_i}}}{\sum_j e^{s_j}}$, so

$$
L_i = -\log \frac{e^{s_{y_i}}}{\sum_j e^{s_j}}
$$

**Step C: log of a fraction.** $\log\frac{a}{b} = \log a - \log b$:

$$
L_i = -\Big(\log e^{s_{y_i}} - \log\sum_j e^{s_j}\Big) = -\log e^{s_{y_i}} + \log\sum_j e^{s_j}
$$

**Step D: $\log e^{z} = z$** (natural log):

$$
\boxed{L_i = -s_{y_i} + \log\sum_j e^{s_j}}
$$

Why this form is useful:
- For the **gradient** (section 3): the two terms are easy to differentiate separately. $-s_{y_i}$ gives $-\mathbb{1}[k=y_i]$, and $\log\sum_j e^{s_j}$ (called **log-sum-exp**) gives $p_k$.
- For **intuition**: the loss is "the log-sum-exp of all scores (a soft version of the max score) minus the correct score". If the correct score dominates, $\log\sum_j e^{s_j} \approx s_{y_i}$ and the loss is near 0.
- For **numerical stability**: subtracting $m = \max_j s_j$ from every score leaves $L_i$ unchanged, because $-(s_{y_i}-m) + \log\sum_j e^{s_j - m} = -s_{y_i} + m + \log\big(e^{-m}\sum_j e^{s_j}\big) = -s_{y_i} + \log\sum_j e^{s_j}$. This is why the code can do `scores -= np.max(scores)`.

---

## 3. Step 1: derivative of $L_i$ with respect to the scores

Write $L_i = -s_{y_i} + \log\sum_j e^{s_j}$ and differentiate with respect to one score $s_k$:

- First term: $\frac{\partial (-s_{y_i})}{\partial s_k} = -1$ if $k = y_i$, otherwise $0$. Write this as $-\mathbb{1}[k = y_i]$.
- Second term (chain rule on $\log$): $\frac{\partial}{\partial s_k}\log\sum_j e^{s_j} = \frac{e^{s_k}}{\sum_j e^{s_j}} = p_k$.

So

$$
\boxed{\frac{\partial L_i}{\partial s_k} = p_k - \mathbb{1}[k = y_i]}
$$

Read it as: **predicted probability minus target** (target is one-hot).
- Correct class $k = y_i$: $p_{y_i} - 1 \le 0$. Negative, so gradient descent will **increase** this score.
- Wrong class $k \ne y_i$: $p_k \ge 0$. Positive, so gradient descent will **decrease** this score, more strongly the more probability it wrongly took.

---

## 4. Step 2: from scores to the weights (chain rule)

$s_k = x_i \cdot W_{:,k} = \sum_d x_{i,d} W_{d,k}$. Therefore

$$
\frac{\partial s_k}{\partial W_{:,k}} = x_i, \qquad \frac{\partial s_k}{\partial W_{:,j}} = 0 \text{ for } j \ne k
$$

Chain rule: column $k$ of $W$ only affects $L_i$ through $s_k$, so

$$
\boxed{\frac{\partial L_i}{\partial W_{:,k}} = \frac{\partial L_i}{\partial s_k}\cdot \frac{\partial s_k}{\partial W_{:,k}} = \big(p_k - \mathbb{1}[k = y_i]\big)\, x_i}
$$

Shape check: a scalar times $x_i$ of shape $(D,)$ gives $(D,)$, the same shape as `W[:, k]`. ✔

> **Careful, a common misreading.** The zero $\frac{\partial s_k}{\partial W_{:,j}} = 0$ ($j \ne k$) is about **score $k$ vs column $j$**, not about the correct label. Here $k$ is **any** class, not $y_i$. Every column $j$ still reaches $L_i$ through its own score $s_j$, because $L_i$ depends on **all** scores via $\log\sum_j e^{s_j}$. So every column gets a non-zero gradient: $p_k - 1$ for the correct class, $p_k$ for the others. The zero only says there are no "cross wires" (column 0 does not change score 1).

So for **one** example, the gradient matrix $\frac{\partial L_i}{\partial W}$ has, in every column $k$, a copy of $x_i$ scaled by $(p_k - \mathbb{1}[k=y_i])$.

---

## 5. Step 3: average over the minibatch

$$
\frac{\partial}{\partial W}\Big(\frac{1}{N}\sum_i L_i\Big) = \frac{1}{N}\sum_i \frac{\partial L_i}{\partial W}
$$

Accumulate the per-example gradients (sum), then divide by $N$ once at the end, exactly like the loss.

---

## 6. Step 4: the regularisation term

$R(W) = \sum_{d,c} W_{d,c}^2$. Each entry appears in exactly one squared term, so

$$
\frac{\partial}{\partial W_{d,c}}\Big(\lambda \sum W^2\Big) = 2\lambda W_{d,c}
\quad\Longrightarrow\quad
\frac{\partial (\lambda R)}{\partial W} = 2\lambda W
$$

The factor 2 must match the loss: the code uses `reg * np.sum(W * W)` (no $\frac12$), so the gradient is `2 * reg * W`. If the loss used $\frac12\lambda\sum W^2$, the gradient would be $\lambda W$. A mismatch here is the most common reason the gradient check passes with `reg=0` but fails with `reg>0`.

---

## 7. Final result

$$
\boxed{\frac{\partial L}{\partial W_{:,k}} = \frac{1}{N}\sum_{i=1}^{N}\big(p_{i,k} - \mathbb{1}[k = y_i]\big)\, x_i \;+\; 2\lambda W_{:,k}}
$$

---

## 8. From formula to loop (the structure, not the code)

| Formula piece | Where it goes in `softmax_loss_naive` |
|---|---|
| $\sum_i$ | the existing loop `for i in range(num_train)` |
| one column $k$ at a time | a new inner loop over classes |
| $p_{i,k}$ | `p[k]`, already computed inside the outer loop |
| $\mathbb{1}[k = y_i]$ | a comparison between `k` and `y[i]` |
| $x_i$ | `X[i]` |
| add into column $k$ | `dW[:, k] += ...` |
| $\frac{1}{N}$ | once, after the outer loop |
| $2\lambda W$ | once, after dividing by $N$ |

Because $p$ is already computed for the loss, the gradient is computed **in the same loop**, which is what the TODO in the file suggests.

---

## 9. Worked example (check by hand)

$D = 2$, $C = 3$, one example $x = [1, 2]$, label $y = 2$ (third class), $\lambda = 0$.

Suppose the scores give $p = [0.2, 0.3, 0.5]$. Then $p - \text{onehot}(y) = [0.2, 0.3, -0.5]$ and

$$
\frac{\partial L}{\partial W} =
\begin{bmatrix}
0.2\cdot 1 & 0.3\cdot 1 & -0.5\cdot 1\\
0.2\cdot 2 & 0.3\cdot 2 & -0.5\cdot 2
\end{bmatrix}
=
\begin{bmatrix}
0.2 & 0.3 & -0.5\\
0.4 & 0.6 & -1.0
\end{bmatrix}
$$

Gradient descent $W \leftarrow W - \eta\,\frac{\partial L}{\partial W}$ moves column 2 **towards** $x$ (correct-class template becomes more like this image) and columns 0, 1 **away** from $x$. Each **row** of the gradient sums to 0, because $\sum_k (p_k - \mathbb{1}[k=y]) = 1 - 1 = 0$.

---

## 10. Sanity checks for the notebook

- **Initial loss ≈ 2.3**: $W$ is tiny, so all scores ≈ 0, so $p_k \approx 1/10$ for every class, so $L_i \approx -\log 0.1 = \log 10 \approx 2.303$. Regularisation is negligible because $W$ is tiny.
- **Gradient check**: compare the analytic `dW` with a numerical estimate $\frac{L(W + h) - L(W - h)}{2h}$ at random entries. Relative error around $10^{-7}$ or smaller means correct. Run it once with `reg = 0` (tests sections 3–5) and once with `reg > 0` (also tests section 6).

---

## Check yourself

- **Q1** Why does column $k$ of the gradient only involve $p_k$ and not the other probabilities, even though $p_k$ depends on all scores?
  (Hint: separate "$L_i$ depends on all scores" from "$s_k$ depends only on column $k$". The dependence on other scores is already inside the value of $p_k$.)
- **Q2** In the worked example, why does every row of $\frac{\partial L}{\partial W}$ sum to 0? Does this still hold when $\lambda > 0$?
- **Q3** If the model is already very confident and correct ($p_{y_i} \approx 1$), how big is the gradient from that example? What does this mean for learning?
- **Q4** Gradient check passes with `reg=0` but fails with `reg=5e1`. Where would you look first?
- **Q0** Without looking, derive $L_i = -s_{y_i} + \log\sum_j e^{s_j}$ from cross-entropy. Which step uses the fact that the target is one-hot?
- **Q5** Why divide `dW` by `num_train`, and why must the regularisation gradient **not** be divided by it?

---

## Corrections

<!-- Format: **Qx**: what I thought → what is actually correct, and why. -->
