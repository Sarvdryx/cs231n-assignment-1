# 04 · Softmax inline questions and hyperparameter tuning

Sources: CS231n notes *Linear classification* and *Optimization* (gradient checking), `softmax.ipynb`. Follows `02-linear-classification.md` and `03-softmax-gradient.md`.

Goal of this file: the full reasoning behind the four inline questions of the Softmax notebook, filling the gaps found in my first answers, plus what I learned from tuning.

---

## 1. Inline Question 1: why is the initial loss close to $-\log(0.1)$?

The argument is a chain of four steps. My first answer skipped step 3.

1. `W` is initialised as `0.0001 * np.random.randn(D, C)`, so every weight is tiny.
2. Scores $s_k = x_i \cdot W_{:,k}$ are therefore all close to 0, and close to **each other**.
3. Softmax of $C$ almost equal scores is almost **uniform**: $p_k = \frac{e^{s_k}}{\sum_j e^{s_j}} \approx \frac{e^0}{C \cdot e^0} = \frac{1}{C}$. With $C = 10$ classes, $p_{y_i} \approx 0.1$.
4. The loss of one example is $L_i = -\log p_{y_i} \approx -\log 0.1 = \log 10 \approx 2.303$. Averaging over the batch keeps it at 2.303, and the regularisation term $\lambda \sum W^2$ is negligible because $W$ is tiny.

**Why this is a useful sanity check**: it tests the whole forward pass (scores, softmax, log, averaging) before any training. Typical bugs it catches:

| Loss you see | Likely bug |
|---|---|
| ≈ 2.3 | Correct |
| ≈ 0 or negative | Forgot the minus sign, or used $\log p$ of the wrong class |
| Much larger than 2.3 | `W` initialised too large, or regularisation added with a huge `reg` |
| `nan` / `inf` | Exponentials overflow: forgot to subtract the max score |

General rule: for $C$ classes, a correct softmax with small random weights starts at $\log C$.

---

## 2. Inline Question 2: why can the SVM gradient check fail?

### 2.1 How gradient checking works

The **numerical gradient** approximates the derivative with a centred difference:

$$\frac{df}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}$$

It assumes $f$ is smooth (differentiable) on the whole interval $[x - h, x + h]$. The **analytic gradient** is the formula we derived and coded. Gradient check compares the two with the relative error

$$\text{rel\_error} = \frac{|g_{\text{analytic}} - g_{\text{numeric}}|}{\max(|g_{\text{analytic}}|, |g_{\text{numeric}}|)}$$

### 2.2 Kinks

The SVM loss is built from $\max(0, s_j - s_{y_i} + \Delta)$. The hinge $\max(0, z)$ is not differentiable at $z = 0$: the derivative is 0 on the left and 1 on the right. Such a point is called a **kink**. At a kink we use a **subgradient** (any value between the left and right derivatives; in practice the code picks 0 or 1). This is standard and correct.

### 2.3 What actually goes wrong

If a margin $z = s_j - s_{y_i} + \Delta$ lies within $h$ of 0, then $z - h$ and $z + h$ fall on **different sides** of the kink. The centred difference then averages two different slopes and returns a value that matches neither.

So **the numerical gradient is the one that is wrong**, not the analytic one. My first answer said this was worrying because "the formula might be wrong and the update could not happen". That is the wrong way round:

- The analytic gradient is a valid subgradient, so gradient descent still works.
- It only happens when a margin falls inside a window of width $2h$ around the kink, which is rare with $h \approx 10^{-5}$.
- A mismatch in **one or two** dimensions out of many is therefore not a reason for concern. A mismatch in **most** dimensions would be: that points to a real bug.

### 2.4 One-dimensional example

$f(x) = \max(0, x)$, true derivative at $x > 0$ is 1. Take $x = 10^{-5}$ and $h = 10^{-4}$:

- $f(x + h) = f(1.1 \times 10^{-4}) = 1.1 \times 10^{-4}$
- $f(x - h) = f(-0.9 \times 10^{-4}) = 0$
- numerical gradient $= \frac{1.1 \times 10^{-4} - 0}{2 \times 10^{-4}} = 0.55$

The analytic gradient is 1, the numerical one is 0.55: relative error 0.45, a clear "failure" even though the code is correct.

### 2.5 Effect of the margin $\Delta$

At initialisation all scores are close to 0, so $s_j - s_{y_i} \approx 0$ and every margin is $z \approx \Delta$.

- $\Delta = 0$: every margin sits **right on the kink**, so failures become frequent.
- $\Delta = 1$ (the usual value): margins start around 1, far from the kink at 0, so failures are rare.

Larger $\Delta$ moves the kink away from where the margins are concentrated, so the check fails less often. (It does not remove the problem entirely: during training some margins will still pass through 0.)

### 2.6 Why the Softmax check never has this problem

Softmax loss uses $\exp$ and $\log$ only, so it is smooth everywhere. There are no kinks, and the centred difference is accurate as long as $h$ is small.

---

## 3. Inline Question 3: what do the learned weights look like?

### 3.1 What to expect

Each column $W_{:,k}$ (3072 values after dropping the bias) reshaped to $32 \times 32 \times 3$ is the **template** of class $k$ (see `02`, section 2). After tuning (val ≈ 0.35), the templates should look like blurry, averaged images with clear colour patterns, for example:

- **ship, plane**: mostly blue (sea, sky), sometimes a darker blob in the centre.
- **frog, deer**: green and brown (grass, forest).
- **car**: often a reddish blob, because many training cars are red.
- **horse**: a brown shape that can look like it has two heads, because horses facing left and right are blended into one template.

### 3.2 Why

The score is a dot product $s_k = x \cdot W_{:,k}$, which is largest when the image "lines up" with the template. Training pushes each template towards what images of that class have in common, so it becomes roughly an average of the class, mostly colour and rough layout. A single linear template cannot store "red car **or** blue car" or "horse facing left **or** right" separately; it can only blend them. That limitation is what neural networks remove with hidden units.

### 3.3 My result

After tuning (val 0.365) the templates match the expectation: plane and ship are blue (ship with dark blue water at the bottom), bird, deer and frog are green with a darker or brownish blob in the centre, car is a red/pink shape with a bluish windshield, truck is reddish with dark wheels, horse is a reddish-brown shape that spreads to both sides (blended left- and right-facing horses), cat and dog are reddish-brown blobs that are hard to tell apart.

### 3.4 If the weights look like random noise

My first observation was "chaotic pixels, no dominant colour". That should not happen for a model with val ≈ 0.35. Check in this order:

1. The visualize cell must run **after** the tuning cell, so it plots `best_softmax`, not an untrained or barely trained model.
2. `best_softmax` must actually be updated inside the tuning loop (it was missing in my first version of the code).
3. The bias row must be dropped before reshaping: `W[:-1, :]`, then reshape to $(32, 32, 3, 10)$.
4. Each template is rescaled to $[0, 255]$ for display; without rescaling, values near 0 look like noise.

---

## 4. Inline Question 4: can a new point change the Softmax loss but not the SVM loss?

**Answer: True.**

Take a new example that the model already classifies correctly with a large gap: $s_{y_i} - s_j \ge \Delta$ for every $j \ne y_i$.

- **SVM**: every term $\max(0, s_j - s_{y_i} + \Delta)$ is 0, so $L_i = 0$ exactly. The total SVM loss (a sum) does not change.
- **Softmax**: $p_{y_i} = \frac{e^{s_{y_i}}}{\sum_j e^{s_j}} < 1$ for any finite scores, so $L_i = -\log p_{y_i} > 0$. It can be very small, but never exactly 0, so the total Softmax loss increases.

The regularisation term does not depend on the data, so adding a point does not change it in either loss.

This is the same idea as section 8 of `02`: the SVM is **satisfied** once the margins are met; Softmax is **never fully satisfied**.

---

## 5. Hyperparameter tuning: what I learned

### 5.1 Rules

- Train only on `X_train`, choose only by `val_acc`, touch `X_test` exactly once at the end.
- Search on a **log scale** first (values like $10^{-7}, 10^{-6}$), because good learning rates and regularisation strengths can differ by orders of magnitude.
- Start coarse, then refine around the best value. If the best value is on the **edge** of the grid, the grid is in the wrong place: extend it in that direction.

### 5.2 Reading the table

| Pattern | Meaning | Action |
|---|---|---|
| train and val both low, close to each other | Underfitting | Larger lr, more iterations, or smaller reg |
| train clearly higher than val | Overfitting | Larger reg |
| accuracy drops sharply or ~0.10, overflow warnings | lr too large | Remove that region |

### 5.3 My runs

| Round | Grid | Best (lr, reg) | Best val |
|---|---|---|---|
| 1 | lr {1e-7, 5e-7, 1e-6} × reg {1e3, 1e4, 5e4} | (1e-6, 1e4) | 0.328 |
| 2 | lr {1e-6, 2e-6, 5e-6} × reg {5e3, 1e4, 2e4} | (2e-6, 1e4) | 0.346 |
| 3 | lr {1.5e-6, 2e-6, 3e-6} × reg {5e3, 1e4, 1.5e4}, 3000 iters | (1.5e-6, 1e4) | 0.365 |

Observations:

- Round 1: train ≈ val everywhere and accuracy grew with lr, with the best value at the edge of the grid → underfitting, moved the grid towards larger lr.
- Round 2: lr = 5e-6 dropped to 0.22–0.32 → past the optimum. The best value is now inside the grid. Still train ≈ val, so more iterations should help more than more regularisation.
- Round 3: doubling the iterations to 3000 lifted every cell, and the best result reached 0.365 (> 0.36), so tuning stopped here. Within this grid the smaller lr (1.5e-6) and smaller reg (5e3–1e4) were best: with more iterations the model no longer needs a large step size. (1.5e-6, 5e3) gave 0.364, within noise of the winner. Train ≈ val still holds (0.347 vs 0.365), so there is no overfitting; a linear classifier is simply limited to about this accuracy on CIFAR-10.
- SGD uses random minibatches, so the same (lr, reg) can differ by about 0.005–0.01 between runs. Differences smaller than that are noise.

Test accuracy (run once, with the best model from round 3): **0.354**. It is about 0.01 below the best val accuracy (0.365). A small drop is expected: the hyperparameters were chosen to maximise val accuracy among 27 runs, so the best val score is slightly optimistic (part of it is luck from SGD noise and from the particular 1,000 val images). A gap this small means the choice generalises; a large gap would suggest the validation set leaked into the choices or is too small.

---

## Check yourself

- **Q1** A softmax classifier on a 100-class dataset with tiny initial weights: what initial loss do you expect?
- **Q2** In gradient check for the SVM, one dimension out of 10 has relative error 0.3, the rest are around $10^{-9}$. Bug or kink? What would make you change your mind?
- **Q3** Why does the ReLU in a neural network raise the same gradient-check issue as the SVM hinge?
- **Q4** Why would the "ship" and "plane" templates be easy to confuse, and what kind of test image would fool them?
- **Q5** Can a new point change the SVM loss but leave the Softmax loss unchanged? Why or why not?
- **Q6** In round 1 the best value was on the edge of the grid. Why is that a signal to move the grid rather than to refine around that point?

---

## Corrections

<!-- Format: **Qx**: what I thought → what is actually correct, and why. -->

- **Inline Q1** Partly correct: small `W` → scores ≈ 0 → loss ≈ $-\log 0.1$. Missing step: almost equal scores make softmax **uniform**, so every class gets probability $1/C = 0.1$, and the loss is $-\log$ of the correct class's probability.
- **Inline Q2** Cause correct (kink of $\max(0, \cdot)$, left and right derivatives differ). "It is a concern because the update cannot happen" → wrong: the analytic (sub)gradient is fine and training works; the **numerical** gradient is what breaks when $x \pm h$ straddle the kink. Rare, so not a concern. Missing: the 1D example (section 2.4) and the effect of $\Delta$ (section 2.5).
- **Inline Q3** "Chaotic pixels, no dominant colour" → a tuned linear classifier shows blurry colour templates. The noise came from plotting before tuning had produced a good `best_softmax`; after tuning the templates appeared as expected (sections 3.3–3.4).
- **Inline Q4** Not answered → True, see section 4.
