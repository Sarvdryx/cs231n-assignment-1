# 02 · Linear classification: SVM and Softmax

Source: CS231n notes *Linear classification* (cs231n.github.io/linear-classify). Prepares for the Softmax part of Assignment 1.

How to use this file: each section ends with **Check yourself** questions. Answer them without looking back, then compare. Mistakes in understanding are recorded in the last section, **Corrections**.

---

## 0. Background knowledge

Short definitions of the ML concepts the notes rely on. For each: what it is, the idea behind it, where it is used, and a tiny example.

### 0.1 Parametric model
- **Definition**: a function $f(x; \theta)$ whose behaviour is fully determined by a fixed set of parameters $\theta$ learned from data.
- **Idea**: kNN keeps the whole training set and compares against it at test time. A parametric model *compresses* the training set into $\theta$; after training, the data can be thrown away.
- **Used in**: linear classifiers, neural networks, CNNs. Prediction cost depends on the size of $\theta$, not on the size of the training set.
- **Example**: $f(x) = Wx + b$ on CIFAR-10 has $10 \times 3072 + 10 = 30{,}730$ numbers. Predicting is one matrix-vector product instead of 50,000 distance computations.

### 0.2 Loss function and empirical risk minimisation
- **Definition**: a loss $L_i$ measures how unhappy we are with the prediction on example $i$. The **empirical risk** is the average loss over the training set, $\frac{1}{N}\sum_i L_i$.
- **Idea**: "learning" becomes "optimisation": find $\theta$ that minimises the average loss (plus regularisation).
- **Used in**: every supervised model. The choice of loss defines what "good" means (hinge for SVM, cross-entropy for Softmax, squared error for regression).
- **Example**: a model predicts 3.0, the target is 5.0. Squared error gives $(3-5)^2 = 4$; absolute error gives $2$.

### 0.3 Overfitting and generalisation
- **Definition**: **overfitting** means low training loss but high loss on unseen data. **Generalisation** is how well the model performs on data it has not seen.
- **Idea**: the goal is never the training set itself; it is a proxy for the real distribution. A model that memorises noise does not transfer.
- **Used in**: the reason we keep validation/test sets and use regularisation.
- **Example**: 1-NN has 0% training error but only about 27% test accuracy on CIFAR-10.

### 0.4 Regularisation (L1 and L2)
- **Definition**: an extra penalty $\lambda R(W)$ added to the loss that prefers "simpler" weights.
  - **L2** (ridge, weight decay): $R(W) = \sum_{k,l} W_{k,l}^2$
  - **L1** (lasso): $R(W) = \sum_{k,l} |W_{k,l}|$
- **Idea**: among the many weight settings that fit the training data, pick the one least likely to be fitting noise. $\lambda$ controls the trade-off between fitting the data and keeping weights small.
- **Behaviour**:
  - L2 penalises large weights heavily (quadratic), so it prefers **many small, spread-out weights**. The model uses all input dimensions a little rather than a few dimensions a lot.
  - L1 pushes many weights to **exactly zero** (sparse solutions), acting as feature selection.
- **Used in**: L2 is the default in deep learning (weight decay in SGD/AdamW). L1 is used when sparsity or interpretability is wanted.
- **Example**: input $x = [1, 1, 1, 1]$, two weight vectors with the same score $w \cdot x = 1$:
  - $w_1 = [1, 0, 0, 0]$: L2 penalty $= 1$, L1 penalty $= 1$
  - $w_2 = [0.25, 0.25, 0.25, 0.25]$: L2 penalty $= 0.25$, L1 penalty $= 1$

  L2 strictly prefers $w_2$ (spread out); L1 sees them as equal.

### 0.5 Maximum Likelihood Estimation (MLE)
- **Definition**: choose the parameters that make the observed data most probable: $\hat\theta = \arg\max_\theta \prod_i p(y_i \mid x_i; \theta)$.
- **Idea**: products of many small probabilities underflow and are awkward to differentiate, so take the log (monotonic, same argmax) and negate it to get a quantity to minimise: the **negative log-likelihood (NLL)** $-\sum_i \log p(y_i \mid x_i; \theta)$.
- **Used in**: almost every probabilistic model. Cross-entropy loss for classification *is* the NLL.
- **Example**: a coin gives 7 heads in 10 flips. Likelihood $p^7(1-p)^3$; setting the derivative of the log-likelihood to zero gives $\hat p = 0.7$.

### 0.6 MAP and the probabilistic view of regularisation
- **Definition**: Maximum A Posteriori adds a prior over the parameters: $\hat\theta = \arg\max_\theta\, p(\text{data} \mid \theta)\, p(\theta)$.
- **Idea**: taking $-\log$, the prior becomes an extra term in the loss. A **Gaussian prior** on the weights gives the **L2** penalty; a **Laplace prior** gives **L1**.
- **Used in**: explains why "data loss + $\lambda R(W)$" is a principled objective, not an arbitrary hack.
- **Example**: a Gaussian prior $p(w) \propto e^{-w^2 / 2\sigma^2}$ gives $-\log p(w) = \frac{w^2}{2\sigma^2} + \text{const}$, i.e. an L2 penalty with $\lambda = \frac{1}{2\sigma^2}$.

### 0.7 Logistic regression
- **Definition**: a linear *binary* classifier that outputs a probability: $p(y = 1 \mid x) = \sigma(w \cdot x + b)$, where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid.
- **Idea**: a linear score can be any real number; the sigmoid squashes it into $(0, 1)$ so it can be read as a probability. Trained by MLE, which gives the **binary cross-entropy** loss $-[y\log p + (1-y)\log(1-p)]$.
- **Used in**: baselines for binary tasks (spam / not spam, defect / OK), the last layer of binary neural classifiers.
- **Relation to Softmax**: the Softmax classifier is the multi-class generalisation (also called multinomial logistic regression). With 2 classes, softmax reduces exactly to a sigmoid of the score difference.
- **Example**: $w \cdot x + b = 2$ gives $p = \sigma(2) \approx 0.88$. If the true label is 1, the loss is $-\log 0.88 \approx 0.13$; if it is 0, the loss is $-\log 0.12 \approx 2.13$.

### 0.8 Cross-entropy and KL divergence
- **Definition**: for a true distribution $p$ and a predicted distribution $q$: $H(p, q) = -\sum_x p(x) \log q(x)$. **KL divergence** $D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \ge 0$ measures how far $q$ is from $p$. They are related by $H(p, q) = H(p) + D_{KL}(p \| q)$.
- **Idea**: in classification $p$ is one-hot (all mass on the correct class), so $H(p) = 0$ and minimising cross-entropy is the same as minimising KL, i.e. making the predicted distribution match the true one. Cross-entropy then collapses to $-\log q(\text{correct class})$.
- **Used in**: the standard classification loss in deep learning.
- **Example**: true class is 2 of 3, predicted $q = [0.2, 0.7, 0.1]$. Loss $= -\log 0.7 \approx 0.36$. If $q = [0.1, 0.1, 0.8]$, loss $= -\log 0.1 \approx 2.30$.

### Check yourself (0)
- **Q0.1** Why does a parametric model predict faster than kNN, and what is the cost of this speed?
- **Q0.2** With the same data loss, why does L2 prefer $[0.25, 0.25, 0.25, 0.25]$ over $[1, 0, 0, 0]$? What would L1 prefer, and why?
- **Q0.3** Why do we minimise the *negative log*-likelihood instead of maximising the likelihood directly?
- **Q0.4** Which prior on the weights corresponds to L2 regularisation?
- **Q0.5** Show that cross-entropy with a one-hot target equals $-\log q_{y}$.

---

## 1. The linear score function

$$f(x_i; W, b) = W x_i + b$$

- $x_i$: image flattened to a column vector of size $D$ (CIFAR-10: $D = 3072$).
- $W$: weights of shape $K \times D$ ($K$ = number of classes, 10). $b$: bias of size $K$.
- Output: $K$ **class scores**. The predicted class is the one with the highest score.
- The input $(x_i, y_i)$ is fixed; we only control $W$ and $b$. Training means setting them so that the correct class scores higher than the others across the training set.
- Advantages over kNN: the training data can be discarded after learning, and classifying an image is a single matrix multiplication.

### Check yourself (1)
- **Q1.1** For CIFAR-10, what are the shapes of $W$, $x_i$, $b$ and $f$? How many parameters in total?
- **Q1.2** Row $k$ of $W$ is involved in computing which score?

---

## 2. Interpreting a linear classifier

- **Weighted sum of pixels**: each class score is a weighted sum of all pixel values. Positive weights mean "this colour at this position supports the class", negative weights mean it argues against it. E.g. a "ship" class may put positive weight on blue pixels near the edges (water).
- **Geometric view**: each image is a point in $D$-dimensional space. Each row of $W$ (with its bias) defines a **hyperplane**; the score is zero on the hyperplane and grows linearly as we move away from it. Changing $W$ rotates the hyperplane, changing $b$ shifts it. Without $b$, every hyperplane would have to pass through the origin.
- **Template matching**: each row of $W$ is a **template (prototype)** for one class. The score is the dot product between the image and the template, so classification is "which template matches best". This is like a nearest-neighbour classifier with **one learned image per class**, using (negative) dot product instead of L1/L2 distance.
- **Limitation**: one template per class must cover every variation of that class. The learned "horse" template looks like a two-headed horse (horses facing left and right are blended); the "car" template is mostly red because many training cars are red. A linear classifier cannot represent "red car *or* blue car" separately. Neural networks fix this with intermediate units that can specialise.

### Check yourself (2)
- **Q2.1** In the geometric view, what does changing $b$ do to the decision boundary? What would go wrong without $b$?
- **Q2.2** Why does the learned horse template look like it has two heads?
- **Q2.3** In what sense is a linear classifier a kind of nearest-neighbour classifier?

---

## 3. The bias trick

- Append a constant $1$ to every input, $x_i \to [x_i; 1]$ (size $D + 1$), and append $b$ as an extra column of $W$ (size $K \times (D + 1)$).
- Then $f(x_i; W) = W x_i$: only one parameter matrix to manage.
- The assignment code usually works with row-vector data, $X$ of shape $(N, D)$ and $W$ of shape $(D, K)$, so scores are `X @ W` with shape $(N, K)$. Same idea, transposed layout; keep track of which one you are in.

### Check yourself (3)
- **Q3.1** After the bias trick, what are the shapes of $x_i$ and $W$ for CIFAR-10?
- **Q3.2** In the assignment's layout `scores = X.dot(W)`, what are the shapes of `X`, `W`, `scores`?

---

## 4. Image preprocessing

- **Mean subtraction (centering)**: compute the mean image over the training set and subtract it from every image, so pixel values are roughly in $[-127, 127]$ instead of $[0, 255]$.
- Optionally scale so each feature is in $[-1, 1]$.
- Zero-mean data is more important than it looks: it makes optimisation (gradient descent) behave much better. The mean must be computed **on the training set only** and then applied to val/test.

### Check yourself (4)
- **Q4.1** Why must the mean image be computed on the training set only?

---

## 5. Multiclass SVM loss

$$L_i = \sum_{j \ne y_i} \max(0,\ s_j - s_{y_i} + \Delta), \qquad s = f(x_i; W)$$

- **Idea**: the correct class score should be higher than every incorrect score **by at least a margin $\Delta$**. Each incorrect class that violates the margin adds to the loss; classes already beyond the margin contribute 0.
- $\max(0, \cdot)$ is the **hinge loss**. A variant is the **squared hinge** $\max(0, \cdot)^2$, which penalises violations more strongly.
- **Example** (from the notes): scores $s = [13, -7, 11]$, correct class is the first, $\Delta = 10$.
  - Class 2: $\max(0, -7 - 13 + 10) = \max(0, -10) = 0$
  - Class 3: $\max(0, 11 - 13 + 10) = 8$
  - $L_i = 8$. The correct score beats class 3 by only 2, less than the required 10.
- The SVM only cares that the margin is satisfied; once it is, it does not care how large the gap is.

### Check yourself (5)
- **Q5.1** Recompute the example above with $\Delta = 1$.
- **Q5.2** With a small random $W$ (all scores ≈ 0), what loss should the SVM give on CIFAR-10 with $\Delta = 1$? Why is this a useful sanity check?
- **Q5.3** What changes if the sum also included $j = y_i$?

---

## 6. Regularisation in the full loss

$$L = \underbrace{\frac{1}{N}\sum_i L_i}_{\text{data loss}} + \underbrace{\lambda R(W)}_{\text{regularisation loss}}, \qquad R(W) = \sum_k\sum_l W_{k,l}^2$$

- **Why it is needed**: the set of $W$ that classify the training data correctly is not unique. If $W$ gives zero SVM loss, then $\alpha W$ for any $\alpha > 1$ also does (all margins scale up). Regularisation removes this ambiguity by preferring smaller weights.
- **Effect**: L2 prefers spread-out weights (see 0.4), so the classifier uses all input dimensions a little instead of relying heavily on a few. This tends to improve generalisation.
- **Biases are usually not regularised**: they do not control how strongly any input dimension influences the score.
- **Setting $\Delta$**: $\Delta$ and $\lambda$ control the same trade-off. The scale of $W$ directly scales the score differences, so any margin can be met by growing $W$; the only real control is how much we allow $W$ to grow, i.e. $\lambda$. So $\Delta = 1.0$ is safe to fix.
- **Regularisation can make zero loss impossible**: with $\lambda > 0$ the total loss can only reach zero when $W = 0$.

### Check yourself (6)
- **Q6.1** Explain why $W$ that achieves zero data loss is not unique.
- **Q6.2** Why is it safe to fix $\Delta = 1$ and tune only $\lambda$?
- **Q6.3** Why are biases not regularised?

---

## 7. Softmax classifier

$$L_i = -\log\left(\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}\right) = -f_{y_i} + \log\sum_j e^{f_j}$$

- **Idea**: interpret scores as **unnormalised log-probabilities**. Exponentiate to make them positive, normalise to sum to 1 (the **softmax function**), and take the negative log of the correct class's probability.
- **Information-theory view**: this is the cross-entropy between the true one-hot distribution and the predicted distribution (see 0.8). Minimising it pushes all predicted probability mass onto the correct class.
- **Probabilistic view**: $P(y_i \mid x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j}}$. Minimising the loss is **MLE** (0.5); adding L2 regularisation is **MAP with a Gaussian prior** (0.6).
- **Numerical stability**: $e^{f}$ overflows for large scores. Multiply top and bottom by a constant $C$, choosing $\log C = -\max_j f_j$, i.e. **subtract the max score before exponentiating**:
  ```python
  f = np.array([123, 456, 789])
  p = np.exp(f) / np.sum(np.exp(f))      # overflow -> nan
  f -= np.max(f)                         # f becomes [-666, -333, 0]
  p = np.exp(f) / np.sum(np.exp(f))      # safe, same result mathematically
  ```
- **Terminology**: "softmax" is the squashing function; the "Softmax classifier" is the model trained with cross-entropy loss on top of it. "Softmax loss" is common but loose naming.
- **Sanity check**: with small random $W$, all probabilities ≈ $1/K$, so the initial loss ≈ $-\log(1/10) = \log 10 \approx 2.303$ on CIFAR-10.
- **Gradient** (needed for the assignment; derived from the formula above): $\frac{\partial L_i}{\partial f_k} = p_k - \mathbb{1}[k = y_i]$. The predicted probabilities minus the one-hot target.

### Check yourself (7)
- **Q7.1** Why does subtracting $\max_j f_j$ not change the softmax probabilities?
- **Q7.2** Compute the softmax probabilities and loss for $f = [1, 2, 3]$ with the correct class being the last one.
- **Q7.3** What is the minimum possible value of the Softmax data loss? Can it be reached with finite weights?
- **Q7.4** Derive $\partial L_i / \partial f_k = p_k - \mathbb{1}[k = y_i]$ yourself (this is tomorrow's code).

---

## 8. SVM vs Softmax

- Both use the same score function $f = Wx + b$; only the interpretation of the scores and the loss differ.
- **Same scores, different losses** (example from the notes): $f = [-2.85, 0.86, 0.28]$, correct class is the third.
  - SVM ($\Delta = 1$): $\max(0, -2.85 - 0.28 + 1) + \max(0, 0.86 - 0.28 + 1) = 0 + 1.58 = 1.58$
  - Softmax: $e^f \approx [0.058, 2.36, 1.32]$, normalised $\approx [0.016, 0.631, 0.353]$, loss $= -\log 0.353 \approx 1.04$
  - The two numbers are not comparable with each other; only comparisons within one loss are meaningful.
- **Scores vs probabilities**: SVM scores are only meaningful relative to each other (margins). Softmax outputs "probabilities", but how peaky or diffuse they are depends on $\lambda$: stronger regularisation means smaller weights, smaller score differences, flatter probabilities. So treat them as **confidences that rank correctly**, not as calibrated probabilities.
- **Local objective**: the SVM is satisfied once the margins are met. Scores $[10, -2, 3]$ and $[10, -100, -100]$ (first class correct, $\Delta = 1$) both give SVM loss 0. Softmax is **never fully satisfied**: it always wants more probability on the correct class and less on the others, so the second set of scores still gives a lower loss than the first.
- **In practice**: performance is usually very similar. The SVM's "don't care once the margin is met" can be a feature (no effort wasted on examples that are already easy, e.g. a frog classifier should not spend effort on cars it already rejects).

### Check yourself (8)
- **Q8.1** Why does the SVM give the same loss to $[10, -2, 3]$ and $[10, -100, -100]$, while Softmax does not?
- **Q8.2** If $\lambda$ is increased a lot, what happens to the Softmax probabilities? Why does this mean they are not "true" probabilities?
- **Q8.3** In one sentence each: what does the SVM want, and what does the Softmax classifier want?

---

## Corrections

Misunderstandings found while answering the questions above, with the corrected understanding.

<!-- Format: **Qx.y**: what I thought → what is actually correct, and why. -->
