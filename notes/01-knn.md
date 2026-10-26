# 01 · Image classification and k-Nearest Neighbor

Sources: CS231n notes *Image Classification* (cs231n.github.io/classification) and the kNN part of Assignment 1 (`knn.ipynb`, `cs231n/classifiers/k_nearest_neighbor.py`).

---

## 1. Key ideas from the CS231n notes

### 1.1 The task and the data-driven approach
- **Image classification**: assign an image one label from a fixed set of labels. To a computer an image is just an array of numbers, e.g. a CIFAR-10 image is `32 × 32 × 3 = 3072` values in [0, 255].
- Main challenges: viewpoint variation, scale variation, deformation, occlusion, illumination, background clutter, intra-class variation. A good model must be invariant to these changes while staying sensitive to differences between classes.
- Hand-written rules ("a cat has two pointy ears...") do not work. Instead we use a **data-driven approach**: provide many labeled examples and let the algorithm learn.
- General pipeline: **input** (labeled training set) → **learning** (train a classifier) → **evaluation** (predict on unseen images, compare with the true labels).

### 1.2 Nearest Neighbor (NN)
- **Train**: memorise the whole training set. **Predict**: find the closest training image and return its label.
- Distance between two images (flattened into vectors):
  - **L1 (Manhattan)**: $d_1(I_1, I_2) = \sum_p |I_1^p - I_2^p|$
  - **L2 (Euclidean)**: $d_2(I_1, I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}$
- L2 penalises large differences more than L1: L2 prefers many moderate differences over one big one. L1 depends on the choice of coordinate axes (rotating the axes changes L1), L2 does not.
- On CIFAR-10, NN with L1 reaches about 38.6%, versus 10% for random guessing and about 95% for humans or CNNs.

### 1.3 k-Nearest Neighbor
- Take the **k** closest training images and **take a majority vote** over their labels.
- Larger k gives **smoother decision boundaries** and is less sensitive to noisy points or outliers. k = 1 fits the training set perfectly and overfits easily.

### 1.4 Choosing hyperparameters with a validation set
- **k** and the **distance metric** are hyperparameters: they are not learned from the data and have to be chosen.
- **Never use the test set to choose hyperparameters.** Doing so tunes the model to the test set, and the test result no longer reflects generalisation. The test set is used **exactly once**, at the very end.
- Hold out part of the training data as a **validation set** for choosing hyperparameters.
- **Cross-validation** (when data is scarce): split the training data into n folds (typically 3, 5 or 10), use each fold in turn as validation and train on the rest, then average the results. More stable but n times more expensive, so with large models people usually use a single validation split.

### 1.5 Pros and cons of kNN
- **Pros**: easy to understand and implement, training is instant.
- **Cons**: **prediction is expensive**, since every test image is compared with the whole training set. In practice we want the opposite: slow training is fine, but prediction must be fast (which is exactly what CNNs give). Prediction can be sped up with Approximate Nearest Neighbor search (e.g. FLANN, FAISS) at some cost in accuracy.
- **Pixel distances do not capture semantics**: an image shifted by a few pixels, partially occluded or darkened can be as far in L2 from the original as a completely different image. The t-SNE figure in the notes shows that nearby images are close mostly because of **background color and brightness distribution**, not because the objects are similar.
- **Curse of dimensionality**: densely covering a 3072-dimensional space needs exponentially many samples, so the "nearest neighbour" is still very far away.

### 1.6 If you really need kNN (from the notes)
1. Normalise the data (zero mean, unit variance per dimension).
2. For high-dimensional data, consider dimensionality reduction (PCA, random projection).
3. Split train/val (about 70–90% train), use cross-validation if time allows.
4. Try many values of k and both L1 and L2.
5. If prediction is too slow, use an ANN library.
6. Record the best hyperparameters. You may retrain on train + val, but then the hyperparameters can no longer be evaluated.

---

## 2. Takeaways from the assignment

### 2.1 Three ways to compute the distance matrix
`dists` has shape `(num_test, num_train)`, where `dists[i, j]` is the L2 distance between test point `i` and training point `j`.

| Method | Idea | Notes |
|---|---|---|
| Two loops | loop over `i`, loop over `j`, compute `sqrt(sum((x_i - t_j)²))` | easiest to get right, used as the reference |
| One loop | loop over `i`, broadcast `X_train - X[i]` with shape `(num_train, D)` | can be **slower** than two loops because it creates a `(num_train, D)` temporary every iteration |
| No loops | expand the square, use matrix multiplication | by far the fastest |

### 2.2 The no-loop trick (the most important part)
Expand:
$$\|x - t\|^2 = \|x\|^2 - 2\,x \cdot t + \|t\|^2$$

For the whole matrix:
```python
test_sq  = np.sum(X**2, axis=1, keepdims=True)   # (num_test, 1)
train_sq = np.sum(self.X_train**2, axis=1)       # (num_train,)
cross    = X @ self.X_train.T                    # (num_test, num_train)
dists = np.sqrt(np.maximum(test_sq - 2 * cross + train_sq, 0))
```
- Two **broadcasts**: `(num_test, 1)` stretches across columns, `(num_train,)` stretches across rows.
- The heavy part is `X @ X_train.T`, which runs in BLAS and is very fast. Same principle as in cv-numpy-drills: push the loops down into a library written in C.
- `np.maximum(..., 0)`: floating-point error can make the difference a tiny negative number, and `sqrt` would return `nan`.
- Correctness check: compare with the two-loop version using the **Frobenius norm** of the difference; the notebook treats it as a match when it is below 0.001.

### 2.3 `predict_labels`
- `np.argsort(dists[i])[:k]` gives the indices of the k nearest neighbours; then look up their labels in `y_train`.
- Vote with `np.bincount(closest_y).argmax()`. On a tie, `argmax` returns the **smallest** index, which is exactly the "pick the smaller label" rule the assignment asks for.

### 2.4 Cross-validation to choose k
- Split with `np.array_split(X_train, num_folds)`. For each fold: `np.concatenate` the other folds as training data, use that fold as validation.
- **Compute `dists` once per fold, not once per (fold, k) pair.** The distance matrix depends only on the split, not on k; only the voting step changes. With 5 folds and 10 values of k, that is 5 distance computations instead of 50.
- Plot mean accuracy with standard deviation (error bars) against k and pick the k with the highest mean. The error bars show how much the result varies between folds, rather than relying on a single number.
- Typical results with the 5000-image training subset: k = 1 around 27%, the best k (often around 10) around 28–29% on test. Very low, consistent with "pixel distances carry no semantics".

### 2.5 Inline questions (summary of my answers)
- **Q1, bright rows and columns in the distance matrix**: a bright row is a test image that is far from almost every training image; a bright column is a training image far from almost every test image. The cause is unusual **raw pixel values** (very dark, very bright, high contrast, unusual background color), not the content. L2 sums over 3072 values, so overall brightness and color dominate the object.
- **Q2, which preprocessing steps leave an L1 NN classifier unchanged**:
  - Subtract the global mean $\mu$: unchanged (differences between images stay the same).
  - Subtract the per-pixel mean $\mu_{ij}$: unchanged (same reason, each dimension shifts by the same constant).
  - Subtract $\mu$ and divide by the global $\sigma$: unchanged (all distances are scaled by the same factor $1/\sigma$, so the neighbour ranking is preserved).
  - Divide by the per-pixel $\sigma_{ij}$: **changes**, because each dimension is scaled differently.
  - Rotate the coordinate axes: **changes**, because L1 depends on the axes (L2 does not).
- **Q3, true statements about kNN**: (2) the training error of 1-NN is always ≤ that of 5-NN, since 1-NN on the training set always finds the point itself, so its error is 0; (4) classification time grows with the size of the training set. The decision boundary is not linear (it is piecewise linear along Voronoi cells), and 1-NN does not always have lower test error than 5-NN.

---

## 3. Interview self-check
1. **Why not choose k on the test set?** Because the test set then becomes part of training; test numbers become optimistic and no longer measure generalisation.
2. **What do kNN training and prediction cost?** Training is O(1) (just store the data). Prediction is O(N·D) per image, where N is the training set size and D the dimensionality. The opposite of what we want in practice.
3. **How do you compute pairwise L2 distances between two point sets without loops?** Expand to $\|x\|^2 - 2x\cdot t + \|t\|^2$: one matrix multiplication and two broadcasts; clamp negatives to 0 before the square root.
4. **How do L1 and L2 differ?** L2 penalises large differences more and is invariant to rotating the axes; L1 is more robust to outliers and depends on the axes.
5. **Why does kNN on raw pixels work poorly for images?** Pixel distances are dominated by brightness, background color and position; they are not invariant to shifts or occlusion; and the dimensionality is very high (curse of dimensionality).
