# Similarity-Based Kernel Optimization

This repository contains an implementation of the paper [Learning Kernel Parameters for Support Vector Classification Using Similarity Embeddings](https://www.esann.org/sites/default/files/proceedings/2024/ES2024-90.pdf), presented in ESANN 24. It consists of a similarity-based approach for optimizing the width parameter (sigma) in radial basis function (RBF) kernels.

Supported kernel types are:
- Gaussian kernel: $K(x_i, x_j) = \exp\left(-\frac{1}{2}\frac{||x_i - x_j||**2}{\sigma**2}\right)$.
- Laplacian kernel: $K(x_i, x_j) = \exp\left(-\frac{1}{2}\frac{||x_i - x_j||}{\sigma}\right)$.

## Files

- `similarity_optimization.py`: Contains the core functions for generating similarity spaces, computing RBF loss, and optimizing the kernel width.

## Functions

### `generate_similarity_space(X, y, sigma, kernel="gaussian")`

Generates a similarity space using a specified kernel function (Gaussian or Laplacian).

### `rbf_loss(X, y, sigma, kernel="gaussian")`

Computes the RBF loss based on class mean similarities.

### `optimize_width(X, y)`

Finds the optimal sigma value to minimize the RBF loss using numerical optimization.

## Usage

This module can be used for tuning kernel parameters in machine learning tasks where similarity-based representations are important. Import the functions and apply them to your dataset to determine the best kernel width for classification tasks.

## Dependencies

- `numpy`
- `scipy`

## License

This project is open-source and available under the Apache 2.0 License.
