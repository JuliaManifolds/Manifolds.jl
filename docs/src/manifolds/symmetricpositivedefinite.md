# Symmetric positive definite matrices

The symmetric positive definite matrices

```math
\mathcal P(n) = \bigl\{ A \in \mathbb R^{n\times n}\ \big|\ A = A^{\mathrm{T}} \text{ and } x^{\mathrm{T}}Ax > 0 \text{ for } 0\neq x \in\mathbb R^n \bigr\}
```

```@docs
SymmetricPositiveDefinite
```

can -- for example -- be illustrated as ellipsoids:  since the eigen values are all positive they can be taken as lengths of the axes of an ellipsoids while the directions are given by the eigenvectors.

![An example set of data](../assets/images/SPDSignal.png)

The manifold can be equipped with different metrics

## Common and Metric Independent functions

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/SymmetricPositiveDefinite.jl"]
Order = [:function]
 Filter = t -> t !== mean
```

## Default Metric: Linear Affine Metric

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/SymmetricPositiveDefiniteLinearAffine.jl"]
Order = [:type]
```

This metric is also the default metric, i.e. any call of the following functions with `P=SymmetricPositiveDefinite(3)` will result in `MetricManifold(P,LinearAffineMetric())`and hence yield the formulae described in this seciton.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/SymmetricPositiveDefiniteLinearAffine.jl"]
Order = [:function]
```

## Log Euclidean Metric

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/SymmetricPositiveDefiniteLogEuclidean.jl"]
Order = [:type, :function]
```

## Log Cholesky Metric

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/SymmetricPositiveDefiniteLogCholesky.jl"]
Order = [:type, :function]
```

## Statistics

```@autodocs
Modules = [Manifolds]
Pages   = ["SymmetricPositiveDefinite.jl"]
Order = [:function]
Filter = t -> t === mean
```

## Literature
