# Symmetric Positive Definite Matrices

The symmetric positive definite matrices

```math
\mathcal P(n) = \bigl\{ A \in \mathbb R^{n\times n}\ \big|\ 
A = A^{\mathrm{T}} \text{ and }
x^{\mathrm{T}}Ax > 0 \text{ for } 0\neq x \in\mathbb R^n \bigr\}
```

```@docs
SymmetricPositiveDefinite
```

can -- for example -- be illustrated as ellipsoids:  since the eigen values are all positive
they can be taken as lengths of the axes of an ellipsoids while the directions are given by
the eigenvectors. 

![An example set of data](../assets/images/SPDSignal.png)

The manifold can be equipped with different metrics

## Checks
```@docs
is_manifold_point(P::SymmetricPositiveDefinite{N},x; kwargs...) where N
is_tangent_vector(P::SymmetricPositiveDefinite{N},x,v; kwargs...) where N
```


## Linear Affine Metric

```@docs
LinearAffineMetric
```

This metric yields the following functions

```@docs
distance(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},x,y) where N
exp!(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, y, x, v) where N
injectivity_radius(P::SymmetricPositiveDefinite, args...)
inner(P::MetricManifold{SymmetricPositiveDefinite{N}, LinearAffineMetric}, x, w, v) where N
log!(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, v, x, y) where N
vector_transport!(::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, vto, x, v, y, ::ParallelTransport) where N
```


## Log Euclidean Metric

```@docs
LogEuclideanMetric
```