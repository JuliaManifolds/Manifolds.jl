# Symmetric Positive Definite Matrices

The symmetric positive definite matrices

```math
\mathcal P(n) = \bigl\{ A \in \mathbb R^{n\times n}\ \big|\ A = A^{\mathrm{T}} \text{ and } x^{\mathrm{T}}Ax > 0 \text{ for } 0\neq x \in\mathbb R^n \bigr\}
```

```@docs
SymmetricPositiveDefinite
```

can -- for example -- be illustrated as ellipsoids:  since the eigen values are all positive
they can be taken as lengths of the axes of an ellipsoids while the directions are given by
the eigenvectors.

![An example set of data](../assets/images/SPDSignal.png)

The manifold can be equipped with different metrics

## Common and Metric Independent functions
```@docs
injectivity_radius(::SymmetricPositiveDefinite{N},a::Vararg{Any,N} where N) where N
check_manifold_point(::SymmetricPositiveDefinite{N},x; kwargs...) where N
check_tangent_vector(::SymmetricPositiveDefinite{N},x,v; kwargs...) where N
manifold_dimension(::SymmetricPositiveDefinite{N}) where N
representation_size(::SymmetricPositiveDefinite{N}) where N
zero_tangent_vector(::SymmetricPositiveDefinite{N},x) where N
zero_tangent_vector!(::SymmetricPositiveDefinite{N}, v, x) where N
```

## Default Metric: Linear Affine Metric

```@docs
LinearAffineMetric
```

This metric is also the default metric, i.e.
any call of the following functions with
`P=SymmetricPositiveDefinite(3)` will result in
`MetricManifold(P,LinearAffineMetric())`and hence yield the formulae described in this seciton.

```@docs
distance(P::SymmetricPositiveDefinite{N},x,y) where N
exp!(P::SymmetricPositiveDefinite{N}, y, x, v) where N
inner(P::SymmetricPositiveDefinite, x, w, v)
log!(P::SymmetricPositiveDefinite{N}, v, x, y) where N
tangent_orthonormal_basis(P::SymmetricPositiveDefinite{N},x,v) where N
vector_transport_to!(P::SymmetricPositiveDefinite{N},vto, x, v, y, m::ParallelTransport) where N
```

## Log Euclidean Metric

```@docs
LogEuclideanMetric
```

And we obtain the following functions

```@docs
distance(P::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric},x,y) where N
```

## Log Cholesky Metric

```@docs
LogCholeskyMetric
```

```@docs
distance(P::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},x,y) where N
exp!(P::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric}, y, x, v) where N
inner(P::MetricManifold{SymmetricPositiveDefinite{N}, LogCholeskyMetric}, x, w, v) where N
log!(P::MetricManifold{SymmetricPositiveDefinite{N}, LogCholeskyMetric}, v, x, y) where N
vector_transport_to!(M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric}, vto, x, v, y, ::ParallelTransport) where N
```

### Literature
