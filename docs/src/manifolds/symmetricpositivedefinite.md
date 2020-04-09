# Symmetric positive definite matrices

```@docs
SymmetricPositiveDefinite
```

This manifold can -- for example -- be illustrated as ellipsoids:  since the eigenvalues are all positive they can be taken as lengths of the axes of an ellipsoids while the directions are given by the eigenvectors.

![An example set of data](../assets/images/SPDSignal.png)

The manifold can be equipped with different metrics

## Common and metric independent functions

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/SymmetricPositiveDefinite.jl"]
Order = [:function]
 Filter = t -> t !== mean
```

## Default metric: the linear affine metric

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

## The log Euclidean metric

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/SymmetricPositiveDefiniteLogEuclidean.jl"]
Order = [:type, :function]
```

## log Cholesky metric

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
