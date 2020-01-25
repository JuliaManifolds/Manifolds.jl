@doc doc"""
    LogEuclideanMetric <: Metric

The LogEuclidean Metric consists of the Euclidean metric applied to all elements after mapping them
into the Lie Algebra, i.e. performing a matrix logarithm beforehand.
"""
struct LogEuclideanMetric <: RiemannianMetric end

@doc doc"""
    distance(M::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric}, x, y)

Compute the distance on the [`SymmetricPositiveDefinite`](@ref) manifold between
`x` and `y` as a [`MetricManifold`](@ref) with [`LogEuclideanMetric`](@ref).
The formula reads

```math
    d_{\mathcal P(n)}(x,y) = \lVert \Log x - \Log y \rVert_{\mathrm{F}}
```

where $\operatorname{Log}$ denotes the matrix logarithm and
$\lVert·\rVert_{\mathrm{F}}$ denotes the matrix Frobenius norm.
"""
function distance(
    M::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric},
    x,
    y,
) where {N}
    return norm(log(Symmetric(x)) - log(Symmetric(y)))
end
