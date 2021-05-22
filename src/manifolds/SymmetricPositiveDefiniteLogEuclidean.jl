@doc raw"""
    LogEuclideanMetric <: RiemannianMetric

The LogEuclidean Metric consists of the Euclidean metric applied to all elements after mapping them
into the Lie Algebra, i.e. performing a matrix logarithm beforehand.
"""
struct LogEuclideanMetric <: RiemannianMetric end

@doc raw"""
    distance(M::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric}, p, q)

Compute the distance on the [`SymmetricPositiveDefinite`](@ref) manifold between
`p` and `q` as a [`MetricManifold`](@ref) with [`LogEuclideanMetric`](@ref).
The formula reads

```math
    d_{\mathcal P(n)}(p,q) = \lVert \operatorname{Log} p - \operatorname{Log} q \rVert_{\mathrm{F}}
```

where $\operatorname{Log}$ denotes the matrix logarithm and
$\lVert\cdot\rVert_{\mathrm{F}}$ denotes the matrix Frobenius norm.
"""
function distance(
    M::MetricManifold{ℝ,SymmetricPositiveDefinite{N},LogEuclideanMetric},
    p,
    q,
) where {N}
    return norm(log(Symmetric(p)) - log(Symmetric(q)))
end
