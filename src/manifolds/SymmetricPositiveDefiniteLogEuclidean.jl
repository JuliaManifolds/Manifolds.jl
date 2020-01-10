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
$\lVert\cdot\rVert_{\mathrm{F}}$ denotes the matrix Frobenius norm.
"""
function distance(M::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric},x,y) where N
    return norm(log(Symmetric(x)) - log(Symmetric(y)))
end

function get_coordinates(M::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric}, x, B::ArbitraryOrthonormalBasis) where N
    dim = manifold_dimension(M)
    vout = similar(v, dim)
    k = 1
    for i in 1:N, j in i:N
        scale = ifelse(i==j, 1, sqrt(2))
        vout[k] = v[i,j]*scale
        k += 1
    end
    return vout
end

function get_vector(M::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric}, x, B::ArbitraryOrthonormalBasis) where N
    dim = manifold_dimension(M)
    vout = similar_result(M, get_vector, x)
    k = 1
    for i in 1:N, j in i:N
        scale = ifelse(i==j, 1, 1/sqrt(2))
        vout[i,j] = v[k]*scale
        vout[j,i] = v[k]*scale
        k += 1
    end
    return vout
end
