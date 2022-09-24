@doc raw"""
    StiefelSubmersionMetric{T<:Real} <: RiemannianMetric

The submersion (or normal) metric family on the [`Stiefel`](@ref) manifold.

# Constructor

    StiefelSubmersionMetric(α)

Construct the submersion metric on the Stiefel manifold with the parameter ``α > -1``.

The submersion metric family has two special cases:
- ``α = -\frac{1}{2}``: [`EuclideanMetric`](@ref)
- ``α = 0``: [`CanonicalMetric`](@ref)
"""
struct StiefelSubmersionMetric{T<:Real} <: RiemannianMetric
    α::T
end


function exp!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    q,
    p,
    X,
) where {n,k}
    α = metric(M).α
    # TODO:
    # - dispatch to exp! for α = -1/2 and α = 0 for efficiency
    # - reduce allocations
    # - handle rank-deficient QB
    if k ≤ div(n, 2)
        # eq. 11
        A = p'X
        Q, B = qr(X - p * A)
        copyto!(
            q,
            [p Matrix(Q)] *
            exp([A/(α + 1) -B'; B zero(B)])[:, 1:k] *
            exp(A * (α / (α + 1))),
        )
    elseif n == k
        copyto!(q, exp((X * p') / (α + 1)) * p * exp((p' * X) * (α / (α + 1))))
    else  # n/2 < k < n
        # eq. 8
        A = p' * X
        C = X * p'
        copyto!(
            q,
            exp((-(2α + 1) / (α + 1)) * (p * A * p') + C - C') * p * exp((α / (α + 1)) * A),
        )
    end
    return q
end

@doc raw"""
    inner(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, X, <:StiefelSubmersionMetric}, p, X, Y)

Compute the inner product on the [`Stiefel`](@ref) manifold with respect to the
[`StiefelSubmersionMetric`](@ref). The formula reads
```math
g_p(X,Y) = \operatorname{tr}\bigl( X^{\mathrm{T}}(I_n - \frac{2α+1}{2(α+1)}pp^{\mathrm{T}})Y \bigr),
```
where ``α`` is the parameter of the metric.
"""
function inner(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    p,
    X,
    Y,
) where {n,k}
    α = metric(M).α
    T = typeof(one(Base.promote_eltypeof(p, X, Y, α)))
    if n == k
        return T(dot(X, Y)) / (2 * (α + 1))
    elseif α == -1 // 2
        return T(dot(X, Y))
    else
        return dot(X, Y) - (T(dot(p'X, p'Y)) * (2α + 1)) / (2 * (α + 1))
    end
end
