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

struct StiefelShootingInverseRetraction{
    T<:Real,
    R<:AbstractRetractionMethod,
    VT<:AbstractVectorTransportMethod,
} <: AbstractInverseRetractionMethod
    max_iterations::Int
    tolerance::T
    num_transport_points::Int
    retraction::R
    vector_transport::VT
end
function StiefelShootingInverseRetraction(;
    max_iterations=1_000,
    tolerance=sqrt(eps()),
    num_transport_points=4,
    retraction=ExponentialRetraction(),
    vector_transport=ScaledVectorTransport(ProjectionTransport()),
)
    return StiefelShootingInverseRetraction(
        max_iterations,
        tolerance,
        num_transport_points,
        retraction,
        vector_transport,
    )
end

struct StiefelPShootingInverseRetraction{T<:Real} <: AbstractInverseRetractionMethod
    max_iterations::Int
    tolerance::T
    num_transport_points::Int
end
function StiefelPShootingInverseRetraction(;
    max_iterations=1_000,
    tolerance=sqrt(eps()),
    num_transport_points=4,
)
    return StiefelPShootingInverseRetraction(
        max_iterations,
        tolerance,
        num_transport_points,
    )
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

function inverse_retract(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    p,
    q,
    method::StiefelShootingInverseRetraction,
) where {n,k}
    X = allocate_result(M, inverse_retract, p, q)
    inverse_retract!(M, X, p, q, method)
    return X
end
function inverse_retract!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    X,
    p,
    q,
    method::StiefelShootingInverseRetraction,
) where {n,k}
    T = real(Base.promote_eltype(X, p, q))
    ts = range(zero(T), one(T); length=method.num_transport_points)
    X .= q .- p
    gap = norm(X)
    project!(M, X, p, X)
    rmul!(X, gap / norm(X))
    i = 1
    Xˢ = allocate(X)
    retr_tX = allocate_result(M, retract, p, X)
    retr_tX_new = allocate_result(M, retract, p, X)
    while (gap > method.tolerance) && (i < method.max_iterations)
        retract!(M, retr_tX, p, X, method.retraction)
        Xˢ .= retr_tX .- q
        gap = norm(Xˢ)
        for t in reverse(ts)
            retract!(M, retr_tX_new, p, t * X, method.retraction)
            vector_transport_to!(M, Xˢ, retr_tX, Xˢ, retr_tX_new, method.vector_transport)
            retr_tX, retr_tX_new = retr_tX_new, retr_tX
        end
        X .-= Xˢ
        i += 1
    end
    return X
end

function inverse_retract(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    p,
    q,
    method::StiefelPShootingInverseRetraction,
) where {n,k}
    X = allocate_result(M, inverse_retract, p, q)
    inverse_retract!(M, X, p, q, method)
    return X
end
function inverse_retract!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    X,
    p,
    q,
    method::StiefelPShootingInverseRetraction,
) where {n,k}
end

