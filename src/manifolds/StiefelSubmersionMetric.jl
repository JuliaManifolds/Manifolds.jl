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

struct StiefelKShootingInverseRetraction{T<:Real} <: AbstractInverseRetractionMethod
    max_iterations::Int
    tolerance::T
    num_transport_points::Int
end
function StiefelKShootingInverseRetraction(;
    max_iterations=1_000,
    tolerance=sqrt(eps()),
    num_transport_points=4,
)
    return StiefelKShootingInverseRetraction(
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
    T = Base.promote_eltype(q, p, X)
    # TODO:
    # - dispatch to exp! for α = -1/2 and α = 0 for efficiency
    # - handle rank-deficient QB
    if k ≤ div(n, 2)
        # eq. 11
        A = allocate(q, T, Size(k, k))
        Y = allocate(X, T)
        F = allocate(A, T, Size(2k, 2k))
        G = allocate(p, T, Size(n, 2k))
        mul!(A, p', X)
        copyto!(Y, X)
        mul!(Y, p, A, -1, true)
        Q, B = qr(Y)
        @views begin
            copyto!(G[1:n, 1:k], p)
            copyto!(G[1:n, (k + 1):(2k)], Matrix(Q))
            F[1:k, 1:k] .= A ./ (α + 1)
            F[1:k, (k + 1):(2k)] .= -B'
            F[(k + 1):(2k), 1:k] .= B
            fill!(F[(k + 1):(2k), (k + 1):(2k)], false)
        end
        rmul!(A, α / (α + 1))
        mul!(q, G, @views(exp(F)[1:(2k), 1:k]) * exp(A))
    elseif n == k
        A = allocate(q, T)
        C = allocate(q, T)
        mul!(A, p', X, α / (α + 1), false)
        mul!(C, X, p', inv(α + 1), false)
        expC = exp(C)
        tmp = mul!(C, p, exp(A))
        mul!(q, expC, tmp)
    else  # n/2 < k < n
        # eq. 8
        A = p' * X
        C = allocate(q, T, Size(n, n))
        tmp = allocate(q, T)
        mul!(C, X, p')
        C .-= C'
        mul!(tmp, p, A)
        mul!(C, tmp, p', -(2α + 1) / (α + 1), true)
        rmul!(A, α / (α + 1))
        mul!(tmp, p, exp(A))
        mul!(q, exp(C), tmp)
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
    gap < method.tolerance && return X
    rmul!(X, gap / norm(X))
    i = 1
    Xˢ = allocate(X)
    retr_tX = allocate_result(M, retract, p, X)
    retr_tX_new = allocate_result(M, retract, p, X)
    while (gap > method.tolerance) && (i < method.max_iterations)
        retract!(M, retr_tX, p, X, method.retraction)
        Xˢ .= retr_tX .- q
        gap = norm(Xˢ)
        project!(M, Xˢ, retr_tX, Xˢ)
        rmul!(Xˢ, gap / norm(Xˢ))
        for t in reverse(ts)[2:end-1]
            retract!(M, retr_tX_new, p, t * X, method.retraction)
            vector_transport_to!(M, Xˢ, retr_tX, Xˢ, retr_tX_new, method.vector_transport)
            retr_tX, retr_tX_new = retr_tX_new, retr_tX
            Xˢ, Xˢ = Xˢ, Xˢ
        end
        vector_transport_to!(M, Xˢ, retr_tX, Xˢ, p, method.vector_transport)
        X .-= Xˢ
        i += 1
    end
    return X
end

function inverse_retract(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    p,
    q,
    method::StiefelKShootingInverseRetraction,
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
    method::StiefelKShootingInverseRetraction,
) where {n,k}
    ts = range(0, 1; length=method.num_transport_points)
    α = metric(M).α
    M̂ = p'q
    skewM̂ = project(SkewHermitianMatrices(k), M̂)
    Q, N̂ = qr(q - p * M̂)
    normN̂² = norm(N̂)^2
    gap = sqrt(norm(M̂ - I)^2 + normN̂²) # γ
    if gap < method.tolerance
        mul!(X, Matrix(Q), R)
        return X
    end
    c = gap / sqrt(norm(skewM̂)^2 + normN̂²)
    A = rmul!(skewM̂, c)
    R = c * N̂
    S = allocate(A)
    Aˢ = allocate(A)
    Rˢ = allocate(R)
    D = allocate(A)
    C = allocate(A, Size(2k, 2k))
    i = 1
    while (gap > method.tolerance) && (i < method.max_iterations)
        @views begin
            C[1:k, 1:k] .= A ./ (α + 1)
            C[1:k, (k + 1):(2k)] .= -R'
            C[(k + 1):(2k), 1:k] .= R
            fill!(C[(k + 1):(2k), (k + 1):(2k)], false)
        end
        D .= A .* (α / (α + 1))
        @views begin
            E = exp(C)[:, 1:k] * exp(D)
            M = E[1:k, 1:k]
            N = E[(k + 1):(2k), 1:k]
        end
        Aˢ .= M .- M̂
        Rˢ .= N .- N̂
        gap = sqrt(norm(Aˢ)^2 + norm(Rˢ)^2)
        _para_trans_kfactors!(Aˢ, Rˢ, S, M, N, gap, method.tolerance, Val(k))
        for t in reverse(ts)[2:end-1]
            @views begin
                E = exp(t * C)[:, 1:k] * exp(t * D)
                M = E[1:k, 1:k]
                N = E[(k + 1):(2k), 1:k]
            end
            _para_trans_kfactors!(Aˢ, Rˢ, S, M, N, gap, method.tolerance, Val(k))
        end
        copyto!(M, I)
        fill!(N, 0)
        _para_trans_kfactors!(Aˢ, Rˢ, S, M, N, gap, method.tolerance, Val(k))
        A .-= Aˢ
        R .-= Rˢ
        i += 1
    end
    mul!(X, p, A)
    mul!(X, Matrix(Q), R, true, true)
    return X
end

_sym(X) = (X + X') / 2

function _para_trans_kfactors!(A, R, S, M, N, gap, ϵ, ::Val{k}) where {k}
    mul!(S, M', A)
    mul!(S, N', R, true, true)
    project!(SymmetricMatrices(k), S, S)
    mul!(A, M, S, -1, true)
    mul!(R, N, S, -1, true)
    l = sqrt(norm(A)^2 + norm(R)^2)
    if l > ϵ
        c = gap / l
        rmul!(A, c)
        rmul!(R, c)
    else
        fill!(A, 0)
        fill!(R, 0)
    end
    return A, R
end

function log!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    X,
    p,
    q,
) where {n,k}
    T = float(real(Base.promote_eltype(X, p, q)))
    tolerance = sqrt(eps(T))
    if k ≤ div(n, 2)
        inverse_retraction =
            StiefelKShootingInverseRetraction(tolerance=tolerance, num_transport_points=4)
    else
        inverse_retraction =
            StiefelShootingInverseRetraction(tolerance=tolerance, num_transport_points=4)
    end
    return inverse_retract!(M, X, p, q, inverse_retraction)
end
