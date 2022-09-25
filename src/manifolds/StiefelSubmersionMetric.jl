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

function inverse_retract!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    X,
    p,
    q,
    method::ShootingInverseRetraction{ExponentialRetraction},
) where {n,k}
    if k > div(n, 2)
        _shooting!(
            M,
            X,
            p,
            q;
            retraction=method.retraction,
            vector_transport=method.vector_transport,
            initial_inverse_retraction=method.initial_inverse_retraction,
            num_transport_points=method.num_transport_points,
            tolerance=method.tolerance,
            max_iterations=method.max_iterations,
        )
        return X
    end
    ts = range(0, 1; length=method.num_transport_points)
    α = metric(M).α
    M̂ = p'q
    skewM̂ = project(SkewHermitianMatrices(k), M̂)
    Q, N̂ = qr(q - p * M̂)
    normN̂² = norm(N̂)^2
    gap = sqrt(norm(M̂ - I)^2 + normN̂²) # γ
    if iszero(gap)
        fill!(X, false)
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
        _vector_transport_factors!(Aˢ, Rˢ, S, M, N, gap, method.tolerance, Val(k))
        for t in reverse(ts)[2:(end - 1)]
            @views begin
                E = exp(t * C)[:, 1:k] * exp(t * D)
                M = E[1:k, 1:k]
                N = E[(k + 1):(2k), 1:k]
            end
            _vector_transport_factors!(Aˢ, Rˢ, S, M, N, gap, method.tolerance, Val(k))
        end
        copyto!(M, I)
        fill!(N, 0)
        _vector_transport_factors!(Aˢ, Rˢ, S, M, N, gap, method.tolerance, Val(k))
        A .-= Aˢ
        R .-= Rˢ
        i += 1
    end
    mul!(X, p, A)
    mul!(X, Matrix(Q), R, true, true)
    return X
end

function inverse_retract_project!(
    M::MetricManifold{ℝ,<:Stiefel,<:StiefelSubmersionMetric},
    X,
    p,
    q,
)
    X .= q .- p
    project!(M, X, p, X)
    return X
end

function _vector_transport_factors!(A, R, S, M, N, gap, ϵ, ::Val{k}) where {k}
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

function log(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    p,
    q;
    tolerance=sqrt(eps(float(real(Base.promote_eltype(p, q))))),
    max_iterations=1_000,
    num_transport_points=4,
) where {n,k}
    X = allocate_result(M, log, p, q)
    log!(
        M,
        X,
        p,
        q;
        tolerance=tolerance,
        max_iterations=max_iterations,
        num_transport_points=num_transport_points,
    )
    return X
end
function log!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    X,
    p,
    q;
    tolerance=sqrt(eps(float(real(eltype(X))))),
    max_iterations=1_000,
    num_transport_points=4,
) where {n,k}
    retraction = ExponentialRetraction()
    initial_inverse_retraction = ProjectionInverseRetraction()
    vector_transport = ScaledVectorTransport(ProjectionTransport())
    inverse_retraction = ShootingInverseRetraction(
        retraction,
        initial_inverse_retraction,
        vector_transport,
        num_transport_points,
        tolerance,
        max_iterations,
    )
    return inverse_retract!(M, X, p, q, inverse_retraction)
end
