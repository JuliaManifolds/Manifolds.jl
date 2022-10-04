@doc raw"""
    StiefelSubmersionMetric{T<:Real} <: RiemannianMetric

The submersion (or normal) metric family on the [`Stiefel`](@ref) manifold.

The family, with a single real parameter ``α>-1``, has two special cases:
- ``α = -\frac{1}{2}``: [`EuclideanMetric`](@ref)
- ``α = 0``: [`CanonicalMetric`](@ref)

The family was described in [^HüperMarkinaLeite2021]. This implementation follows the
description in [^ZimmermanHüper2022].

[^HüperMarkinaLeite2021]:
    > Hüper, M., Markina, A., Leite, R. T. (2021)
    > "A Lagrangian approach to extremal curves on Stiefel manifolds"
    > Journal of Geometric Mechanics, 13(1): 55-72.
    > doi: [10.3934/jgm.2020031](http://dx.doi.org/10.3934/jgm.2020031)
[^ZimmermanHüper2022]:
    > Ralf Zimmerman and Knut Hüper. (2022).
    > "Computing the Riemannian logarithm on the Stiefel manifold: metrics, methods and performance."
    > arXiv: [2103.12046](https://arxiv.org/abs/2103.12046)

# Constructor

    StiefelSubmersionMetric(α)

Construct the submersion metric on the Stiefel manifold with the parameter ``α``.
"""
struct StiefelSubmersionMetric{T<:Real} <: RiemannianMetric
    α::T
end

@doc raw"""
    q = exp(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, <:StiefelSubmersionMetric}, p, X)
    exp!(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, q, <:StiefelSubmersionMetric}, p, X)

Compute the exponential map on the [`Stiefel(n,k)`](@ref) manifold with respect to the [`StiefelSubmersionMetric`](@ref).

The exponential map is given by
````math
\exp_p X = \operatorname{Exp}\bigl(
    -\frac{2α+1}{α+1} p p^\mathrm{T} X p^\mathrm{T} +
    X p^\mathrm{T} - p X^\mathrm{T}
\bigr) p \operatorname{Exp}\bigl(\frac{\alpha}{\alpha+1} p^\mathrm{T} X\bigr)
````
"""
exp(::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric}, ::Any...) where {n,k}

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

"""
    inverse_retract(
        M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
        p,
        q,
        method::ProjectionInverseRetraction,
    )

Compute a projection-based inverse retraction.

The retraction is computed by projecting the logarithm map in the embedding to the tangent
space at ``p``.
"""
inverse_retract(
    ::MetricManifold{ℝ,<:Stiefel,<:StiefelSubmersionMetric},
    ::Any,
    ::Any,
    ::ProjectionInverseRetraction,
)

function inverse_retract_project!(
    M::MetricManifold{ℝ,<:Stiefel,<:StiefelSubmersionMetric},
    X,
    p,
    q,
)
    log!(get_embedding(M), X, p, q)
    project!(M, X, p, X)
    return X
end

function inverse_retract_shooting!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    X,
    p,
    q,
    method::ShootingInverseRetraction{
        ExponentialRetraction,
        ProjectionInverseRetraction,
        ScaledVectorTransport{ProjectionTransport},
    },
) where {n,k}
    if k > div(n, 2)
        # fall back to default method
        invoke(
            inverse_retract_shooting!,
            Tuple{
                MetricManifold{ℝ,Stiefel{n,k,ℝ}},
                typeof(X),
                typeof(p),
                typeof(q),
                typeof(method),
            },
            M,
            X,
            p,
            q,
            method,
        )
    else
        _inverse_retract_shooting_factors!(M, X, p, q, method)
    end
    return X
end

function _inverse_retract_shooting_factors!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    X,
    p,
    q,
    method::ShootingInverseRetraction{
        ExponentialRetraction,
        ProjectionInverseRetraction,
        ScaledVectorTransport{ProjectionTransport},
    },
) where {n,k}
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

@doc raw"""
    log(M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric}, p, q; kwargs...)

Compute the logarithmic map on the [`Stiefel(n,k)`](@ref) manifold with respect to the [`StiefelSubmersionMetric`](@ref).

The logarithmic map is computed using [`ShootingInverseRetraction`](@ref). For
``k \le \frac{n}{2}``, this is sped up using the ``k``-shooting method of
[^ZimmermanHüper2022]. Keyword arguments are forwarded to `ShootingInverseRetraction`; see
that documentation for details. Their defaults are:
- `num_transport_points=4`
- `tolerance=sqrt(eps())`
- `max_iterations=1_000`
"""
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

# StiefelFactorization and StiefelFactorizationAdjoint
# Note: intended only for internal use

@doc doc"""
    StiefelFactorization{UT,XT} <: AbstractManifoldPoint

Represent points (and vectors) on `Stiefel(n, k)` with ``2k × k`` factors.[^ZimmermanHüper2022]

Given a point ``p ∈ \mathrm{St}(n, k)`` and another matrix ``B ∈ ℝ^{n × k}`` for
``k ≤ \lfloor\frac{n}{2}\rfloor`` the factorization is
````math
\begin{aligned}
B &= UZ\\
U &= \begin{bmatrix}p & Q\end{bmatrix} ∈ \mathrm{St}(n, 2k)\\
Z &= \begin{bmatrix}Z_1 \\ Z_2\end{bmatrix}, \quad Z_1,Z_2 ∈ ℝ^{k × k}.
\end{aligned}
````
If ``B ∈ \mathrm{St}(n, k)``, then ``Z ∈ \mathrm{St}(2k, k)``, and if
``B ∈ T_p \mathrm{St}(n, k)``, then ``Z_1 ∈ \mathrm{SkewHermitianMatrices}(k)``.

``Q`` is determined by choice of a second matrix ``A ∈ ℝ^{n × k}`` with the decomposition
````math
\begin{aligned}
A &= UZ\\
Z_1 &= p^\mathrm{T} A \\
Q Z_2 &= (I - p p^\mathrm{T}) A,
\end{aligned}
````
where here ``Q Z_2`` is the QR decomposition.

This factorization is useful because it is closed under addition, subtraction, scaling,
projection, and the Riemannian exponential and logarithm under the
[`StiefelSubmersionMetric`](@ref). That is, if all matrices involved are factorized to have
the same ``U``, then all of these operations and any algorithm that depends only on them can
be performed in terms of the ``2k × k`` matrices ``Z``. For ``n ≫ k``, this can be much more
efficient than working with the full matrices.
"""
struct StiefelFactorization{UT,ZT} <: AbstractManifoldPoint
    U::UT
    Z::ZT
end
function stiefel_factorization(p, q)
    n, k = size(p)
    k ≤ div(n, 2) || throw(ArgumentError("k must be ≤ div(n, 2)"))
    T = Base.promote_eltype(p, q)
    U = allocate(p, T, Size(n, 2k))
    Z = allocate(p, T, Size(2k, k))
    @views begin
        U1 = U[1:n, 1:k]
        U2 = U[1:n, (k + 1):(2k)]
        Z1 = Z[1:k, 1:k]
        Z2 = Z[(k + 1):(2k), 1:k]
    end
    mul!(Z1, p', q)
    copyto!(U1, q)
    mul!(U1, p, Z1, -1, true)
    Q, N = qr(U1)
    copyto!(Z2, N)
    copyto!(U1, p)
    copyto!(U2, Matrix(Q))
    return StiefelFactorization(U, Z)
end
function Base.eltype(F::StiefelFactorization)
    return Base.promote_eltype(F.U, F.Z)
end
Base.size(F::StiefelFactorization) = (size(F.U, 1), size(F.Z, 2))
function Base.similar(F::StiefelFactorization, ::Type{T}=eltype(F), sz=size(F)) where {T}
    size(F) == sz || throw(DimensionMismatch("size of factorization must be preserved"))
    return StiefelFactorization(F.U, similar(F.Z, T))
end
function Base.copyto!(A::StiefelFactorization, B::StiefelFactorization)
    copyto!(A.Z, B.Z)
    return A
end
function Base.copyto!(A::AbstractMatrix{<:Real}, B::StiefelFactorization)
    mul!(A, B.U, B.Z)
    return A
end
function Base.copyto!(A::StiefelFactorization, B::AbstractMatrix{<:Real})
    mul!(A.Z, A.U', B)
    return A
end
LinearAlgebra.dot(A::StiefelFactorization, B::StiefelFactorization) = dot(A.Z, B.Z)
function LinearAlgebra.mul!(
    A::StiefelFactorization,
    B::StiefelFactorization,
    C::AbstractMatrix{<:Real},
    α::Real,
    β::Real,
)
    mul!(A.Z, B.Z, C, α, β)
    return A
end
function project!(
    ::Stiefel{n,k,ℝ},
    q::StiefelFactorization,
    p::StiefelFactorization,
) where {n,k}
    project!(Stiefel(2k, k), q.Z, p.Z)
    return q
end
function exp!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    q::StiefelFactorization,
    p::StiefelFactorization,
    X::StiefelFactorization,
) where {n,k}
    # TODO:
    # - handle rank-deficient QB
    α = metric(M).α
    @views begin
        ZM = X.Z[1:k, 1:k]
        ZN = X.Z[(k + 1):(2k), 1:k]
        qM = q.Z[1:k, 1:k]
    end
    qM .= ZM .* (α / (α + 1))
    D = exp(qM)
    C = allocate(D, Size(2k, 2k))
    @views begin
        C[1:k, 1:k] .= ZM ./ (α + 1)
        C[1:k, (k + 1):(2k)] .= -ZN'
        C[(k + 1):(2k), 1:k] .= ZN
        fill!(C[(k + 1):(2k), (k + 1):(2k)], false)
        mul!(q.Z, exp(C)[1:(2k), 1:k], D)
    end
    return q
end
function Broadcast.BroadcastStyle(::Type{<:StiefelFactorization})
    return Broadcast.Style{StiefelFactorization}()
end
function Broadcast.BroadcastStyle(
    ::Broadcast.AbstractArrayStyle{0},
    b::Broadcast.Style{<:StiefelFactorization},
)
    return b
end
Broadcast.broadcastable(v::StiefelFactorization) = v
function Base.copyto!(
    dest::StiefelFactorization,
    bc::Broadcast.Broadcasted{Broadcast.Style{StiefelFactorization}},
)
    bc.args isa Tuple{Vararg{<:Union{Manifolds.StiefelFactorization,Real}}} ||
        throw(ArgumentError("Not implemented"))
    bc.f ∈ (identity, *, +, -, /) || throw(ArgumentError("Not implemented"))
    Zargs = map(x -> x isa Manifolds.StiefelFactorization ? x.Z : x, bc.args)
    broadcast!(bc.f, dest.Z, Zargs...)
    return dest
end

struct StiefelFactorizationAdjoint{F<:StiefelFactorization}
    data::F
end
Base.parent(F::StiefelFactorizationAdjoint) = F.data
Base.adjoint(F::StiefelFactorization) = StiefelFactorizationAdjoint(F)
Base.adjoint(F::StiefelFactorizationAdjoint) = parent(F)

function Base.:*(A′::StiefelFactorizationAdjoint, B::StiefelFactorization)
    A = adjoint(A′)
    return A.Z' * B.Z
end
