@doc raw"""
    StiefelSubmersionMetric{T<:Real} <: RiemannianMetric

The submersion (or normal) metric family on the [`Stiefel`](@ref) manifold.

The family, with a single real parameter ``α>-1``, has two special cases:
- ``α = -\frac{1}{2}``: [`EuclideanMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.EuclideanMetric)
- ``α = 0``: [`CanonicalMetric`](@ref)

The family was described in [HueperMarkinaSilvaLeite:2021](@cite). This implementation follows the
description in [ZimmermannHueper:2022](@cite).

# Constructor

    StiefelSubmersionMetric(α)

Construct the submersion metric on the Stiefel manifold with the parameter ``α``.
"""
struct StiefelSubmersionMetric{T<:Real} <: RiemannianMetric
    α::T
    StiefelSubmersionMetric(α::T) where {T<:Real} = new{T}(α)
end

@doc raw"""
    q = exp(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, <:StiefelSubmersionMetric}, p, X)
    exp!(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, q, <:StiefelSubmersionMetric}, p, X)

Compute the exponential map on the [`Stiefel(n,k)`](@ref) manifold with respect to the
[`StiefelSubmersionMetric`](@ref).

The exponential map is given by
````math
\exp_p X = \operatorname{Exp}\bigl(
    -\frac{2α+1}{α+1} p p^\mathrm{T} X p^\mathrm{T} +
    X p^\mathrm{T} - p X^\mathrm{T}
\bigr) p \operatorname{Exp}\bigl(\frac{\alpha}{\alpha+1} p^\mathrm{T} X\bigr)
````
This implementation is based on [ZimmermannHueper:2022](@cite).

For ``k < \frac{n}{2}`` the exponential is computed more efficiently using
[`StiefelFactorization`](@ref).
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
    if k ≤ div(n, 2)
        Xfact = stiefel_factorization(p, X)
        pfact = similar(Xfact, eltype(p))
        copyto!(pfact, p)
        qfact = similar(Xfact, eltype(q))
        exp!(M, qfact, pfact, Xfact)
        copyto!(q, qfact)
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
    T = typeof(one(Base.promote_eltype(p, X, Y, α)))
    if n == k
        return T(dot(X, Y)) / (2 * (α + 1))
    elseif α == -1 // 2
        return T(dot(X, Y))
    else
        return dot(X, Y) - (T(dot(p'X, p'Y)) * (2α + 1)) / (2 * (α + 1))
    end
end

function inverse_retract_project!(
    M::MetricManifold{ℝ,<:Stiefel,<:StiefelSubmersionMetric},
    X,
    p,
    q,
)
    return inverse_retract_project!(base_manifold(M), X, p, q)
end

@doc doc"""
    inverse_retract(
        M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
        p,
        q,
        method::ShootingInverseRetraction,
    )

Compute the inverse retraction using [`ShootingInverseRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.ShootingInverseRetraction).

In general the retraction is computed using the generic shooting method.

    inverse_retract(
        M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
        p,
        q,
        method::ShootingInverseRetraction{
            ExponentialRetraction,
            ProjectionInverseRetraction,
            <:Union{ProjectionTransport,ScaledVectorTransport{ProjectionTransport}},
        },
    )

Compute the inverse retraction using [`ShootingInverseRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.ShootingInverseRetraction) more efficiently.

For ``k < \frac{n}{2}`` the retraction is computed more efficiently using
[`StiefelFactorization`](@ref).
"""
inverse_retract(
    ::MetricManifold{ℝ,<:Stiefel,<:StiefelSubmersionMetric},
    ::Any,
    ::Any,
    ::ShootingInverseRetraction,
)

function inverse_retract_shooting!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    X::AbstractMatrix,
    p::AbstractMatrix,
    q::AbstractMatrix,
    method::ShootingInverseRetraction{
        ExponentialRetraction,
        ProjectionInverseRetraction,
        <:Union{ProjectionTransport,ScaledVectorTransport{ProjectionTransport}},
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
        qfact = stiefel_factorization(p, q)
        pfact = similar(qfact, eltype(p))
        copyto!(pfact, p)
        Xfact = similar(qfact, eltype(X))
        inverse_retract_shooting!(M, Xfact, pfact, qfact, method)
        copyto!(X, Xfact)
    end
    return X
end

@doc raw"""
    log(M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric}, p, q; kwargs...)

Compute the logarithmic map on the [`Stiefel(n,k)`](@ref) manifold with respect to the [`StiefelSubmersionMetric`](@ref).

The logarithmic map is computed using [`ShootingInverseRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.ShootingInverseRetraction). For
``k ≤ \lfloor\frac{n}{2}\rfloor``, this is sped up using the ``k``-shooting method of [ZimmermannHueper:2022](@cite).
Keyword arguments are forwarded to `ShootingInverseRetraction`; see
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

@doc raw"""
    Y = riemannian_Hessian(M::MetricManifold{ℝ,Stiefel{n,k,ℝ}, StiefelSubmersionMetric},, p, G, H, X)
    riemannian_Hessian!(MetricManifold{ℝ,Stiefel{n,k,ℝ}, StiefelSubmersionMetric},, Y, p, G, H, X)

Compute the Riemannian Hessian ``\operatorname{Hess} f(p)[X]`` given the
Euclidean gradient ``∇ f(\tilde p)`` in `G` and the Euclidean Hessian ``∇^2 f(\tilde p)[\tilde X]`` in `H`,
where ``\tilde p, \tilde X`` are the representations of ``p,X`` in the embedding,.

Here, we adopt Eq. (5.6) [Nguyen:2023](@cite), for the [`CanonicalMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.EuclideanMetric)
``α_0=1, α_1=\frac{1}{2}`` in their formula. The formula reads

```math
    \operatorname{Hess}f(p)[X]
    =
    \operatorname{proj}_{T_p\mathcal M}\Bigl(
        ∇^2f(p)[X] - \frac{1}{2} X \bigl( (∇f(p))^{\mathrm{H}}p + p^{\mathrm{H}}∇f(p)\bigr)
        - \frac{2α+1}{2(α+1)} \bigl( P ∇f(p) p^{\mathrm{H}} + p ∇f(p))^{\mathrm{H}} P)X
    \Bigr),
```
where ``P = I-pp^{\mathrm{H}}``.

Compared to Eq. (5.6) we have that their ``α_0 = 1``and ``\alpha_1 =  \frac{2α+1}{2(α+1)} + 1``.
"""
riemannian_Hessian(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    p,
    G,
    H,
    X,
) where {n,k}

function riemannian_Hessian!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    Y,
    p,
    G,
    H,
    X,
) where {n,k}
    α = metric(M).α
    Gp = symmetrize(G' * p)
    Z = symmetrize((I - p * p') * G * p')
    project!(M, Y, p, H - X * Gp - (2 * α + 1) / (α + 1) * Z * X)
    return Y
end

# StiefelFactorization code
# Note: intended only for internal use

@doc raw"""
    StiefelFactorization{UT,XT} <: AbstractManifoldPoint

Represent points (and vectors) on `Stiefel(n, k)` with ``2k × k`` factors [ZimmermannHueper:2022](@cite).

Given a point ``p ∈ \mathrm{St}(n, k)`` and another matrix ``B ∈ ℝ^{n × k}`` for
``k ≤ \lfloor\frac{n}{2}\rfloor`` the factorization is
````math
\begin{aligned}
B &= UZ\\
U &= \begin{bmatrix}p & Q\end{bmatrix} ∈ \mathrm{St}(n, 2k)\\
Z &= \begin{bmatrix}Z_1 \\ Z_2\end{bmatrix}, \quad Z_1,Z_2 ∈ ℝ^{k × k}.
\end{aligned}
````
If ``B ∈ \mathrm{St}(n, k)``, then ``Z ∈ \mathrm{St}(2k, k)``.
Note that not every matrix ``B`` can be factorized in this way.

For a fixed ``U``, if ``r ∈ \mathrm{St}(n, k)`` has the factor ``Z_r ∈ \mathrm{St}(2k, k)``,
then ``X_r ∈ T_r \mathrm{St}(n, k)`` has the factor
``Z_{X_r} ∈ T_{Z_r} \mathrm{St}(2k, k)``.

``Q`` is determined by choice of a second matrix ``A ∈ ℝ^{n × k}`` with the decomposition
````math
\begin{aligned}
A &= UZ\\
Z_1 &= p^\mathrm{T} A \\
Q Z_2 &= (I - p p^\mathrm{T}) A,
\end{aligned}
````
where here ``Q Z_2`` is the any decomposition that produces ``Q ∈ \mathrm{St}(n, k)``, for
which we choose the QR decomposition.

This factorization is useful because it is closed under addition, subtraction, scaling,
projection, and the Riemannian exponential and logarithm under the
[`StiefelSubmersionMetric`](@ref). That is, if all matrices involved are factorized to have
the same ``U``, then all of these operations and any algorithm that depends only on them can
be performed in terms of the ``2k × k`` matrices ``Z``. For ``n ≫ k``, this can be much more
efficient than working with the full matrices.

!!! warning
    This type is intended strictly for internal use and should not be directly used.
"""
struct StiefelFactorization{UT,ZT} <: AbstractManifoldPoint
    U::UT
    Z::ZT
end
"""
    stiefel_factorization(p, x) -> StiefelFactorization

Compute the [`StiefelFactorization`](@ref) of ``x`` relative to the point ``p``.
"""
function stiefel_factorization(p, x)
    n, k = size(p)
    T = Base.promote_eltype(p, x)
    U = allocate(p, T, Size(n, 2k))
    Z = allocate(p, T, Size(2k, k))
    xfact = StiefelFactorization(U, Z)
    @views begin
        U1 = U[1:n, 1:k]
        U2 = U[1:n, (k + 1):(2k)]
        Z1 = Z[1:k, 1:k]
        Z2 = Z[(k + 1):(2k), 1:k]
    end
    if p ≈ x
        copyto!(U1, p)
        copyto!(U2, qr(U1).Q[1:n, (k + 1):(2k)])
        copyto!(xfact, x)
    else
        copyto!(U1, x)
        mul!(Z1, p', x)
        mul!(U1, p, Z1, -1, true)
        Q, N = qr(U1)
        copyto!(Z2, N)
        copyto!(U1, p)
        copyto!(U2, Matrix(Q))
    end
    return xfact
end
function Base.eltype(F::StiefelFactorization)
    return Base.promote_eltype(F.U, F.Z)
end
Base.size(F::StiefelFactorization) = (size(F.U, 1), size(F.Z, 2))
function Base.similar(F::StiefelFactorization, ::Type{T}=eltype(F), sz=size(F)) where {T}
    size(F) == sz || throw(DimensionMismatch("size of factorization must be preserved"))
    return StiefelFactorization(convert(AbstractArray{T}, F.U), similar(F.Z, T))
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
    bc.args isa Tuple{Vararg{Union{Manifolds.StiefelFactorization,Real}}} ||
        throw(ArgumentError("Not implemented"))
    bc.f ∈ (identity, *, +, -, /) || throw(ArgumentError("Not implemented"))
    Zargs = map(x -> x isa Manifolds.StiefelFactorization ? x.Z : x, bc.args)
    broadcast!(bc.f, dest.Z, Zargs...)
    return dest
end
function project!(
    ::Stiefel{n,k,ℝ},
    q::StiefelFactorization,
    p::StiefelFactorization,
) where {n,k}
    project!(Stiefel(2k, k), q.Z, p.Z)
    return q
end
function project!(
    ::Stiefel{n,k,ℝ},
    Y::StiefelFactorization,
    p::StiefelFactorization,
    X::StiefelFactorization,
) where {n,k}
    project!(Stiefel(2k, k), Y.Z, p.Z, X.Z)
    return Y
end
function inner(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    p::StiefelFactorization,
    X::StiefelFactorization,
    Y::StiefelFactorization,
) where {n,k}
    Msub = MetricManifold(Stiefel(2k, k), metric(M))
    return inner(Msub, p.Z, X.Z, Y.Z)
end
function exp!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    q::StiefelFactorization,
    p::StiefelFactorization,
    X::StiefelFactorization,
) where {n,k}
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
function Base.:*(t::Number, sf::StiefelFactorization)
    return StiefelFactorization(sf.U, t * sf.Z)
end
