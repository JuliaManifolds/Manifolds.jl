#
# Default implementation for the matrix type, i.e. as congruence class Stiefel matrices
#
"""
    StiefelPoint <: AbstractManifoldPoint

A point on a [`Stiefel`](@ref) manifold.
This point is mainly used for representing points on the [`Grassmann`](@ref) where this
is also the default representation and hence equivalent to using `AbstractMatrices` thereon.
they can also used be used as points on Stiefel.
"""
struct StiefelPoint{T <: AbstractMatrix} <: AbstractManifoldPoint
    value::T
end

"""
    StiefelTangentVector <: AbstractTangentVector

A tangent vector on the [`Grassmann`](@ref) manifold represented by a tangent vector from
the tangent space of a corresponding point from the [`Stiefel`](@ref) manifold,
see [`StiefelPoint`](@ref).
This is the default representation so is can be used interchangeably with just abstract matrices.
"""
struct StiefelTangentVector{T <: AbstractMatrix} <: AbstractTangentVector
    value::T
end

ManifoldsBase.@manifold_element_forwards StiefelPoint value
ManifoldsBase.@manifold_vector_forwards StiefelTangentVector value
ManifoldsBase.@default_manifold_fallbacks Stiefel StiefelPoint StiefelTangentVector value value
ManifoldsBase.@default_manifold_fallbacks (Stiefel{ℝ}) StiefelPoint StiefelTangentVector value value
ManifoldsBase.@default_manifold_fallbacks Grassmann StiefelPoint StiefelTangentVector value value

function default_vector_transport_method(::Grassmann, ::Type{<:AbstractArray})
    return ParallelTransport()
end
default_vector_transport_method(::Grassmann, ::Type{<:StiefelPoint}) = ParallelTransport()

@doc raw"""
    distance(M::Grassmann, p, q)

Compute the Riemannian distance on [`Grassmann`](@ref) manifold `M```= \mathrm{Gr}(n,k)``.

The distance is given by

````math
d_{\mathrm{Gr}(n,k)}(p,q) = \operatorname{norm}(\log_p(q)).
````
"""
function distance(::Grassmann, p, q)
    z = p' * q
    S = svd(q / z - p).S
    return norm(map(atan, S))
end

embed(::Grassmann, p) = p
embed(::Grassmann, p, X) = X
embed!(::Grassmann, q, p) = copyto!(q, p)
embed!(::Grassmann, Y, p, X) = copyto!(Y, X)
embed!(::Grassmann, q, p::StiefelPoint) = copyto!(q, p.value)
embed!(::Grassmann, Y, p::StiefelPoint, X::StiefelTangentVector) = copyto!(Y, X.value)
embed(::Grassmann, p::StiefelPoint) = p.value
embed(::Grassmann, p::StiefelPoint, X::StiefelTangentVector) = X.value
embed!(::Stiefel, q, p::StiefelPoint) = copyto!(q, p.value)
embed!(::Stiefel, Y, p::StiefelPoint, X::StiefelTangentVector) = copyto!(Y, X.value)
embed(::Stiefel, p::StiefelPoint) = p.value
embed(::Stiefel, p::StiefelPoint, X::StiefelTangentVector) = X.value

@doc raw"""
    exp(M::Grassmann, p, X)

Compute the exponential map on the [`Grassmann`](@ref) `M` ``= \mathrm{Gr}(n,k)`` starting in
`p` with tangent vector (direction) `X`. Let ``X = USV`` denote the SVD decomposition of ``X``.
Then the exponential map is written using

````math
z = p V\cos(S)V^\mathrm{H} + U\sin(S)V^\mathrm{H},
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transposed or Hermitian and the
cosine and sine are applied element wise to the diagonal entries of ``S``. A final QR
decomposition ``z=QR`` is performed for numerical stability reasons, yielding the result as

````math
\exp_p X = Q.
````
"""
exp(::Grassmann, ::Any, ::Any)

function exp!(M::Grassmann, q, p, X)
    norm(M, p, X) ≈ 0 && return copyto!(q, p)
    d = svd(X)
    z = (p * (d.V .* cos.(d.S')) + d.U .* sin.(d.S')) * d.Vt
    return copyto!(q, Array(qr(z).Q))
end


@doc raw"""
    get_coordinates(M::Grasmmann{ℝ}, p, X, B::DefaultOrthonormalBasis)

Given a point `p` on the [`Grassmann`](@ref) manifold `M` in Stiefel representation,
i.e. ``p ∈ ℝ^{n×k}`` compute the coordinates ``c ∈ ℝ^{k(n-k)}`` representing the
tangent vector `X` with respect to the [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`).

The tangent space is characterized by ``p^{\mathrm{T}}X = 0_{k×k}``,
where ``0_{k×k}`` is the ``k×k`` matrix containing zeros
A default orthonormal basis can is constructed as follows: Since ``p`` has full
column rank ``k``, the null space of ``p^{\mathrm{T}}`` is of dimension ``n-k``.
Let ``v_1,…v_{n-k}`` be an orthonormal basis of that null space, then
```math
X_1 = \bigl(v_1, 0_k,…0_k), X_2 = \bigl(v_2 0_k,…,0_k), …, X_{n-k} = \bigl(v_{n-k} 0_k,…0_k),
X_{n-k1} = \bigl(0_k, v_1, 0_k, …, 0_k), …, X_{(n-k)k} = (0_k,\ldots,0_k,v_{n-k}),
```
where ``0_k`` denotes the ``k``-dimensional zero vector.
Let ``V = (v_1,…,v_{n-k}) ∈ ℝ^{n×(n-k)}`` be the matrix of the basis vectors

For a tangent vector ``X`` we know that every column `X_i``, ``i=1,…,k`` lies in
the span of ``v_1,…v_{n-k}`` by looking at every column of the matrix equation
``p^{\mathrm{T}}X = 0_{k×k}``.

Hence we can compute the coordinates ``c`` in ``k`` “blocks” ``C_1,…C_k ∈ ℝ^{n-k}``
by solving the linear systems

```math
VC_i = X_i, \qquad i=1,…,k.
```
"""
get_coordinates(::Grassmann{ℝ}, p, c, ::DefaultOrthonormalBasis)

function get_coordinates_orthonormal!(M::Grassmann{ℝ}, c, p, X, ::RealNumbers)
    n, k = get_parameter(M.size)
    V = nullspace(p') # from SVD, so we have (n-k) ON columns in R^n
    c .= vec(V \ X)
    return c
end

function get_embedding(::Grassmann{𝔽, TypeParameter{Tuple{n, k}}}) where {n, k, 𝔽}
    return Stiefel(n, k, 𝔽)
end
function get_embedding(M::Grassmann{𝔽, Tuple{Int, Int}}) where {𝔽}
    n, k = get_parameter(M.size)
    return Stiefel(n, k, 𝔽; parameter = :field)
end

function ManifoldsBase.get_embedding_type(::Grassmann)
    return ManifoldsBase.IsometricallyEmbeddedManifoldType()
end

function ManifoldsBase.get_forwarding_type(::Grassmann, f, ::Type{<:StiefelPoint})
    return ManifoldsBase.EmbeddedForwardingType()
end
function ManifoldsBase.get_forwarding_type(::Stiefel, f, ::Type{<:StiefelPoint})
    return ManifoldsBase.EmbeddedForwardingType()
end

@doc raw"""
    get_vector(M::Grasmmann{ℝ}, p, c, B::DefaultOrthonormalBasis)

Given a point `p` on the [`Grassmann`](@ref) manifold `M` in Stiefel representation,
i.e. ``p ∈ ℝ^{n×k}`` reconstruct a tangent vector with respect to the [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`)
given coefficients ``c ∈ ℝ^{k(n-k)}``.

The tangent space is characterized by ``p^{\mathrm{T}}X = 0_{k×k}``,
where ``0_{k×k}`` is the ``k×k`` matrix containing zeros
A default orthonormal basis can is constructed as follows: Since ``p`` has full
column rank ``k``, the null space of ``p^{\mathrm{T}}`` is of dimension ``n-k``.
Let ``v_1,…v_{n-k}`` be an orthonormal basis of that null space, then
```math
X_1 = \bigl(v_1, 0_k,…0_k), X_2 = \bigl(v_2 0_k,…,0_k), …, X_{n-k} = \bigl(v_{n-k} 0_k,…0_k),
X_{n-k1} = \bigl(0_k, v_1, 0_k, …, 0_k), …, X_{(n-k)k} = (0_k,\ldots,0_k,v_{n-k}),
```
where ``0_k`` denotes the ``k``-dimensional zero vector.
Let ``V = (v_1,…,v_{n-k}) ∈ ℝ^{n×(n-k)}`` be the matrix of the basis vectors.

Reconstructing a tangent vector ``X``` from a vector of coefficients ``c_1,…,c_{k(n-k)}``
can be done based on the ONB ``v_1,…v_{n-k}``. The ``i``th column ``C_i`` of ``X``
is given by using “blocks” ``C_1,…C_k ∈ ℝ^{n-k}`` of the coordinates and compute

```math
X_i = VC_i = \sum_{j=1}^{n-k} c_{i(n-k) + j}v_j,\qquad i=1,…k
```
"""
get_vector(::Grassmann{ℝ}, p, c, ::DefaultOrthonormalBasis)

function get_vector_orthonormal!(M::Grassmann{ℝ}, X, p, c, ::RealNumbers)
    n, k = get_parameter(M.size)
    V = nullspace(p') # from SVD, so we have (n-k) ON columns in R^n
    mul!(X, V, reshape(c, n - k, k))
    return X
end
@doc raw"""
    inner(M::Grassmann, p, X, Y)

Compute the inner product for two tangent vectors `X`, `Y` from the tangent space
of `p` on the [`Grassmann`](@ref) manifold `M`. The formula reads

````math
g_p(X,Y) = \operatorname{tr}(X^{\mathrm{H}}Y),
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transposed or Hermitian.
"""
inner(::Grassmann, p, X, Y) = dot(X, Y)

@doc raw"""
    inverse_retract(M::Grassmann, p, q, ::PolarInverseRetraction)

Compute the inverse retraction for the [`PolarRetraction`](@extref `ManifoldsBase.PolarRetraction`), on the
[`Grassmann`](@ref) manifold `M`, i.e.,

````math
\operatorname{retr}_p^{-1}q = q*(p^\mathrm{H}q)^{-1} - p,
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transposed or Hermitian.
"""
inverse_retract(::Grassmann, ::Any, ::Any, ::PolarInverseRetraction)

function inverse_retract_polar!(::Grassmann, X, p, q)
    X .= q / (p' * q) .- p
    return X
end

@doc raw"""
    inverse_retract(M, p, q, ::QRInverseRetraction)

Compute the inverse retraction for the [`QRRetraction`](@extref `ManifoldsBase.QRRetraction`), on the
[`Grassmann`](@ref) manifold `M`, i.e.,

````math
\operatorname{retr}_p^{-1}q = q(p^\mathrm{H}q)^{-1} - p,
````
where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transposed or Hermitian.
"""
inverse_retract(::Grassmann, ::Any, ::Any, ::QRInverseRetraction)

function inverse_retract_qr!(::Grassmann, X, p, q)
    X .= q / (p' * q) .- p
    return X
end

@doc raw"""
    log(M::Grassmann, p, q)

Compute the logarithmic map on the [`Grassmann`](@ref) `M` `` = \mathcal M=\mathrm{Gr}(n,k)``,
i.e. the tangent vector `X` whose corresponding [`geodesic`](@extref `ManifoldsBase.geodesic-Tuple{AbstractManifold, Any, Any}`) starting from `p`
reaches `q` after time 1 on `M`. The formula reads

````math
\log_p q = V⋅ \operatorname{atan}(S) ⋅ U^\mathrm{H},
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transposed or Hermitian.
The matrices ``U`` and ``V`` are the unitary matrices, and ``S`` is the diagonal matrix
containing the singular values of the SVD-decomposition

````math
USV = (q^\mathrm{H}p)^{-1} ( q^\mathrm{H} - q^\mathrm{H}pp^\mathrm{H}).
````

In this formula the ``\operatorname{atan}`` is meant elementwise.
"""
log(::Grassmann, ::Any...)

function log!(M::Grassmann, X, p, q)
    inverse_retract_polar!(M, X, p, q)
    d = svd(X)
    mul!(X, d.U, atan.(d.S) .* d.Vt)
    return X
end

@doc raw"""
    parallel_transport_direction(M::Grassmann, p, X, Y)

Compute the parallel transport of ``X \in   T_p\mathcal M`` along the
geodesic starting in direction ``\dot γ (0) = Y``.

 Let ``Y = USV`` denote the SVD decomposition of ``Y``.
Then the parallel transport is given by the formula according to Equation (8.5) (p. 171) [AbsilMahonySepulchre:2008](@cite) as

```math
\mathcal P_{p,Y} X = -pV \sin(S)U^{\mathrm{T}}X + U\cos(S)U^{\mathrm{T}}X + (I-UU^{\mathrm{T}})X
```

where the sine and cosine applied to the diagonal matrix ``S`` are meant to be elementwise
"""
parallel_transport_direction(M::Grassmann, p, X, Y)

# Hook into default since here we have direction first
function parallel_transport_direction(M::Grassmann, p, X, Y)
    Z = zero_vector(M, exp(M, p, X))
    return parallel_transport_direction!(M, Z, p, X, Y)
end

function parallel_transport_direction!(M::Grassmann, Z, p, X, Y)
    d = svd(Y)
    return copyto!(
        M, Z, p,
        (-p * d.V .* sin.(d.S') + d.U .* cos.(d.S')) * (d.U' * X) + (I - d.U * d.U') * X,
    )
end

@doc raw"""
    parallel_transport_to(M::Grassmann, p, X, q)

Compute the parallel transport of ``X ∈  T_p\mathcal M`` along the
geodesic connecting ``p`` to ``q``.

This method uses the [logarithmic map](@ref log(::Grassmann, ::Any...)) and the [parallel transport in that direction](@ref parallel_transport_direction(M::Grassmann, p, X, Y)).
"""
parallel_transport_to(M::Grassmann, p, X, q)

function parallel_transport_to!(M::Grassmann, Z, p, X, q)
    Y = log(M, p, q)
    return parallel_transport_direction!(M, Z, p, X, Y)
end

@doc raw"""
    project(M::Grassmann, p)

Project `p` from the embedding onto the [`Grassmann`](@ref) `M`, i.e. compute `q`
as the polar decomposition of ``p`` such that ``q^{\mathrm{H}}q`` is the identity,
where ``⋅^{\mathrm{H}}`` denotes the Hermitian, i.e. complex conjugate transposed.
"""
project(::Grassmann, ::Any)

function project!(::Grassmann, q, p)
    s = svd(p)
    mul!(q, s.U, s.Vt)
    return q
end

@doc raw"""
    project(M::Grassmann, p, X)

Project the `n`-by-`k` `X` onto the tangent space of `p` on the [`Grassmann`](@ref) `M`,
which is computed by

````math
\operatorname{proj_p}(X) = X - pp^{\mathrm{H}}X,
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transposed or Hermitian.
"""
project(::Grassmann, ::Any...)

function project!(::Grassmann, Y, p, X)
    copyto!(Y, X)
    mul!(Y, p, p' * X, -1, 1)
    return Y
end

@doc raw"""
    rand(M::Grassmann; σ::Real=1.0, vector_at=nothing)

When `vector_at` is `nothing`, return a random point `p` on [`Grassmann`](@ref) manifold `M`
by generating a random (Gaussian) matrix with standard deviation `σ` in matching
size, which is orthonormal.

When `vector_at` is not `nothing`, return a (Gaussian) random vector from the tangent space
``T_p\mathrm{Gr}(n,k)`` with mean zero and standard deviation `σ` by projecting a random
Matrix onto the tangent space at `vector_at`.
"""
rand(M::Grassmann; σ::Real = 1.0)

function Random.rand!(
        rng::AbstractRNG, M::Grassmann{𝔽}, pX;
        σ::Real = one(real(eltype(pX))), vector_at = nothing,
    ) where {𝔽}
    if vector_at === nothing
        n, k = get_parameter(M.size)
        V = σ * randn(rng, 𝔽 === ℝ ? Float64 : ComplexF64, (n, k))
        pX .= qr(V).Q[:, 1:k]
    else
        Z = σ * randn(rng, eltype(pX), size(pX))
        project!(M, pX, vector_at, Z)
        pX ./= norm(pX)
    end
    return pX
end

@doc raw"""
    representation_size(M::Grassmann)

Return the representation size or matrix dimension of a point on the [`Grassmann`](@ref)
`M`, i.e. ``(n,k)`` for both the real-valued and the complex value case.
"""
representation_size(M::Grassmann) = get_parameter(M.size)

@doc raw"""
    retract(M::Grassmann, p, X, ::PolarRetraction)

Compute the SVD-based retraction [`PolarRetraction`](@extref `ManifoldsBase.PolarRetraction`) on the
[`Grassmann`](@ref) `M`. With ``USV = p + X`` the retraction reads
````math
\operatorname{retr}_p X = UV^\mathrm{H},
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transposed or Hermitian.
"""
retract(::Grassmann, ::Any, ::Any, ::PolarRetraction)

function ManifoldsBase.retract_polar!(M::Grassmann, q, p, X)
    return ManifoldsBase.retract_polar_fused!(M, q, p, X, one(eltype(p)))
end
function ManifoldsBase.retract_polar_fused!(M::Grassmann, q, p, X, t::Number)
    q .= p .+ t .* X
    project!(M, q, q)
    return q
end

@doc raw"""
    retract(M::Grassmann, p, X, ::QRRetraction )

Compute the QR-based retraction [`QRRetraction`](@extref `ManifoldsBase.QRRetraction`) on the
[`Grassmann`](@ref) `M`. With ``QR = p + X`` the retraction reads
````math
\operatorname{retr}_p X = QD,
````
where D is a ``m×n`` matrix with
````math
D = \operatorname{diag}\left( \operatorname{sgn}\left(R_{ii}+\frac{1}{2}\right)_{i=1}^n \right).
````
"""
retract(::Grassmann, ::Any, ::Any, ::QRRetraction)

function ManifoldsBase.retract_qr!(M::Grassmann, q, p, X)
    return ManifoldsBase.retract_qr_fused!(M, q, p, X, one(eltype(p)))
end
function ManifoldsBase.retract_qr_fused!(::Grassmann, q, p, X, t::Number)
    q .= p .+ t .* X
    qrfac = qr(q)
    d = diag(qrfac.R)
    copyto!(q, qrfac.Q)
    q .*= Diagonal(sign.(transpose(d) .+ 1 // 2))
    return q
end

@doc raw"""
    riemannian_Hessian(M::Grassmann, p, G, H, X)

The Riemannian Hessian can be computed by adopting Eq. (6.6) [Nguyen:2023](@cite),
where we use for the [`EuclideanMetric`](@extref `ManifoldsBase.EuclideanMetric`) ``α_0=α_1=1`` in their formula.
Let ``\nabla f(p)`` denote the Euclidean gradient `G`,
``\nabla^2 f(p)[X]`` the Euclidean Hessian `H`. Then the formula reads

```math
    \operatorname{Hess}f(p)[X]
    =
    \operatorname{proj}_{T_p\mathcal M}\Bigl(
        ∇^2f(p)[X] - X p^{\mathrm{H}}∇f(p)
    \Bigr).
```

Compared to Eq. (5.6) also the metric conversion simplifies to the identity.
"""
riemannian_Hessian(M::Grassmann, p, G, H, X)

function riemannian_Hessian!(M::Grassmann, Y, p, G, H, X)
    project!(M, Y, p, H - X * p' * G)
    return Y
end

@doc raw"""
    riemann_tensor(::Grassmann{<:Any,ℝ}, p, X, Y, Z)

Compute the value of Riemann tensor on the real [`Grassmann`](@ref) manifold.
The formula reads [Rentmeesters:2011](@cite)

```math
R(X,Y)Z = (XY^\mathrm{T} - YX^\mathrm{T})Z + Z(Y^\mathrm{T}X - X^\mathrm{T}Y).
```
"""
riemann_tensor(::Grassmann{ℝ}, p, X, Y, Z)

function riemann_tensor!(::Grassmann{ℝ}, Xresult, p, X, Y, Z)
    XYᵀ = X * Y'
    YXᵀ = XYᵀ'
    YᵀX = Y' * X
    XᵀY = YᵀX'
    Xresult .= (XYᵀ - YXᵀ) * Z .- Z * (YᵀX - XᵀY)
    return Xresult
end

function Base.show(io::IO, ::Grassmann{𝔽, TypeParameter{Tuple{n, k}}}) where {n, k, 𝔽}
    return print(io, "Grassmann($(n), $(k), $(𝔽))")
end
function Base.show(io::IO, M::Grassmann{𝔽, Tuple{Int, Int}}) where {𝔽}
    n, k = get_parameter(M.size)
    return print(io, "Grassmann($(n), $(k), $(𝔽); parameter=:field)")
end
Base.show(io::IO, p::StiefelPoint) = print(io, "StiefelPoint($(p.value))")
Base.show(io::IO, X::StiefelTangentVector) = print(io, "StiefelTangentVector($(X.value))")

# switch order and not dispatch on the _to variant
function vector_transport_direction(M::Grassmann, p, X, Y, ::ParallelTransport)
    return parallel_transport_direction(M, p, X, Y)
end
function vector_transport_direction!(M::Grassmann, Z, p, X, Y, ::ParallelTransport)
    return parallel_transport_direction!(M, Z, p, X, Y)
end

@doc raw"""
    vector_transport_to(M::Grassmann, p, X, q, ::ProjectionTransport)

compute the projection based transport on the [`Grassmann`](@ref) `M` by
interpreting `X` from the tangent space at `p` as a point in the embedding and
projecting it onto the tangent space at q.
"""
vector_transport_to(::Grassmann, ::Any, ::Any, ::Any, ::ProjectionTransport)

@doc raw"""
    zero_vector(M::Grassmann, p)

Return the zero tangent vector from the tangent space at `p` on the [`Grassmann`](@ref) `M`,
which is given by a zero matrix the same size as `p`.
"""
zero_vector(::Grassmann, ::Any...)

zero_vector!(::Grassmann, X, p) = fill!(X, 0)
