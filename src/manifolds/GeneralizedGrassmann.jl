@doc raw"""
    GeneralizedGrassmann{𝔽,T,TB<:AbstractMatrix} <: AbstractDecoratorManifold{𝔽}

The generalized Grassmann manifold ``\operatorname{Gr}(n,k,B)`` consists of all subspaces
spanned by ``k`` linear independent vectors ``𝔽^n``, where ``𝔽  ∈ \{ℝ, ℂ\}`` is either the real- (or complex-) valued vectors.
This yields all ``k``-dimensional subspaces of ``ℝ^n`` for the real-valued case and all ``2k``-dimensional subspaces
of ``ℂ^n`` for the second.

The manifold can be represented as

````math
\operatorname{Gr}(n, k, B) := \bigl\{ \operatorname{span}(p)\ \big|\ p ∈ 𝔽^{n×k}, p^\mathrm{H}Bp = I_k\},
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate (or Hermitian) transpose and
``I_k`` is the ``k×k`` identity matrix. This means, that the columns of ``p``
form an unitary basis of the subspace with respect to the scaled inner product, that is a
point on ``\operatorname{Gr}(n,k,B)``, and hence the subspace can actually be represented by
a whole equivalence class of representers. For ``B=I_n`` this simplifies to the [`Grassmann`](@ref) manifold.

The tangent space at a point (subspace) ``p`` is given by

````math
T_x\mathrm{Gr}(n,k,B) = \bigl\{
X ∈ 𝔽^{n×k} :
X^{\mathrm{H}}Bp + p^{\mathrm{H}}BX = 0_{k} \bigr\},
````

where ``0_{k}`` denotes the ``k×k`` zero matrix.

Note that a point ``p ∈ \operatorname{Gr}(n,k,B)`` might be represented by
different matrices (i.e. matrices with ``B``-unitary column vectors that span
the same subspace). Different representations of ``p`` also lead to different
representation matrices for the tangent space ``T_p\mathrm{Gr}(n,k,B)``

The manifold is named after
[Hermann G. Graßmann](https://en.wikipedia.org/wiki/Hermann_Grassmann) (1809-1877).

# Constructor

    GeneralizedGrassmann(n, k, B=I_n, field=ℝ)

Generate the (real-valued) Generalized Grassmann manifold of ``n×k`` dimensional
orthonormal matrices with scalar product `B`.
"""
struct GeneralizedGrassmann{𝔽, T, TB <: AbstractMatrix} <: AbstractDecoratorManifold{𝔽}
    size::T
    B::TB
end

function GeneralizedGrassmann(
        n::Int,
        k::Int,
        B::AbstractMatrix = Matrix{Float64}(I, n, n),
        𝔽::AbstractNumbers = ℝ;
        parameter::Symbol = :type,
    )
    size = wrap_type_parameter(parameter, (n, k))
    return GeneralizedGrassmann{𝔽, typeof(size), typeof(B)}(size, B)
end

@doc raw"""
    change_representer(M::GeneralizedGrassmann, ::EuclideanMetric, p, X)

Change `X` to the corresponding representer of a cotangent vector at `p` with respect to the scaled metric
of the [`GeneralizedGrassmann`](@ref) `M`, i.e, since

```math
g_p(X,Y) = \operatorname{tr}(Y^{\mathrm{H}}BZ) = \operatorname{tr}(X^{\mathrm{H}}Z) = ⟨X,Z⟩
```

has to hold for all ``Z``, where the repreenter `X` is given, the resulting representer with
respect to the metric on the [`GeneralizedGrassmann`](@ref) is given by ``Y = B^{-1}X``.
"""
change_representer(::GeneralizedGrassmann, ::EuclideanMetric, ::Any, ::Any)

function change_representer!(M::GeneralizedGrassmann, Y, ::EuclideanMetric, p, X)
    return copyto!(M, Y, p, M.B \ X)
end

@doc raw"""
    change_metric(M::GeneralizedGrassmann, ::EuclideanMetric, p X)

Change `X` to the corresponding vector with respect to the metric of the [`GeneralizedGrassmann`](@ref) `M`,
i.e. let ``B=LL'`` be the Cholesky decomposition of the matrix `M.B`, then the corresponding vector is ``L\X``.
"""
change_metric(M::GeneralizedGrassmann, ::EuclideanMetric, ::Any, ::Any)

function change_metric!(M::GeneralizedGrassmann, Y, ::EuclideanMetric, p, X)
    C2 = cholesky(M.B).L
    copyto!(Y, X)
    ldiv!(C2, Y)
    return Y
end

@doc raw"""
    check_point(M::GeneralizedGrassmann, p)

Check whether `p` is representing a point on the [`GeneralizedGrassmann`](@ref) `M`, i.e. its
a `n`-by-`k` matrix of unitary column vectors with respect to the B inner product and
of correct `eltype` with respect to `𝔽`.
"""
function check_point(M::GeneralizedGrassmann, p; kwargs...)
    return nothing # everything already checked in the embedding (generalized Stiefel)
end

@doc raw"""
    check_vector(M::GeneralizedGrassmann, p, X; kwargs...)

Check whether `X` is a tangent vector in the tangent space of `p` on
the [`GeneralizedGrassmann`](@ref) `M`, i.e. that `X` is of size and type as well as that

````math
    p^{\mathrm{H}}BX + \overline{X^{\mathrm{H}}Bp} = 0_k,
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transpose or Hermitian,
``\overline{⋅}`` the (elementwise) complex conjugate, and ``0_k`` denotes the ``k×k`` zero natrix.
"""
function check_vector(M::GeneralizedGrassmann, p, X; kwargs...)
    return nothing # everything already checked in the embedding (generalized Stiefel)
end

@doc raw"""
    distance(M::GeneralizedGrassmann, p, q)

Compute the Riemannian distance on [`GeneralizedGrassmann`](@ref)
manifold `M```= \mathrm{Gr}(n,k,B)``.

The distance is given by
````math
d_{\mathrm{Gr}(n,k,B)}(p,q) = \operatorname{norm}(\log_p(q)).
````
"""
function distance(M::GeneralizedGrassmann, p, q)
    z = p' * M.B' * q
    X = allocate_result(M, log, p, q)
    X .= q / z .- p
    d = svd(X)
    X .= d.U .* atan.(d.S')
    return norm(M, p, X)
end

embed(::GeneralizedGrassmann, p) = p
embed(::GeneralizedGrassmann, p, X) = X

@doc raw"""
    exp(M::GeneralizedGrassmann, p, X)

Compute the exponential map on the [`GeneralizedGrassmann`](@ref) `M` ``= \mathrm{Gr}(n,k,B)``
starting in `p` with tangent vector (direction) `X`. Let ``X^{\mathrm{H}}BX = USV`` denote the
SVD decomposition of ``X^{\mathrm{H}}BX``. Then the exponential map is written using

````math
\exp_p X = p V\cos(S)V^\mathrm{H} + U\sin(S)V^\mathrm{H},
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transposed or Hermitian and the
cosine and sine are applied element wise to the diagonal entries of ``S``.
"""
exp(::GeneralizedGrassmann, ::Any...)

function exp!(M::GeneralizedGrassmann, q, p, X)
    norm(M, p, X) ≈ 0 && return copyto!(q, p)
    d = svd(X' * M.B * X)
    V = d.Vt
    S = abs.(sqrt.(d.S))
    mul!(q, p * (V .* cos.(S')) + X * (V .* usinc.(S')), V)
    project!(M, q, q)
    return q
end

@doc raw"""
    injectivity_radius(M::GeneralizedGrassmann)
    injectivity_radius(M::GeneralizedGrassmann, p)

Return the injectivity radius on the [`GeneralizedGrassmann`](@ref) `M`,
which is ``\frac{π}{2}``.
"""
injectivity_radius(::GeneralizedGrassmann) = π / 2
injectivity_radius(::GeneralizedGrassmann, p) = π / 2
injectivity_radius(::GeneralizedGrassmann, ::AbstractRetractionMethod) = π / 2
injectivity_radius(::GeneralizedGrassmann, p, ::AbstractRetractionMethod) = π / 2

"""
    is_flat(M::GeneralizedGrassmann)

Return true if [`GeneralizedGrassmann`](@ref) `M` is one-dimensional.
"""
is_flat(M::GeneralizedGrassmann) = manifold_dimension(M) == 1

function get_embedding(M::GeneralizedGrassmann{𝔽, TypeParameter{Tuple{n, k}}}) where {n, k, 𝔽}
    return GeneralizedStiefel(n, k, M.B, 𝔽)
end
function get_embedding(M::GeneralizedGrassmann{𝔽, Tuple{Int, Int}}) where {𝔽}
    n, k = get_parameter(M.size)
    return GeneralizedStiefel(n, k, M.B, 𝔽; parameter = :field)
end

function ManifoldsBase.get_embedding_type(::GeneralizedGrassmann)
    return ManifoldsBase.EmbeddedManifoldType()
end

@doc raw"""
    inner(M::GeneralizedGrassmann, p, X, Y)

Compute the inner product for two tangent vectors `X`, `Y` from the tangent space
of `p` on the [`GeneralizedGrassmann`](@ref) manifold `M`. The formula reads

````math
g_p(X,Y) = \operatorname{tr}(X^{\mathrm{H}}BY),
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transposed or Hermitian.
"""
inner(M::GeneralizedGrassmann, p, X, Y) = dot(X, M.B, Y)

function _isapprox(M::GeneralizedGrassmann, p, X, Y; atol = sqrt(max_eps(X, Y)), kwargs...)
    return isapprox(norm(M, p, X - Y), 0; atol = atol, kwargs...)
end
function _isapprox(M::GeneralizedGrassmann, p, q; atol = sqrt(max_eps(p, q)), kwargs...)
    return isapprox(distance(M, p, q), 0; atol = atol, kwargs...)
end

@doc raw"""
    log(M::GeneralizedGrassmann, p, q)

Compute the logarithmic map on the [`GeneralizedGrassmann`](@ref) `M` `` = \mathcal M=\mathrm{Gr}(n,k,B)``,
i.e. the tangent vector `X` whose corresponding [`geodesic`](@extref `ManifoldsBase.geodesic-Tuple{AbstractManifold, Any, Any}`) starting from `p`
reaches `q` after time 1 on `M`. The formula reads

````math
\log_p q = V⋅ \operatorname{atan}(S) ⋅ U^\mathrm{H},
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transposed or Hermitian.
The matrices ``U`` and ``V`` are the unitary matrices, and ``S`` is the diagonal matrix
containing the singular values of the SVD-decomposition

````math
USV = (q^\mathrm{H}Bp)^{-1} ( q^\mathrm{H} - q^\mathrm{H}Bpp^\mathrm{H}).
````

In this formula the ``\operatorname{atan}`` is meant elementwise.
"""
log(::GeneralizedGrassmann, ::Any...)

function log!(M::GeneralizedGrassmann, X, p, q)
    z = p' * M.B' * q
    X .= q / z .- p
    d = svd(X)
    return mul!(X, d.U, atan.(d.S) .* d.Vt)
end

@doc raw"""
    manifold_dimension(M::GeneralizedGrassmann)

Return the dimension of the [`GeneralizedGrassmann(n,k,𝔽)`](@ref) manifold `M`, i.e.

````math
\dim \operatorname{Gr}(n,k,B) = k(n-k) \dim_ℝ 𝔽,
````

where ``\dim_ℝ 𝔽`` is the [`real_dimension`](@extref `ManifoldsBase.real_dimension-Tuple{ManifoldsBase.AbstractNumbers}`) of `𝔽`.
"""
function manifold_dimension(M::GeneralizedGrassmann{𝔽}) where {𝔽}
    n, k = get_parameter(M.size)
    return k * (n - k) * real_dimension(𝔽)
end

"""
    mean(
        M::GeneralizedGrassmann,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/4);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@extref `ManifoldsBase.GeodesicInterpolationWithinRadius`).
"""
mean(::GeneralizedGrassmann, ::Any...)

function default_approximation_method(::GeneralizedGrassmann, ::typeof(mean))
    return GeodesicInterpolationWithinRadius(π / 4)
end

@doc raw"""
    project(M::GeneralizedGrassmann, p)

Project `p` from the embedding onto the [`GeneralizedGrassmann`](@ref) `M`, i.e. compute `q`
as the polar decomposition of ``p`` such that ``q^{\mathrm{H}}Bq`` is the identity,
where ``⋅^{\mathrm{H}}`` denotes the Hermitian, i.e. complex conjugate transpose.
"""
project(::GeneralizedGrassmann, ::Any)

function project!(M::GeneralizedGrassmann, q, p)
    s = svd(p)
    e = eigen(s.U' * M.B * s.U)
    qsinv = e.vectors ./ sqrt.(transpose(e.values))
    q .= s.U * qsinv * e.vectors' * s.V'
    return q
end

@doc raw"""
    project(M::GeneralizedGrassmann, p, X)

Project the `n`-by-`k` `X` onto the tangent space of `p` on the
[`GeneralizedGrassmann`](@ref) `M`, which is computed by

````math
\operatorname{proj_p}(X) = X - pp^{\mathrm{H}}B^\mathrm{T}X,
````

where ``⋅^{\mathrm{H}}`` denotes the complex conjugate transposed or Hermitian
and ``⋅^{\mathrm{T}}`` the transpose.
"""
project(::GeneralizedGrassmann, ::Any, ::Any)

function project!(M::GeneralizedGrassmann, Y, p, X)
    A = p' * M.B' * X
    copyto!(Y, X)
    mul!(Y, p, Hermitian((A .+ A') ./ 2), -1, 1)
    return Y
end

@doc raw"""
    rand(::GeneralizedGrassmann; vector_at=nothing, σ::Real=1.0)

When `vector_at` is `nothing`, return a random (Gaussian) point `p` on the [`GeneralizedGrassmann`](@ref)
manifold `M` by generating a (Gaussian) matrix with standard deviation `σ` and return the
(generalized) orthogonalized version, i.e. return the projection onto the manifold of the
Q component of the QR decomposition of the random matrix of size ``n×k``.

When `vector_at` is not `nothing`, return a (Gaussian) random vector from the tangent space
``T_{vector\_at}\mathrm{St}(n,k)`` with mean zero and standard deviation `σ` by projecting a
random Matrix onto the tangent vector at `vector_at`.
"""
rand(::GeneralizedGrassmann; σ::Real = 1.0)

function Random.rand!(
        rng::AbstractRNG,
        M::GeneralizedGrassmann{ℝ},
        pX;
        vector_at = nothing,
        σ::Real = one(real(eltype(pX))),
    )
    n, k = get_parameter(M.size)
    if vector_at === nothing
        A = σ * randn(rng, eltype(pX), n, k)
        project!(M, pX, Matrix(qr(A).Q))
    else
        Z = σ * randn(rng, eltype(pX), size(pX))
        project!(M, pX, vector_at, Z)
        normalize!(pX)
    end
    return pX
end

@doc raw"""
    representation_size(M::GeneralizedGrassmann)

Return the representation size or matrix dimension of a point on the [`GeneralizedGrassmann`](@ref)
`M`, i.e. ``(n,k)`` for both the real-valued and the complex value case.
"""
representation_size(M::GeneralizedGrassmann) = get_parameter(M.size)

@doc raw"""
    retract(M::GeneralizedGrassmann, p, X, ::PolarRetraction)

Compute the SVD-based retraction [`PolarRetraction`](@extref `ManifoldsBase.PolarRetraction`) on the
[`GeneralizedGrassmann`](@ref) `M`, by
[`project`](@ref project(M::GeneralizedGrassmann, p))ing ``p + X`` onto `M`.
"""
retract(::GeneralizedGrassmann, ::Any, ::Any, ::PolarRetraction)

function ManifoldsBase.retract_polar!(M::GeneralizedGrassmann, q, p, X)
    return ManifoldsBase.retract_polar_fused!(M, q, p, X, one(eltype(p)))
end
function ManifoldsBase.retract_polar_fused!(M::GeneralizedGrassmann, q, p, X, t::Number)
    q .= p .+ t .* X
    project!(M, q, q)
    return q
end

function ManifoldsBase.retract_project!(M::GeneralizedGrassmann, q, p, X)
    return ManifoldsBase.retract_project_fused!(M, q, p, X, one(eltype(p)))
end
function ManifoldsBase.retract_project_fused!(M::GeneralizedGrassmann, q, p, X, t::Number)
    q .= p .+ t .* X
    project!(M, q, q)
    return q
end

function Base.show(
        io::IO,
        M::GeneralizedGrassmann{𝔽, TypeParameter{Tuple{n, k}}},
    ) where {n, k, 𝔽}
    return print(io, "GeneralizedGrassmann($(n), $(k), $(M.B), $(𝔽))")
end
function Base.show(io::IO, M::GeneralizedGrassmann{𝔽, Tuple{Int, Int}}) where {𝔽}
    n, k = get_parameter(M.size)
    return print(io, "GeneralizedGrassmann($(n), $(k), $(M.B), $(𝔽); parameter=:field)")
end

@doc raw"""
    zero_vector(M::GeneralizedGrassmann, p)

Return the zero tangent vector from the tangent space at `p` on the
[`GeneralizedGrassmann`](@ref) `M`, which is given by a zero matrix the same size as `p`.
"""
zero_vector(::GeneralizedGrassmann, ::Any...)

zero_vector!(::GeneralizedGrassmann, X, p) = fill!(X, 0)
