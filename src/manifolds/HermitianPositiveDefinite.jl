@doc raw"""
    HermitianPositiveDefinite{𝔽,T} <: AbstractDecoratorManifold{𝔽}

The manifold of Hermitian positive definite matrices

````math
\mathcal H(n) :=
\bigl\{
p ∈ 𝔽^{n×n}\ \big|\ p = p^{\mathrm{H}} \text{ and } a^{\mathrm{H}}pa > 0 \text{ for all } a ∈ 𝔽^{n}\backslash\{0\}
\bigr\}
````

over the field ``𝔽 ∈ \{ℝ, ℂ\}``, where usually ``𝔽=ℂ``.
For the case ``𝔽=ℝ`` this manifold is the manifold of symmetric positive definite matrices,
see [`SymmetricPositiveDefinite`](@ref).

The tangent space at ``p∈\mathcal H(n)`` reads

```math
    T_p\mathcal H(n) =
    \bigl\{
        X \in 𝔽^{n×n} \big|\ X=X^\mathrm{H}
    \bigr\},
```
i.e. the set of Hermitian matrices, which are the symmetric matrices for ``𝔽=ℝ``.

The default metric is the [`AffineInvariantMetric`](@ref).

# Constructor

    HermitianPositiveDefinite(n, 𝔽=ℂ; parameter::Symbol=:type)

generates the manifold of Hermitian positive definite matrices ``\mathcal H(n) \subset 𝔽^{n×n}``.
"""
struct HermitianPositiveDefinite{𝔽, T} <: AbstractDecoratorManifold{𝔽}
    size::T
end
function HermitianPositiveDefinite(
        n::Int, 𝔽::Union{RealNumbers, ComplexNumbers} = ℂ; parameter::Symbol = :type
    )
    n < 0 && throw(
        DomainError(n, "The matrix size n of Hermitian positive definite matrices must be nonnegative, but was $(n).")
    )
    size = wrap_type_parameter(parameter, (n,))
    return HermitianPositiveDefinite{𝔽, typeof(size)}(size)
end

@doc raw"""
    MatrixSqrtManifoldPoint{P,Q,R,E} <: AbstractManifoldPoint

A point represented by a matrix ``p`` together with its eigenvalue decomposition.

The matrix ``p ∈ 𝔽^{n×n}`` is over a field ``𝔽 ∈ \{ℝ,ℂ\}``; besides its eigenvalue decomposition
the point optionally stores ``p`` itself, its matrix square root ``p^{\frac{1}{2}}`` and the inverse
``p^{-\frac{1}{2}}`` of the square root, to avoid their repeated computation.

This is for example used on the [`HermitianPositiveDefinite`](@ref) manifold.

# Fields

* `p::P` the matrix representation of the point or `missing`
* `eigen::E` the eigenvalue decomposition `eigen(Hermitian(p))` of `p`
* `sqrt::Q` the matrix square root of `p` or `missing`
* `sqrt_inv::R` the inverse of the matrix square root of `p` or `missing`

Only the field `eigen` is always stored. Each of the other three fields can be set to `missing`
(its field type is then `Missing`) to indicate that it is not stored; it is then computed from
`eigen` when required, but still not stored.

# Constructor

    MatrixSqrtManifoldPoint(p::AbstractMatrix; store_p=true, store_sqrt=true, store_sqrt_inv=true)

Create a `MatrixSqrtManifoldPoint` from the Hermitian positive definite matrix `p`, where you can
optionally store `p`, `sqrt` and `sqrt_inv`.
"""
struct MatrixSqrtManifoldPoint{
        P <: Union{AbstractMatrix, Missing},
        Q <: Union{AbstractMatrix, Missing},
        R <: Union{AbstractMatrix, Missing},
        E <: Eigen,
    } <: AbstractManifoldPoint
    p::P
    eigen::E
    sqrt::Q
    sqrt_inv::R
end

ManifoldsBase.@fieldwise_equality MatrixSqrtManifoldPoint (eigen,)

MatrixSqrtManifoldPoint(p::MatrixSqrtManifoldPoint) = p
function MatrixSqrtManifoldPoint(
        p::AbstractMatrix; store_p = true, store_sqrt = true, store_sqrt_inv = true
    )
    e = eigen(Hermitian(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
    if store_sqrt
        s_sqrt = Diagonal(sqrt.(S))
        p_sqrt = U * s_sqrt * U'
    else
        p_sqrt = missing
    end
    if store_sqrt_inv
        s_sqrt_inv = Diagonal(1 ./ sqrt.(S))
        p_sqrt_inv = U * s_sqrt_inv * U'
    else
        p_sqrt_inv = missing
    end
    if store_p
        q = p
    else
        q = missing
    end
    return MatrixSqrtManifoldPoint{typeof(q), typeof(p_sqrt), typeof(p_sqrt_inv), typeof(e)}(
        q, e, p_sqrt, p_sqrt_inv
    )
end
convert(::Type{MatrixSqrtManifoldPoint}, p::AbstractMatrix) = MatrixSqrtManifoldPoint(p)

function allocate(p::MatrixSqrtManifoldPoint)
    return MatrixSqrtManifoldPoint(
        ismissing(p.p) ? missing : allocate(p.p),
        Eigen(allocate(p.eigen.values), allocate(p.eigen.vectors)),
        ismissing(p.sqrt) ? missing : allocate(p.sqrt),
        ismissing(p.sqrt_inv) ? missing : allocate(p.sqrt_inv),
    )
end
function allocate(p::MatrixSqrtManifoldPoint, ::Type{T}) where {T}
    return MatrixSqrtManifoldPoint(
        ismissing(p.p) ? missing : allocate(p.p, T),
        # eigenvalues of a Hermitian matrix are real, also for a complex `T`
        Eigen(allocate(p.eigen.values, real(T)), allocate(p.eigen.vectors, T)),
        ismissing(p.sqrt) ? missing : allocate(p.sqrt, T),
        ismissing(p.sqrt_inv) ? missing : allocate(p.sqrt_inv, T),
    )
end

function allocate_result(
        M::HermitianPositiveDefinite, ::typeof(zero_vector), p::MatrixSqrtManifoldPoint
    )
    return allocate_result(M, zero_vector, convert(AbstractMatrix, p))
end
function allocate_coordinates(M::HermitianPositiveDefinite, p::MatrixSqrtManifoldPoint, T, n::Int)
    return allocate_coordinates(M, convert(AbstractMatrix, p), T, n)
end
function allocate_result(
        M::HermitianPositiveDefinite, ::typeof(get_vector), p::MatrixSqrtManifoldPoint, c
    )
    return allocate_result(M, get_vector, convert(AbstractMatrix, p), c)
end

function allocation_promotion_function(
        ::HermitianPositiveDefinite{ℂ}, ::Union{typeof(get_vector), typeof(rand)}, ::Tuple
    )
    return complex
end

@doc raw"""
    check_point(M::HermitianPositiveDefinite, p; kwargs...)

checks, whether `p` is a valid point on the [`HermitianPositiveDefinite`](@ref) `M`,
i.e. is a matrix ``p ∈ 𝔽^{n×n}`` over the field ``𝔽 ∈ \{ℝ,ℂ\}`` of `M`, is Hermitian (``p^{\mathrm{H}} = p``,
which is symmetric, ``p^{\mathrm{T}} = p``, for ``𝔽=ℝ``)
and positive definite, that is ``a^{\mathrm{H}}pa > 0`` for all ``a ∈ 𝔽^{n}\backslash\{0\}``.
The tolerance for the second to last test can be set using the `kwargs...`.
"""
function check_point(M::HermitianPositiveDefinite, p; kwargs...)
    if !isapprox(p, p'; kwargs...)
        return DomainError(
            norm(p - p'), "The point $(p) does not lie on $(M) since it is not a Hermitian matrix.",
        )
    end
    if !isposdef(Hermitian(p))
        return DomainError(
            eigvals(p), "The point $p does not lie on $(M) since it is not a positive definite matrix.",
        )
    end
    return nothing
end
function check_point(M::HermitianPositiveDefinite, p::MatrixSqrtManifoldPoint; kwargs...)
    return check_point(M, convert(AbstractMatrix, p); kwargs...)
end

@doc raw"""
    check_vector(M::HermitianPositiveDefinite, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`HermitianPositiveDefinite`](@ref) `M`.
After [`check_point`](@ref)`(M,p)`, `X` has to be of the same dimension as `p` and a Hermitian
matrix ``X ∈ 𝔽^{n×n}``, ``X^{\mathrm{H}} = X`` (a symmetric matrix, ``X^{\mathrm{T}} = X``, for ``𝔽=ℝ``),
i.e. this stores tangent vectors as elements of the corresponding Lie group.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_vector(M::HermitianPositiveDefinite, p, X; kwargs...)
    if !isapprox(X, X'; kwargs...)
        return DomainError(
            X,
            "The vector $(X) is not a tangent to a point on $(M) (represented as an element of the Lie algebra) since it is not Hermitian.",
        )
    end
    return nothing
end
function check_vector(M::HermitianPositiveDefinite, p::MatrixSqrtManifoldPoint, X; kwargs...)
    return check_vector(M, convert(AbstractMatrix, p), X; kwargs...)
end

function check_size(M::HermitianPositiveDefinite, p::MatrixSqrtManifoldPoint; kwargs...)
    return check_size(M, convert(AbstractMatrix, p))
end
function check_size(M::HermitianPositiveDefinite, p::MatrixSqrtManifoldPoint, X; kwargs...)
    return check_size(M, convert(AbstractMatrix, p), X)
end

function Base.copy(p::MatrixSqrtManifoldPoint)
    return MatrixSqrtManifoldPoint(
        ismissing(p.p) ? missing : copy(p.p),
        Eigen(copy(p.eigen.values), copy(p.eigen.vectors)),
        ismissing(p.sqrt) ? missing : copy(p.sqrt),
        ismissing(p.sqrt_inv) ? missing : copy(p.sqrt_inv),
    )
end

#
# Lazy copyto, only copy if both are not missing,
# create from `p` if it is a nonmissing field in q.
#
function copyto!(q::MatrixSqrtManifoldPoint, p::MatrixSqrtManifoldPoint)
    if !ismissing(q.p) # we have to fill the Fields
        if !ismissing(p.p)
            !ismissing(q.p) && copyto!(q.p, p.p)
        else # otherwise compute and copy
            copyto!(q.p, convert(AbstractMatrix, p))
        end
    end
    copyto!(q.eigen.values, p.eigen.values)
    copyto!(q.eigen.vectors, p.eigen.vectors)
    if !ismissing(q.sqrt)
        if !ismissing(p.sqrt)
            copyto!(q.sqrt, p.sqrt)
        else # otherwise compute and copy
            copyto!(q.sqrt, hpd_sqrt(p))
        end
    end
    if !ismissing(q.sqrt_inv)
        if !ismissing(p.sqrt_inv)
            copyto!(q.sqrt_inv, p.sqrt_inv)
        else # otherwise compute and copy
            copyto!(q.sqrt_inv, hpd_sqrt_inv(p))
        end
    end
    return q
end

metric(::HermitianPositiveDefinite) = AffineInvariantMetric()

embed(::HermitianPositiveDefinite, p) = p
embed(::HermitianPositiveDefinite, p::MatrixSqrtManifoldPoint) = convert(AbstractMatrix, p)
embed(::HermitianPositiveDefinite, p, X) = X

function get_embedding(::HermitianPositiveDefinite{𝔽, TypeParameter{Tuple{n}}}) where {𝔽, n}
    return Euclidean(n, n; field = 𝔽)
end
function get_embedding(M::HermitianPositiveDefinite{𝔽, Tuple{Int}}) where {𝔽}
    n = get_parameter(M.size)[1]
    return Euclidean(n, n; field = 𝔽, parameter = :field)
end

function ManifoldsBase.get_embedding_type(
        ::HermitianPositiveDefinite, ::Type{<:MatrixSqrtManifoldPoint}
    )
    return ManifoldsBase.EmbeddedManifoldType(ManifoldsBase.IndirectEmbedding())
end
function ManifoldsBase.get_embedding_type(::HermitianPositiveDefinite)
    return ManifoldsBase.EmbeddedManifoldType()
end

# Internal function for nicer printing.
get_parameter_type(::HermitianPositiveDefinite{𝔽, <:TypeParameter}) where {𝔽} = :type
get_parameter_type(::HermitianPositiveDefinite{𝔽, Tuple{Int}}) where {𝔽} = :field

@doc raw"""
    injectivity_radius(M::HermitianPositiveDefinite[, p])
    injectivity_radius(M::MetricManifold{HermitianPositiveDefinite,AffineInvariantMetric}[, p])

Return the injectivity radius of the [`HermitianPositiveDefinite`](@ref).
Since `M` is a Hadamard manifold with respect to the [`AffineInvariantMetric`](@ref),
see [Smith:2005](@cite), Section II-B and Section III-C, the injectivity radius is globally ``∞``.
"""
injectivity_radius(::HermitianPositiveDefinite) = Inf
injectivity_radius(::HermitianPositiveDefinite, p) = Inf
injectivity_radius(::HermitianPositiveDefinite, ::AbstractRetractionMethod) = Inf
injectivity_radius(::HermitianPositiveDefinite, p, ::AbstractRetractionMethod) = Inf

function isapprox(p::MatrixSqrtManifoldPoint, q::MatrixSqrtManifoldPoint; kwargs...)
    return isapprox(convert(AbstractMatrix, p), convert(AbstractMatrix, q); kwargs...)
end

"""
    is_flat(M::HermitianPositiveDefinite)

Return true if the [`HermitianPositiveDefinite`](@ref) `M` is one-dimensional, i.e. for
``n = 1``, and false otherwise.
"""
is_flat(M::HermitianPositiveDefinite) = get_parameter(M.size)[1] == 1

@doc raw"""
    manifold_dimension(M::HermitianPositiveDefinite{ℂ})

returns the dimension of
[`HermitianPositiveDefinite`](@ref) `M` ``=\mathcal H(n), n ∈ ℕ``, over ``𝔽 = ℂ``, i.e.
````math
\dim \mathcal H(n) = n^2,
````
the real dimension of the Hermitian matrices.
For the real case see [`manifold_dimension`](@ref manifold_dimension(::SymmetricPositiveDefinite)).
"""
function manifold_dimension(M::HermitianPositiveDefinite{ℂ})
    n = get_parameter(M.size)[1]
    return n^2
end

"""
    mean(
        M::HermitianPositiveDefinite,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolation();
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` using
[`GeodesicInterpolation`](@extref `ManifoldsBase.GeodesicInterpolation`).
"""
mean(::HermitianPositiveDefinite, ::Any)

function default_approximation_method(::HermitianPositiveDefinite, ::typeof(mean))
    return GeodesicInterpolation()
end

@doc raw"""
    convert(::Type{AbstractMatrix}, p::MatrixSqrtManifoldPoint)

return the point `p` as a matrix.
The matrix is either stored within the [`MatrixSqrtManifoldPoint`](@ref) or reconstructed from `p.eigen`.
"""
convert(::Type{AbstractMatrix}, p::MatrixSqrtManifoldPoint) = p.p
function convert(::Type{AbstractMatrix}, p::MatrixSqrtManifoldPoint{Missing})
    return (p.eigen.vectors * Diagonal(p.eigen.values) * p.eigen.vectors')
end

@doc raw"""
    hpd_sqrt(p::AbstractMatrix)
    hpd_sqrt(p::MatrixSqrtManifoldPoint)

return ``p^{\frac{1}{2}}`` by either computing it (if it is missing or for the `AbstractMatrix`)
or returning the stored value from within the [`MatrixSqrtManifoldPoint`](@ref).

This method assumes that `p` represents a Hermitian positive definite matrix.
"""
hpd_sqrt(p)

function hpd_sqrt(p::AbstractMatrix)
    e = eigen(Hermitian(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
    Ssqrt = Diagonal(sqrt.(S))
    return Hermitian(U * Ssqrt * U')
end
hpd_sqrt(p::MatrixSqrtManifoldPoint) = Hermitian(p.sqrt)
function hpd_sqrt(p::MatrixSqrtManifoldPoint{P, Missing}) where {P <: Union{AbstractMatrix, Missing}}
    U = p.eigen.vectors
    S = max.(p.eigen.values, floatmin(eltype(p.eigen.values)))
    Ssqrt = Diagonal(sqrt.(S))
    return Hermitian(U * Ssqrt * U')
end

@doc raw"""
    hpd_sqrt_inv(p::MatrixSqrtManifoldPoint)

return ``p^{-\frac{1}{2}}`` by either computing it (if it is missing or for the `AbstractMatrix`)
or returning the stored value from within the [`MatrixSqrtManifoldPoint`](@ref).

This method assumes that `p` represents a Hermitian positive definite matrix.
"""
hpd_sqrt_inv(p::MatrixSqrtManifoldPoint) = Hermitian(p.sqrt_inv)
function hpd_sqrt_inv(
        p::MatrixSqrtManifoldPoint{P, Q, Missing}
    ) where {P <: Union{AbstractMatrix, Missing}, Q <: Union{AbstractMatrix, Missing}}
    U = p.eigen.vectors
    S = max.(p.eigen.values, floatmin(eltype(p.eigen.values)))
    SsqrtInv = Diagonal(1 ./ sqrt.(S))
    return Hermitian(U * SsqrtInv * U')
end

@doc raw"""
    hpd_sqrt_and_sqrt_inv(p::AbstractMatrix)
    hpd_sqrt_and_sqrt_inv(p::MatrixSqrtManifoldPoint)

Return ``p^{\frac{1}{2}}`` and ``p^{-\frac{1}{2}}`` by either computing them (if they are missing or for the `AbstractMatrix`)
or returning their stored value from within the [`MatrixSqrtManifoldPoint`](@ref).

Compared to calling single methods [`hpd_sqrt`](@ref Manifolds.hpd_sqrt) and [`hpd_sqrt_inv`](@ref Manifolds.hpd_sqrt_inv) this method
only computes the eigenvectors once for the case of the `AbstractMatrix` or if both are missing.

This method assumes that `p` represents a Hermitian positive definite matrix.
"""
hpd_sqrt_and_sqrt_inv(p)

function hpd_sqrt_and_sqrt_inv(p::AbstractMatrix)
    e = eigen(Hermitian(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
    Ssqrt = Diagonal(sqrt.(S))
    SsqrtInv = Diagonal(1 ./ sqrt.(S))
    return (Hermitian(U * Ssqrt * U'), Hermitian(U * SsqrtInv * U'))
end
function hpd_sqrt_and_sqrt_inv(
        p::MatrixSqrtManifoldPoint{P}
    ) where {P <: Union{Missing, AbstractMatrix}}
    return (Hermitian(p.sqrt), Hermitian(p.sqrt_inv))
end
function hpd_sqrt_and_sqrt_inv(
        p::MatrixSqrtManifoldPoint{P, Q, Missing}
    ) where {P <: Union{Missing, AbstractMatrix}, Q <: AbstractMatrix}
    return (Hermitian(p.sqrt), hpd_sqrt_inv(p))
end
function hpd_sqrt_and_sqrt_inv(
        p::MatrixSqrtManifoldPoint{P, Missing, R}
    ) where {P <: Union{Missing, AbstractMatrix}, R <: AbstractMatrix}
    return (hpd_sqrt(p), Hermitian(p.sqrt_inv))
end
function hpd_sqrt_and_sqrt_inv(
        p::MatrixSqrtManifoldPoint{P, Missing, Missing}
    ) where {P <: Union{Missing, AbstractMatrix}}
    S = max.(p.eigen.values, floatmin(eltype(p.eigen.values)))
    U = p.eigen.vectors
    Ssqrt = Diagonal(sqrt.(S))
    SsqrtInv = Diagonal(1 ./ sqrt.(S))
    return (Hermitian(U * Ssqrt * U'), Hermitian(U * SsqrtInv * U'))
end

Base.eltype(p::MatrixSqrtManifoldPoint) = eltype(p.eigen)

@doc raw"""
    project(M::HermitianPositiveDefinite, p, X)

project a matrix from the embedding onto the tangent space ``T_p\mathcal H(n)`` of the
[`HermitianPositiveDefinite`](@ref) matrices, i.e. the set of Hermitian matrices,
by taking the Hermitian part ``\frac{1}{2}(X + X^{\mathrm{H}})``.
"""
project(::HermitianPositiveDefinite, p, X)

project!(::HermitianPositiveDefinite, Y, p, X) = (Y .= Hermitian((X + X') / 2))

@doc raw"""
    rand(M::HermitianPositiveDefinite{ℂ}; vector_at=nothing, tangent_distr=:Gaussian, σ::Real=1/sqrt(n))

Generate a random Hermitian positive definite matrix on the complex
[`HermitianPositiveDefinite`](@ref) manifold `M`, or, if `vector_at` is a point ``p``, a random
tangent vector at ``p``. The parameter `σ` has no effect on a point.

For `tangent_distr = :Gaussian` the coordinates of the tangent vector in the
[`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`) of ``T_p\mathcal H(n)``
are independent and normally distributed with standard deviation `σ`.
For `tangent_distr = :Rician` the tangent vector is ``RR^{\mathrm{H}}`` with
``R = L + \sqrt{σ}U``, where ``L`` is the lower triangular Cholesky factor of ``p`` and ``U`` is
upper triangular with independent complex standard normal entries; here `σ` defaults to
``1/\lVert p\rVert_{\mathrm{F}}`` instead.
"""
rand(M::HermitianPositiveDefinite{ℂ}; σ::Real = 1)

function allocate_result(
        M::HermitianPositiveDefinite, ::typeof(Random.rand), p::MatrixSqrtManifoldPoint
    )
    return zero_vector(M, p)
end

function Random.rand!(
        rng::AbstractRNG,
        M::HermitianPositiveDefinite{ℂ},
        pX;
        vector_at = nothing,
        tangent_distr = :Gaussian,
        σ::Real = (tangent_distr === :Rician && vector_at !== nothing) ?
            one(real(eltype(pX))) / norm(convert(AbstractMatrix, vector_at)) :
            one(real(eltype(pX))) / sqrt(get_parameter(M.size)[1]),
    )
    N = get_parameter(M.size)[1]
    T = eltype(pX)
    if vector_at === nothing
        D = Diagonal(1 .+ rand(rng, N)) # random diagonal matrix
        s = qr(σ * randn(rng, T, N, N)) # random unitary q
        if pX isa MatrixSqrtManifoldPoint
            pX.eigen.values .= D.diag
            pX.eigen.vectors .= s.Q
            sq = sqrt.(D.diag)
            !ismissing(pX.p) && (pX.p .= Hermitian(s.Q * D * s.Q'))
            !ismissing(pX.sqrt) && (pX.sqrt .= s.Q * Diagonal(sq) * s.Q')
            !ismissing(pX.sqrt_inv) && (pX.sqrt_inv .= s.Q * Diagonal(1 ./ sq) * s.Q')
        else
            pX .= Hermitian(s.Q * D * s.Q')
        end
    elseif tangent_distr === :Gaussian
        # an orthonormal basis of the tangent space at vector_at
        Ξ = get_vectors(M, vector_at, get_basis(M, vector_at, DefaultOrthonormalBasis()))
        pX .= sum(σ * randn(rng, length(Ξ)) .* Ξ)
    elseif tangent_distr === :Rician
        C = cholesky(Hermitian(convert(AbstractMatrix, vector_at)))
        R = C.L + sqrt(σ) * triu(randn(rng, T, N, N), 0)
        pX .= R * R'
    end
    return pX
end

@doc raw"""
    representation_size(M::HermitianPositiveDefinite)

Return the size of an array representing an element on the
[`HermitianPositiveDefinite`](@ref) manifold `M`, i.e. ``n×n``, the size of such a
Hermitian positive definite matrix on ``\mathcal M = \mathcal H(n)``.
"""
function representation_size(M::HermitianPositiveDefinite)
    n = get_parameter(M.size)[1]
    return (n, n)
end

function Base.show(io::IO, M::HermitianPositiveDefinite{𝔽}) where {𝔽}
    n = get_parameter(M.size)[1]
    p = 𝔽 === ℂ ? "" : ", $𝔽"
    kw = get_parameter_type(M) === :type ? "" : "; parameter=:$(get_parameter_type(M))"
    return print(io, "HermitianPositiveDefinite($n$p$kw)")
end

function Base.show(io::IO, ::MIME"text/plain", p::MatrixSqrtManifoldPoint)
    pre = " "
    summary(io, p)
    println(io, "\np:")
    sp = sprint(show, "text/plain", p.p; context = io, sizehint = 0)
    sp = replace(sp, '\n' => "\n$(pre)")
    println(io, pre, sp)
    println(io, "p^{1/2}:")
    sps = sprint(show, "text/plain", p.sqrt; context = io, sizehint = 0)
    sps = replace(sps, '\n' => "\n$(pre)")
    println(io, pre, sps)
    println(io, "p^{-1/2}:")
    spi = sprint(show, "text/plain", p.sqrt_inv; context = io, sizehint = 0)
    spi = replace(spi, '\n' => "\n$(pre)")
    return print(io, pre, spi)
end

@doc raw"""
    zero_vector(M::HermitianPositiveDefinite, p)
    zero_vector!(M::HermitianPositiveDefinite, X, p)

returns the zero tangent vector in the tangent space of the Hermitian positive
definite matrix `p` on the [`HermitianPositiveDefinite`](@ref) manifold `M`.
It can be returned in place of `X`.
"""
zero_vector(::HermitianPositiveDefinite, ::Any)

function zero_vector(M::HermitianPositiveDefinite, p::MatrixSqrtManifoldPoint)
    return zero_vector(M, convert(AbstractMatrix, p))
end

zero_vector!(::HermitianPositiveDefinite, X, ::Any) = fill!(X, 0)
