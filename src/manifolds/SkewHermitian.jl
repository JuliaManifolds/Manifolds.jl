@doc raw"""
    SkewHermitianMatrices{𝔽, T} <: AbstractDecoratorManifold{𝔽}

The [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)  ``\operatorname{SkewHerm}(n)`` consisting of the real- or
complex-valued skew-hermitian matrices of size ``n×n``, i.e. the set

````math
\operatorname{SkewHerm}(n) = \bigl\{p  ∈ 𝔽^{n×n}\ \big|\ p^{\mathrm{H}} = -p \bigr\},
````

where ``⋅^{\mathrm{H}}`` denotes the Hermitian, i.e. complex conjugate transpose,
and the field ``𝔽 ∈ \{ ℝ, ℂ, ℍ\}``.

Though it is slightly redundant, usually the matrices are stored as ``n×n`` arrays.

Note that in this representation, the real-valued part of the diagonal must be zero,
which is also reflected in the
[`manifold_dimension`](@ref manifold_dimension(::SkewHermitianMatrices)).

# Constructor

    SkewHermitianMatrices(n::Int, field::AbstractNumbers=ℝ; parameter::Symbol=:type)

Generate the manifold of ``n×n`` skew-hermitian matrices.
"""
struct SkewHermitianMatrices{𝔽, T} <: AbstractDecoratorManifold{𝔽}
    size::T
end

function SkewHermitianMatrices(n::Int, field::AbstractNumbers = ℝ; parameter::Symbol = :type)
    n < 0 && throw(
        DomainError(n, "The matrix size n of skew-Hermitian matrices must be nonnegative, but was $(n).")
    )
    size = wrap_type_parameter(parameter, (n,))
    return SkewHermitianMatrices{field, typeof(size)}(size)
end

@doc raw"""
    SkewSymmetricMatrices{T}

Generate the manifold of ``n×n`` real skew-symmetric matrices.
This is equivalent to [`SkewHermitianMatrices(n, ℝ)`](@ref).

# Constructor

    SkewSymmetricMatrices(n::Int; parameter::Symbol=:type)
"""
const SkewSymmetricMatrices{T} = SkewHermitianMatrices{ℝ, T}

function SkewSymmetricMatrices(n::Int; parameter::Symbol = :type)
    return SkewHermitianMatrices(n; parameter = parameter)
end

function allocation_promotion_function(
        ::SkewHermitianMatrices{ℂ},
        ::typeof(get_vector),
        args::Tuple,
    )
    return complex
end

@doc raw"""
    check_point(M::SkewHermitianMatrices, p; kwargs...)

Check whether `p` is a valid manifold point on the [`SkewHermitianMatrices`](@ref) `M`, i.e.
whether `p` is a skew-hermitian matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](@extref ManifoldsBase number-system) `𝔽`.

The tolerance for the skew-symmetry of `p` can be set using `kwargs...`.
"""
function check_point(M::SkewHermitianMatrices{𝔽}, p; kwargs...) where {𝔽}
    if !isapprox(p, -p'; kwargs...)
        return DomainError(
            norm(p + p'),
            "The point $(p) does not lie on $M, since it is not skew-hermitian.",
        )
    end
    return nothing
end

"""
    check_vector(M::SkewHermitianMatrices, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SkewHermitianMatrices`](@ref) `M`, i.e. `X` must be a skew-hermitian matrix of size `(n,n)`
and its values have to be from the correct [`AbstractNumbers`](@extref ManifoldsBase number-system).
The tolerance for the skew-symmetry of `p` and `X` can be set using `kwargs...`.
"""
function check_vector(M::SkewHermitianMatrices, p, X; kwargs...)
    return check_point(M, X; kwargs...)  # manifold is its own tangent space
end

function get_basis(M::SkewHermitianMatrices, p, B::DiagonalizingOrthonormalBasis)
    Ξ = get_basis(M, p, DefaultOrthonormalBasis()).data
    κ = zeros(real(eltype(p)), manifold_dimension(M))
    return CachedBasis(B, κ, Ξ)
end

function get_coordinates_orthonormal!(M::SkewSymmetricMatrices, Y, p, X, ::RealNumbers)
    N = get_parameter(M.size)[1]
    dim = manifold_dimension(M)
    @assert size(Y) == (dim,)
    @assert size(X) == (N, N)
    @assert dim == div(N * (N - 1), 2)
    k = 1
    for i in 1:N, j in (i + 1):N
        @inbounds Y[k] = X[i, j] * sqrt(2)
        k += 1
    end
    return Y
end
function get_coordinates_orthonormal!(
        M::SkewHermitianMatrices{ℂ},
        Y,
        p,
        X,
        ::RealNumbers,
    )
    N = get_parameter(M.size)[1]
    dim = manifold_dimension(M)
    @assert size(Y) == (dim,)
    @assert size(X) == (N, N)
    @assert dim == N^2
    k = 1
    for i in 1:N, j in i:N
        if i == j # real-part zero on the diagonal -> just one basis vector per diag entry
            @inbounds Y[k] = imag(X[i, j])
            k += 1
        else
            @inbounds Y[k] = real(X[i, j]) * sqrt(2)
            @inbounds Y[k + 1] = imag(X[i, j]) * sqrt(2)
            k += 2
        end
    end
    return Y
end

function get_embedding(::SkewHermitianMatrices{𝔽, TypeParameter{Tuple{N}}}) where {N, 𝔽}
    return Euclidean(N, N; field = 𝔽)
end
function get_embedding(M::SkewHermitianMatrices{𝔽, Tuple{Int}}) where {𝔽}
    N = get_parameter(M.size)[1]
    return Euclidean(N, N; field = 𝔽, parameter = :field)
end

function ManifoldsBase.get_embedding_type(::SkewHermitianMatrices)
    return ManifoldsBase.EmbeddedSubmanifoldType()
end

function get_vector_orthonormal!(M::SkewSymmetricMatrices, Y, p, X, ::RealNumbers)
    N = get_parameter(M.size)[1]
    dim = manifold_dimension(M)
    @assert size(X) == (dim,)
    @assert size(Y) == (N, N)
    k = 1
    for i in 1:N
        Y[i, i] = convert(eltype(p), 0.0)
    end
    for i in 1:N, j in (i + 1):N
        @inbounds Y[i, j] = X[k] / sqrt(2)
        @inbounds Y[j, i] = -X[k] / sqrt(2)
        k += 1
    end
    return Y
end
function get_vector_orthonormal!(
        M::SkewHermitianMatrices{ℂ},
        Y,
        p,
        X,
        ::RealNumbers,
    )
    N = get_parameter(M.size)[1]
    dim = manifold_dimension(M)
    @assert size(X) == (dim,)
    @assert size(Y) == (N, N)
    k = 1
    for i in 1:N, j in i:N
        if i == j # real zero on the diag
            @inbounds Y[i, j] = X[k] * 1im
            k += 1
        else
            @inbounds Y[i, j] = (X[k] + X[k + 1] * 1im) / sqrt(2)
            k += 2
            @inbounds Y[j, i] = -conj(Y[i, j])
        end
    end
    return Y
end

"""
    is_flat(::SkewHermitianMatrices)

Return true. [`SkewHermitianMatrices`](@ref) is a flat manifold.
"""
is_flat(M::SkewHermitianMatrices) = true

@doc raw"""
    manifold_dimension(M::SkewHermitianMatrices)

Return the dimension of the [`SkewHermitianMatrices`](@ref) matrix `M` over the number
system `𝔽`, i.e.

````math
\dim \mathrm{SkewHerm}(n,𝔽) = \frac{n(n+1)}{2} \dim_ℝ 𝔽 - n,
````

where ``\dim_ℝ 𝔽`` is the [`real_dimension`](@extref `ManifoldsBase.real_dimension-Tuple{ManifoldsBase.AbstractNumbers}`) of ``𝔽``. The first term corresponds to
only the upper triangular elements of the matrix being unique, and the second term
corresponds to the constraint that the real part of the diagonal be zero.
"""
function manifold_dimension(M::SkewHermitianMatrices{𝔽}) where {𝔽}
    N = get_parameter(M.size)[1]
    return div(N * (N + 1), 2) * real_dimension(𝔽) - N
end

@doc raw"""
    project(M::SkewHermitianMatrices, p)

Projects `p` from the embedding onto the [`SkewHermitianMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_{\operatorname{SkewHerm}(n)}(p) = \frac{1}{2} \bigl( p - p^{\mathrm{H}} \bigr),
````

where ``⋅^{\mathrm{H}}`` denotes the Hermitian, i.e. complex conjugate transposed.
"""
project(::SkewHermitianMatrices, ::Any)

function project!(::SkewHermitianMatrices, q, p)
    q .= (p .- p') ./ 2
    return q
end

@doc raw"""
    project(M::SkewHermitianMatrices, p, X)

Project the matrix `X` onto the tangent space at `p` on the [`SkewHermitianMatrices`](@ref) `M`,

````math
\operatorname{proj}_p(X) = \frac{1}{2} \bigl( X - X^{\mathrm{H}} \bigr),
````

where ``⋅^{\mathrm{H}}`` denotes the Hermitian, i.e. complex conjugate transposed.
"""
project(::SkewHermitianMatrices, ::Any, ::Any)

project!(M::SkewHermitianMatrices, Y, p, X) = project!(M, Y, X)

function Random.rand!(
        rng::AbstractRNG,
        M::SkewHermitianMatrices,
        pX;
        σ::Real = one(real(eltype(pX))),
        kwargs...,
    )
    rand!(rng, pX)
    pX .= (σ / (2 * norm(pX))) .* (pX - pX')
    return pX
end

function representation_size(M::SkewHermitianMatrices)
    N = get_parameter(M.size)[1]
    return (N, N)
end

function Base.show(io::IO, ::SkewHermitianMatrices{𝔽, TypeParameter{Tuple{n}}}) where {n, 𝔽}
    return print(io, "SkewHermitianMatrices($(n), $(𝔽))")
end
function Base.show(io::IO, ::SkewSymmetricMatrices{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "SkewSymmetricMatrices($(n))")
end
function Base.show(io::IO, M::SkewHermitianMatrices{𝔽, Tuple{Int}}) where {𝔽}
    n = get_parameter(M.size)[1]
    return print(io, "SkewHermitianMatrices($(n), $(𝔽); parameter=:field)")
end
function Base.show(io::IO, M::SkewSymmetricMatrices{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return print(io, "SkewSymmetricMatrices($(n); parameter=:field)")
end

@doc raw"""
    Y = Weingarten(M::SkewSymmetricMatrices, p, X, V)
    Weingarten!(M::SkewSymmetricMatrices, Y, p, X, V)

Compute the Weingarten map ``\mathcal W_p`` at `p` on the [`SkewSymmetricMatrices`](@ref) `M` with respect to the
tangent vector ``X \in T_p\mathcal M`` and the normal vector ``V \in N_p\mathcal M``.

Since this a flat space by itself, the result is always the zero tangent vector.
"""
Weingarten(::SkewSymmetricMatrices, p, X, V)

Weingarten!(::SkewSymmetricMatrices, Y, p, X, V) = fill!(Y, 0)
