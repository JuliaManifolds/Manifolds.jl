@doc raw"""
    SkewHermitianMatrices{n,𝔽} <: AbstractDecoratorManifold{𝔽}

The [`AbstractManifold`](@ref) $ \operatorname{SkewHerm}(n)$ consisting of the real- or
complex-valued skew-hermitian matrices of size ``n × n``, i.e. the set

````math
\operatorname{SkewHerm}(n) = \bigl\{p  ∈ 𝔽^{n × n}\ \big|\ p^{\mathrm{H}} = -p \bigr\},
````
where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transpose,
and the field $𝔽 ∈ \{ ℝ, ℂ, ℍ\}$.

Though it is slightly redundant, usually the matrices are stored as ``n × n`` arrays.

Note that in this representation, the real-valued part of the diagonal must be zero,
which is also reflected in the
[`manifold_dimension`](@ref manifold_dimension(::SkewHermitianMatrices{N,𝔽}) where {N,𝔽}).

# Constructor

    SkewHermitianMatrices(n::Int, field::AbstractNumbers=ℝ)

Generate the manifold of ``n × n`` skew-hermitian matrices.
"""
struct SkewHermitianMatrices{n,𝔽} <: AbstractDecoratorManifold{𝔽} end

function SkewHermitianMatrices(n::Int, field::AbstractNumbers=ℝ)
    return SkewHermitianMatrices{n,field}()
end

@doc raw"""
    SkewSymmetricMatrices{n}

Generate the manifold of ``n × n`` real skew-symmetric matrices.
This is equivalent to [`SkewHermitianMatrices(n, ℝ)`](@ref).

# Constructor

    SkewSymmetricMatrices(n::Int)
"""
const SkewSymmetricMatrices{n} = SkewHermitianMatrices{n,ℝ}

SkewSymmetricMatrices(n::Int) = SkewSymmetricMatrices{n}()
@deprecate SkewSymmetricMatrices(n::Int, 𝔽) SkewHermitianMatrices(n, 𝔽)

function active_traits(f, ::SkewSymmetricMatrices, args...)
    return merge_traits(IsEmbeddedSubmanifold())
end

function allocation_promotion_function(
    ::SkewHermitianMatrices{<:Any,ℂ},
    ::typeof(get_vector),
    args::Tuple,
)
    return complex
end

@doc raw"""
    check_point(M::SkewHermitianMatrices{n,𝔽}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`SkewHermitianMatrices`](@ref) `M`, i.e.
whether `p` is a skew-hermitian matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](@ref) `𝔽`.

The tolerance for the skew-symmetry of `p` can be set using `kwargs...`.
"""
function check_point(M::SkewHermitianMatrices{n,𝔽}, p; kwargs...) where {n,𝔽}
    mpv = check_point(decorated_manifold(M), p; kwargs...)
    mpv === nothing || return mpv
    if !isapprox(p, -p'; kwargs...)
        return DomainError(
            norm(p + p'),
            "The point $(p) does not lie on $M, since it is not skew-hermitian.",
        )
    end
    return nothing
end

"""
    check_vector(M::SkewHermitianMatrices{n}, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SkewHermitianMatrices`](@ref) `M`, i.e. `X` must be a skew-hermitian matrix of size `(n,n)`
and its values have to be from the correct [`AbstractNumbers`](@ref).
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

function get_coordinates_orthonormal!(
    M::SkewSymmetricMatrices{N},
    Y,
    p,
    X,
    ::RealNumbers,
) where {N}
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
    M::SkewHermitianMatrices{N,ℂ},
    Y,
    p,
    X,
    ::ComplexNumbers,
) where {N}
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

get_embedding(::SkewHermitianMatrices{N,𝔽}) where {N,𝔽} = Euclidean(N, N; field=𝔽)

function get_vector_orthonormal!(
    M::SkewSymmetricMatrices{N},
    Y,
    p,
    X,
    ::RealNumbers,
) where {N}
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
    M::SkewHermitianMatrices{N,ℂ},
    Y,
    p,
    X,
    ::ComplexNumbers,
) where {N}
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

@doc raw"""
    manifold_dimension(M::SkewHermitianMatrices{n,𝔽})

Return the dimension of the [`SkewHermitianMatrices`](@ref) matrix `M` over the number
system `𝔽`, i.e.

````math
\dim \mathrm{SkewHerm}(n,ℝ) = \frac{n(n+1)}{2} \dim_ℝ 𝔽 - n,
````

where ``\dim_ℝ 𝔽`` is the [`real_dimension`](@ref) of ``𝔽``. The first term corresponds to
only the upper triangular elements of the matrix being unique, and the second term
corresponds to the constraint that the real part of the diagonal be zero.
"""
function manifold_dimension(::SkewHermitianMatrices{N,𝔽}) where {N,𝔽}
    return div(N * (N + 1), 2) * real_dimension(𝔽) - N
end

function number_of_coordinates(M::SkewHermitianMatrices{N,ℂ}, ::AbstractBasis{ℂ}) where {N}
    return manifold_dimension(M)
end

@doc raw"""
    project(M::SkewHermitianMatrices, p)

Projects `p` from the embedding onto the [`SkewHermitianMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_{\operatorname{SkewHerm}(n)}(p) = \frac{1}{2} \bigl( p - p^{\mathrm{H}} \bigr),
````

where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transposed.
"""
project(::SkewHermitianMatrices, ::Any)

function project!(M::SkewHermitianMatrices, q, p)
    q .= (p .- p') ./ 2
    return q
end

@doc raw"""
    project(M::SkewHermitianMatrices, p, X)

Project the matrix `X` onto the tangent space at `p` on the [`SkewHermitianMatrices`](@ref) `M`,

````math
\operatorname{proj}_p(X) = \frac{1}{2} \bigl( X - X^{\mathrm{H}} \bigr),
````

where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transposed.
"""
project(::SkewHermitianMatrices, ::Any, ::Any)

project!(M::SkewHermitianMatrices, Y, p, X) = project!(M, Y, X)

representation_size(::SkewHermitianMatrices{N}) where {N} = (N, N)

function Base.show(io::IO, ::SkewHermitianMatrices{n,F}) where {n,F}
    return print(io, "SkewHermitianMatrices($(n), $(F))")
end
function Base.show(io::IO, ::SkewSymmetricMatrices{n}) where {n}
    return print(io, "SkewSymmetricMatrices($(n))")
end
