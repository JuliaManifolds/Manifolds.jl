@doc raw"""
    InvertibleMatrices{𝔽,T} <: AbstractDecoratorManifold{𝔽}

The [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)
consisting of the real- or complex-valued invertible matrices, that is the set

```math
\bigl\{p  ∈ 𝔽^{n×n}\ \big|\ \det(p) \neq 0 \bigr\},
```
where the field ``𝔽 ∈ \{ ℝ, ℂ\}``.

# Constructor

    InvertibleMatrices(n::Int, field::AbstractNumbers=ℝ)

Generate the manifold of ``n×n`` invertible matrices.
"""
struct InvertibleMatrices{𝔽,T} <: AbstractDecoratorManifold{𝔽}
    size::T
end

function InvertibleMatrices(n::Int, field::AbstractNumbers=ℝ; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, (n,))
    return InvertibleMatrices{field,typeof(size)}(size)
end

function active_traits(f, ::InvertibleMatrices, args...)
    return merge_traits(IsEmbeddedSubmanifold())
end

@doc raw"""
    check_point(M::InvertibleMatrices{n,𝔽}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`InvertibleMatrices`](@ref) `M`, i.e.
whether `p` is an invertible matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](@extref ManifoldsBase number-system) `𝔽`.
"""
function check_point(M::InvertibleMatrices, p; kwargs...)
    if det(p) == 0
        return DomainError(
            det(p),
            "The point $(p) does not lie on $(M), since its determinant is zero and hence it is not invertible.",
        )
    end
    return nothing
end

"""
    check_vector(M::InvertibleMatrices{n,𝔽}, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`InvertibleMatrices`](@ref) `M`, which are all matrices of size ``n×n``
its values have to be from the correct [`AbstractNumbers`](@extref ManifoldsBase number-system).
"""
function check_vector(M::InvertibleMatrices, p, X; kwargs...)
    return nothing
end

embed(::InvertibleMatrices, p) = p
embed(::InvertibleMatrices, p, X) = X

function get_coordinates(
    ::InvertibleMatrices{ℝ,<:Any},
    p,
    X,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
)
    return vec(X)
end

function get_coordinates!(
    ::InvertibleMatrices{ℝ,<:Any},
    Xⁱ,
    p,
    X,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
)
    return copyto!(Xⁱ, X)
end

function get_embedding(::InvertibleMatrices{𝔽,TypeParameter{Tuple{n}}}) where {n,𝔽}
    return Euclidean(n, n; field=𝔽)
end
function get_embedding(M::InvertibleMatrices{𝔽,Tuple{Int}}) where {𝔽}
    n = get_parameter(M.size)[1]
    return Euclidean(n, n; field=𝔽, parameter=:field)
end

function get_vector(
    M::InvertibleMatrices{ℝ,<:Any},
    p,
    Xⁱ,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
)
    n = get_parameter(M.size)[1]
    return reshape(Xⁱ, n, n)
end

function get_vector!(
    ::InvertibleMatrices{ℝ,<:Any},
    X,
    p,
    Xⁱ,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
)
    return copyto!(X, Xⁱ)
end

"""
    is_flat(::InvertibleMatrices)

Return true. [`InvertibleMatrices`](@ref) is a flat manifold.
"""
is_flat(M::InvertibleMatrices) = true

@doc raw"""
    manifold_dimension(M::InvertibleMatrices{n,𝔽})

Return the dimension of the [`InvertibleMatrices`](@ref) matrix `M` over the number system
`𝔽`, which is the same dimension as its embedding, the [`Euclidean`](@ref)`(n, n; field=𝔽)`.
"""
function manifold_dimension(M::InvertibleMatrices{<:Any,𝔽}) where {𝔽}
    return manifold_dimension(get_embedding(M))
end

@doc raw"""
    Random.rand(M::InvertibleMatrices; vector_at=nothing, kwargs...)

If `vector_at` is `nothing`, return a random point on the [`InvertibleMatrices`](@ref)
manifold `M` by using `rand` in the embedding.

If `vector_at` is not `nothing`, return a random tangent vector from the tangent space of
the point `vector_at` on the [`InvertibleMatrices`](@ref) by using by using `rand` in the
embedding.
"""
rand(M::InvertibleMatrices; kwargs...)

function Random.rand!(M::InvertibleMatrices, pX; kwargs...)
    rand!(get_embedding(M), pX; kwargs...)
    while det(pX) == 0
        rand!(get_embedding(M), pX; kwargs...)
    end
    return pX
end
function Random.rand!(rng::AbstractRNG, M::InvertibleMatrices, pX; kwargs...)
    rand!(rng, get_embedding(M), pX; kwargs...)
    while det(pX) == 0
        rand!(rng, get_embedding(M), pX; kwargs...)
    end
    return pX
end

function Base.show(io::IO, ::InvertibleMatrices{𝔽,TypeParameter{Tuple{n}}}) where {n,𝔽}
    return print(io, "InvertibleMatrices($(n), $(𝔽))")
end
function Base.show(io::IO, M::InvertibleMatrices{𝔽,Tuple{Int}}) where {𝔽}
    n = get_parameter(M.size)[1]
    return print(io, "InvertibleMatrices($(n), $(𝔽); parameter=:field)")
end

@doc raw"""
    Y = Weingarten(M::InvertibleMatrices, p, X, V)
    Weingarten!(M::InvertibleMatrices, Y, p, X, V)

Compute the Weingarten map ``\mathcal W_p`` at `p` on the [`InvertibleMatrices`](@ref) `M` with respect to the
tangent vector ``X \in T_p\mathcal M`` and the normal vector ``V \in N_p\mathcal M``.

Since this a flat space by itself, the result is always the zero tangent vector.
"""
Weingarten(::InvertibleMatrices, p, X, V)

Weingarten!(::InvertibleMatrices, Y, p, X, V) = fill!(Y, 0)
