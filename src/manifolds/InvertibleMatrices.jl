@doc raw"""
    InvertibleMatrices{𝔽,T} <: AbstractDecoratorManifold{𝔽}

The [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)  ``\operatorname{Sym}(n)``
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

function allocation_promotion_function(
    ::InvertibleMatrices{ℂ,<:Any},
    ::typeof(get_vector),
    args::Tuple,
)
    return complex
end

@doc raw"""
    check_point(M::InvertibleMatrices{n,𝔽}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`InvertibleMatrices`](@ref) `M`, i.e.
whether `p` is a symmetric matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](@extref ManifoldsBase number-system) `𝔽`.

The tolerance for the symmetry of `p` can be set using `kwargs...`.
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
[`InvertibleMatrices`](@ref) `M`, which are all matrces of size ``n×n```
its values have to be from the correct [`AbstractNumbers`](@extref ManifoldsBase number-system).
"""
function check_vector(M::InvertibleMatrices, p, X; kwargs...)
    return nothing
end

embed(::InvertibleMatrices, p) = p
embed(::InvertibleMatrices, p, X) = X

function get_embedding(::InvertibleMatrices{𝔽,TypeParameter{Tuple{n}}}) where {n,𝔽}
    return Euclidean(n, n; field=𝔽)
end
function get_embedding(M::InvertibleMatrices{𝔽,Tuple{Int}}) where {𝔽}
    n = get_parameter(M.size)[1]
    return Euclidean(n, n; field=𝔽, parameter=:field)
end

"""
    is_flat(::InvertibleMatrices)

Return true. [`InvertibleMatrices`](@ref) is a flat manifold.
"""
is_flat(M::InvertibleMatrices) = true

@doc raw"""
    manifold_dimension(M::InvertibleMatrices{n,𝔽})

Return the dimension of the [`InvertibleMatrices`](@ref) matrix `M` over the number system
`𝔽`, which is the same dimension as its embedding, the [`Euclidean`](@ref)`(n,n)`.
"""
function manifold_dimension(M::InvertibleMatrices{<:Any,𝔽}) where {𝔽}
    return manifold_dimension(get_embedding(M))
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
