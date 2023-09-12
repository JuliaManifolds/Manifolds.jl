@doc raw"""
    OrthogonalPoint <: AbstractManifoldPoint

A type to represent points on a manifold [`Flag`](@ref) in the orthogonal coordinates
representation, i.e. a rotation matrix.
"""
struct OrthogonalPoint{T<:AbstractMatrix} <: AbstractManifoldPoint
    value::T
end

@doc raw"""
    OrthogonalTVector <: TVector

A type to represent tangent vectors to points on a [`Flag`](@ref) manifold  in the
orthogonal coordinates representation.
"""
struct OrthogonalTVector{T<:AbstractMatrix} <: TVector
    value::T
end

ManifoldsBase.@manifold_vector_forwards OrthogonalTVector value
ManifoldsBase.@manifold_element_forwards OrthogonalPoint value

Base.eltype(p::OrthogonalPoint) = eltype(p.value)
Base.eltype(X::OrthogonalTVector) = eltype(X.value)

"""
    ZeroTuple

Internal structure for representing shape of a [`Flag`](@ref) manifold. Behaves like
a normal tuple, except at index zero returns value 0.
"""
struct ZeroTuple{TupT}
    x::TupT
end

ZeroTuple(x::Tuple) = ZeroTuple{typeof(x)}(x)

function Base.getindex(t::ZeroTuple, i::Int)
    if i == 0
        return 0
    else
        return t.x[i]
    end
end

@doc raw"""
    Flag{N,d} <: AbstractDecoratorManifold{ℝ}

Flag manifold of ``d`` subspaces of ``ℝ^N`` [YeWongLim:2021](@cite). By default the manifold uses
the Stiefel coordinates representation, embedding it in the [`Stiefel`](@ref) manifold.
The other available representation is an embedding in [`OrthogonalMatrices`](@ref).
It can be utilized using [`OrthogonalPoint`](@ref) and [`OrthogonalTVector`](@ref) wrappers.

Tangent space is represented in the block-skew-symmetric form.

# Constructor

    Flag(N, n1, n2, ..., nd)

Generate the manifold ``\operatorname{Flag}(n_1, n_2, ..., n_d; N)`` of subspaces
```math
𝕍_1 ⊆ 𝕍_2 ⊆ ⋯ ⊆ V_d, \quad \operatorname{dim}(𝕍_i) = n_i
```
where ``𝕍_i`` for ``i ∈ 1, 2, …, d`` are subspaces of ``ℝ^N`` of dimension
``\operatorname{dim} 𝕍_i = n_i``.
"""
struct Flag{N,dp1} <: AbstractDecoratorManifold{ℝ}
    subspace_dimensions::ZeroTuple{NTuple{dp1,Int}}
end

function Flag(N, ns::Vararg{Int,I}) where {I}
    if ns[1] <= 0
        error(
            "First dimension in the sequence ns must be strictly positive, but is $(ns[1]).",
        )
    end
    for i in 1:(length(ns) - 1)
        if ns[i] >= ns[i + 1]
            error("Sequence of dimensions must be strictly increasing, received $ns")
        end
    end
    if ns[end] >= N
        error(
            "Last dimension in sequence (given: $(ns[end])) must be strictly lower than N (given: $N).",
        )
    end
    return Flag{N,I + 1}(ZeroTuple(tuple(ns..., N)))
end

function active_traits(f, ::Flag, args...)
    return merge_traits(IsEmbeddedManifold())
end

"""
    get_embedding(M::Flag)

Get the embedding of the [`Flag`](@ref) manifold `M`, i.e. the [`Stiefel`](@ref) manifold.
"""
get_embedding(M::Flag{N,dp1}) where {N,dp1} = Stiefel(N, M.subspace_dimensions[dp1 - 1])

@doc raw"""
    injectivity_radius(M::Flag)
    injectivity_radius(M::Flag, p)

Return the injectivity radius on the [`Flag`](@ref) `M`, which is $\frac{π}{2}$.
"""
injectivity_radius(::Flag) = π / 2
injectivity_radius(::Flag, p) = π / 2
injectivity_radius(::Flag, ::AbstractRetractionMethod) = π / 2
injectivity_radius(::Flag, p, ::AbstractRetractionMethod) = π / 2

function Base.isapprox(M::Flag, p, X, Y; atol=sqrt(max_eps(X, Y)), kwargs...)
    return isapprox(norm(M, p, X - Y), 0; atol=atol, kwargs...)
end
function Base.isapprox(M::Flag, p, q; atol=sqrt(max_eps(p, q)), kwargs...)
    X = inverse_retract(M, p, q, PolarInverseRetraction())
    return isapprox(norm(M, p, X), 0; atol=atol, kwargs...)
end

@doc raw"""
    manifold_dimension(M::Flag)

Return dimension of flag manifold ``\operatorname{Flag}(n_1, n_2, ..., n_d; N)``.
The formula reads ``\sum_{i=1}^d (n_i-n_{i-1})(N-n_i)``.
"""
function manifold_dimension(M::Flag{N,dp1}) where {N,dp1}
    dim = 0
    for i in 1:(dp1 - 1)
        dim +=
            (M.subspace_dimensions[i] - M.subspace_dimensions[i - 1]) *
            (N - M.subspace_dimensions[i])
    end
    return dim
end

function Base.show(io::IO, M::Flag{N}) where {N}
    print(io, "Flag($(N)")
    for d_i in M.subspace_dimensions.x[1:(end - 1)]
        print(io, ", $d_i")
    end
    return print(io, ")")
end

"""
    convert(::Type{AbstractMatrix}, M::Flag, p::OrthogonalPoint, X::OrthogonalTVector)

Convert tangent vector from [`Flag`](@ref) manifold `M` from orthogonal representation to
Stiefel representation.
"""
function convert(::Type{AbstractMatrix}, M::Flag, p::OrthogonalPoint, X::OrthogonalTVector)
    (N, k) = representation_size(M)
    return p.value * X.value[:, 1:k]
end

"""
    convert(::Type{OrthogonalTVector}, M::Flag, p::AbstractMatrix, X::AbstractMatrix)

Convert tangent vector from [`Flag`](@ref) manifold `M` from Stiefel representation to
orthogonal representation.
"""
function convert(::Type{OrthogonalTVector}, M::Flag, p::AbstractMatrix, X::AbstractMatrix)
    (N, k) = representation_size(M)
    out = similar(X, N, N)
    fill!(out, 0)

    p_ortho = convert(OrthogonalPoint, M, p)
    pX = p_ortho.value' * X

    out[:, 1:k] = pX
    out[1:k, (k + 1):N] = -transpose(view(pX, (k + 1):N, 1:k))

    return OrthogonalTVector(out)
end

"""
    convert(::Type{AbstractMatrix}, M::Flag, p::OrthogonalPoint)

Convert point `p` from [`Flag`](@ref) manifold `M` from orthogonal representation to
Stiefel representation.
"""
function convert(::Type{AbstractMatrix}, M::Flag, p::OrthogonalPoint)
    (N, k) = representation_size(M)
    return p.value[:, 1:k]
end

"""
    convert(::Type{OrthogonalPoint}, M::Flag, p::AbstractMatrix)

Convert point `p` from [`Flag`](@ref) manifold `M` from Stiefel representation to
orthogonal representation.
"""
function convert(::Type{OrthogonalPoint}, M::Flag, p::AbstractMatrix)
    (N, k) = representation_size(M)
    out = similar(p, N, N)
    fill!(out, 0)
    out[:, 1:k] = p
    out[:, (k + 1):N] = nullspace(p')
    return OrthogonalPoint(out)
end
