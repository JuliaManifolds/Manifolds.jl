@doc raw"""
     ProjectorPoint <: AbstractManifoldPoint

 A type to represent points on a manifold [`Grassmann`](@ref) that are orthogonal projectors,
 i.e. a matrix ``p ∈ \mathbb F^{n,n}`` projecting onto a ``k``-dimensional subspace.
 """
 struct ProjectorPoint{T<:AbstractMatrix} <: AbstractManifoldPoint
     value::T
 end

 @doc raw"""
     ProjectorTVector <: TVector

 A type to represent tangent vectors to points on a manifold that are orthogonal projectors.
 """
 struct ProjectorTVector{T<:AbstractMatrix} <: TVector
     value::T
 end

 @doc raw"""
     check_point(::Grassmann{n,k}, p::ProjectorPoint; kwargs...)

 Check whether a orthogonal projector is a point from the [`Grassmann`](@ref)`(n,k)` manifold,
 i.e. the [`ProjectorPoint`](@ref) ``p ∈ \mathbb F^{n×n}``, ``\mathbb F ∈ \{\mathbb R, \mathbb C\}``
 has to fulfill ``p^{\mathrm{T}} = p``, ``p^2=p``, and ``\operatorname{rank} p = k`.
 """
function check_point(M::Grassmann{n,k,𝔽}, p::ProjectorPoint; kwargs...) where {n,k,𝔽}
   c = p.value * p.value
    if !isapprox(c, p.value; kwargs...)
        return DomainError(
            norm(c - p.value),
            "The point $(p) is not equal to its square $c, so it does not lie on $M.",
        )
    end
    if !isapprox(p.value, transpose(p.value); kwargs...)
        return DomainError(
            norm(c - p),
            "The point $(p) is not equal to its transpose, so it does not lie on $M.",
        )
    end
    k2 = rank(p.value; kwargs...)
    if k2 != k
        return DomainError(
            k2,
            "The point $(p) is a projector of rank $k2 and not of rank $k, so it does not lie on $(M).",
        )
    end
    return nothing
end

@doc raw"""
    check_size(M::Grassmann{n,k,𝔽}, p::ProjectorPoint; kwargs...) where {n,k}

check that the [`ProjectorPoint`](@ref) is of correctsize, i.e. from ``\mathbb F^{n×n}``
"""
function check_size(M::Grassmann{n,k,𝔽}, p::ProjectorPoint) where {n,k,𝔽}
    return check_size(get_embedding(M,p), p.value; kwargs,...)
end

@doc raw"""
    check_size(M::Grassmann{n,k,𝔽}, p::ProjectorPoint, X::ProjectorTVector; kwargs...) where {n,k}

check that the [`ProjectorTVector`](@ref) is of correctsize, i.e. from ``\mathbb F^{n×n}``
"""
function check_size(M::Grassmann{n,k,𝔽}, p::ProjectorPoint, X::ProjectorTVector) where {n,k,𝔽}
    return check_size(get_embedding(M,p), p.value, X.value; kwargs...)
end

@doc raw"""
    check_vector(::Grassmann{n,k,𝔽}, p::ProjectorPoint, X::ProjectorTVector; kwargs...) where {n,k,𝔽}

Check whether the [`ProjectorTVector`]('ref) `X` is from the tangent space ``T_p\operatorname{Gr}(n,k) ``
at the [`ProjectorPoint`](@ref) `p` on the [`Grassmann`](@ref) manifold ``\operatorname{Gr}(n,k)``.
This means that `X` has to be symmetric and that

```math
Xp + pX = X
```

must hold, where the `kwargs` can be used to check both for symmetrix of ``X```
and this equality up to a certain tolerance.
"""
function check_vector(
    M::Grassmann{n,k,𝔽},
    p::ProjectorPoint,
    X::ProjectorTVector;
    kwargs...,
) where {n,k,𝔽}
    if !isapprox(X.value, X.value'; kwargs...)
        return DomainError(
            norm(X.value - X.value'),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not symmetric.",
        )
    end
    if !isapprox(X.value * p.value + p.value * X.value, X.value; kwargs...)
        return DomainError(
            norm(X.value * p.value + p.value * X.value - X.value),
            "The matrix $(X) does not lie in the tangent space of $(p) on $(M), since X*p + p*X is not equal to X.",
        )
    end
    return nothing
end

@doc raw"""
    get_embedding(M::Grassmann{n,k,𝔽}, p::ProjectorPoint) where {n,k,𝔽}

return the embedding of the [`ProjectorPoint`](@ref) representation of the [`Grassmann`](@ref)
manifold, i.e. the Euclidean space ``\mathbb F^{n×n}``.
"""
get_embedding(::Grassmann{n,k,𝔽},::ProjectorPoint) where {n,k,𝔽} = Euclidean(n, n; field=𝔽)

@doc raw"""
    representation_size(M::Grassmann{n,k}, p::ProjectorPoint)

Return the represenation size or matrix dimension of a point on the [`Grassmann`](@ref)
`M` when using [`ProjectorPoint`§(@ref)s, i.e. ``(n,n)``.
"""
@generated representation_size(::Grassmann{n,k}, p::ProjectorPoint) where {n,k} = (n, n)


Base.show(io::IO, p::ProjectorPoint) = print(io, "ProjectorPoint($(p.value))")
Base.show(io::IO, X::ProjectorTVector) = print(io, "ProjectorTVector($(X.value))")