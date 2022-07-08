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
            norm(p.value - transpose(p.value)),
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
function check_size(M::Grassmann{n,k,𝔽}, p::ProjectorPoint; kwargs...) where {n,k,𝔽}
    return check_size(get_embedding(M, p), p.value; kwargs...)
end

@doc raw"""
    check_vector(::Grassmann{n,k,𝔽}, p::ProjectorPoint, X::ProjectorTVector; kwargs...) where {n,k,𝔽}

Check whether the [`ProjectorTVector`](@ref) `X` is from the tangent space ``T_p\operatorname{Gr}(n,k) ``
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

embed!(::Grassmann, q, p::ProjectorPoint) = copyto!(q, p.value)
embed!(::Grassmann, Y, p, X::ProjectorTVector) = copyto!(Y, X.value)
embed(::Grassmann, p::ProjectorPoint) = p.value
embed(::Grassmann, p, X::ProjectorTVector) = X.value

@doc raw"""
    get_embedding(M::Grassmann{n,k,𝔽}, p::ProjectorPoint) where {n,k,𝔽}

return the embedding of the [`ProjectorPoint`](@ref) representation of the [`Grassmann`](@ref)
manifold, i.e. the Euclidean space ``\mathbb F^{n×n}``.
"""
get_embedding(::Grassmann{n,k,𝔽}, ::ProjectorPoint) where {n,k,𝔽} = Euclidean(n, n; field=𝔽)

@doc raw"""
    representation_size(M::Grassmann{n,k}, p::ProjectorPoint)

Return the represenation size or matrix dimension of a point on the [`Grassmann`](@ref)
`M` when using [`ProjectorPoint`](@ref)s, i.e. ``(n,n)``.
"""
@generated representation_size(::Grassmann{n,k}, p::ProjectorPoint) where {n,k} = (n, n)

@doc raw"""
    canonical_project!(M::Grassmann{n,k}, q::ProjectorPoint, p)

Compute the canonical projection ``π(p)`` from the [`Stiefel`](@ref) manifold onto the [`Grassmann`](@ref)
manifold when represented as [`ProjectorPoint`](@ref), i.e.

```math
    π^{\mathrm{SG}}(p) = pp^{\mathrm{T}}
```
"""
function canonical_project!(::Grassmann{n,k}, q::ProjectorPoint, p) where {n,k}
    q.value .= p * p'
    return q
end
function canonical_project!(
    M::Grassmann{n,k},
    q::ProjectorPoint,
    p::StiefelPoint,
) where {n,k}
    return canonical_project!(M, q, p.value)
end

@doc raw"""
    canonical_project!(M::Grassmann{n,k}, q::ProjectorPoint, p)

Compute the canonical projection ``π(p)`` from the [`Stiefel`](@ref) manifold onto the [`Grassmann`](@ref)
manifold when represented as [`ProjectorPoint`](@ref), i.e.

```math
    Dπ^{\mathrm{SG}}(p)[X] = Xp^{\mathrm{T}} + pX^{\mathrm{T}}
```
"""
function differential_canonical_project!(
    ::Grassmann{n,k},
    Y::ProjectorTVector,
    p,
    X,
) where {n,k}
    copyto!(Y.value, X * p' + p * X')
    return Y
end
function differential_canonical_project!(
    M::Grassmann{n,k},
    Y::ProjectorPoint,
    p::StiefelPoint,
    X::StiefelTVector,
) where {n,k}
    return differenital_canonical_project!(M, Y, p.value, X.value)
end

@doc raw"""
    exp(M::Grassmann, p::ProjectorPoint, X::ProjectorTVector)

Compute the exponential map on the [`Grassmann`](@ref) as

```math
    \exp_pX = \operatorname{Exp}([X,p])p\operatorname{Exp}(-[X,p]),
```
where ``\operatorname{Exp}`` denotes the matrix exponential and ``[A,B] = AB-BA`` denotes the matrix commutator.

For details, see Proposition 3.2 in [^BendokatZimmermannAbsil2020].
"""
exp(M::Grassmann, p::ProjectorPoint, X::ProjectorTVector)

function exp!(::Grassmann, q::ProjectorPoint, p::ProjectorPoint, X::ProjectorTVector)
    xppx = X.value * p.value - p.value * X.value
    q.value .= exp(xppx) * p.value * exp(-xppx)
    return q
end
@doc raw"""
    horizontal_lift(N::Stiefel{n,k}, q, X::ProjectorTVector)

Compute the horizontal lift of `X` from the tangent space at ``p=π(q)``
on the [`Grassmann`](@ref) manifold, i.e.

```
Y = Xq ∈ T_q\mathrm{St}(n,k)
```

"""
horizontal_lift(::Stiefel, q, X::ProjectorTVector)

horizontal_lift!(::Stiefel, Y, q, X::ProjectorTVector) = copyto!(Y, X.value * q)

@doc raw"""
    parallel_transport_direction(
        M::Grassmann,
        p::ProjectorPoint,
        X::ProjectorTVector,
        d::ProjectorTVector
    )

Compute the parallel transport of `X` from the tangent space at `p` into direction `d`,
i.e. to ``q=\exp_pd``. The formula is given in Proposition 3.5 of [^BendokatZimmermannAbsil2020] as

```math
\mathcal{P}_{q ← p}(X) = \operatorname{Exp}([d,p])X\operatorname{Exp}(-[d,p]),
```

where ``\operatorname{Exp}`` denotes the matrix exponential and ``[A,B] = AB-BA`` denotes the matrix commutator.
"""
parallel_transport_direction(
    M::Grassmann,
    p::ProjectorPoint,
    X::ProjectorTVector,
    d::ProjectorTVector,
)

function parallel_transport_direction(
    ::Grassmann,
    Y::ProjectorTVector,
    p::ProjectorPoint,
    X::ProjectorTVector,
    d::ProjectorTVector,
)
    dppd = d.value * p.value - p.value * d.value
    return Y.value .= exp(dppd) * X.value * exp(-dppd)
end

Base.show(io::IO, p::ProjectorPoint) = print(io, "ProjectorPoint($(p.value))")
Base.show(io::IO, X::ProjectorTVector) = print(io, "ProjectorTVector($(X.value))")
