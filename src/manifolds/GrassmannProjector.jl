@doc raw"""
    ProjectorPoint <: AbstractManifoldPoint

A type to represent points on a manifold [`Grassmann`](@ref) that are orthogonal projectors,
i.e. a matrix ``p ∈ \mathbb F^{n,n}`` projecting onto a ``k``-dimensional subspace.
"""
struct ProjectorPoint{T<:AbstractMatrix} <: AbstractManifoldPoint
    value::T
end

@doc raw"""
    ProjectorTangentVector <: AbstractTangentVector

A type to represent tangent vectors to points on a [`Grassmann`](@ref) manifold that are orthogonal projectors.
"""
struct ProjectorTangentVector{T<:AbstractMatrix} <: AbstractTangentVector
    value::T
end

ManifoldsBase.@manifold_vector_forwards ProjectorTangentVector value
ManifoldsBase.@manifold_element_forwards ProjectorPoint value

@doc raw"""
    check_point(::Grassmann, p::ProjectorPoint; kwargs...)

Check whether an orthogonal projector is a point from the [`Grassmann`](@ref)`(n,k)` manifold,
i.e. the [`ProjectorPoint`](@ref) ``p ∈ \mathbb F^{n×n}``, ``\mathbb F ∈ \{\mathbb R, \mathbb C\}``
has to fulfill ``p^{\mathrm{T}} = p``, ``p^2=p``, and ``\operatorname{rank} p = k`.
"""
function check_point(M::Grassmann, p::ProjectorPoint; kwargs...)
    n, k = get_parameter(M.size)
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
    check_size(M::Grassmann, p::ProjectorPoint; kwargs...)

Check that the [`ProjectorPoint`](@ref) is of correct size, i.e. from ``\mathbb F^{n×n}``
"""
function check_size(M::Grassmann, p::ProjectorPoint; kwargs...)
    return check_size(get_embedding(M, p), p.value; kwargs...)
end

@doc raw"""
    check_vector(::Grassmann, p::ProjectorPoint, X::ProjectorTangentVector; kwargs...)

Check whether the [`ProjectorTangentVector`](@ref) `X` is from the tangent space ``T_p\operatorname{Gr}(n,k) ``
at the [`ProjectorPoint`](@ref) `p` on the [`Grassmann`](@ref) manifold ``\operatorname{Gr}(n,k)``.
This means that `X` has to be symmetric and that

```math
Xp + pX = X
```

must hold, where the `kwargs` can be used to check both for symmetrix of ``X```
and this equality up to a certain tolerance.
"""
function check_vector(M::Grassmann, p::ProjectorPoint, X::ProjectorTangentVector; kwargs...)
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
embed!(::Grassmann, Y, p, X::ProjectorTangentVector) = copyto!(Y, X.value)
embed(::Grassmann, p::ProjectorPoint) = p.value
embed(::Grassmann, p, X::ProjectorTangentVector) = X.value

@doc raw"""
    get_embedding(M::Grassmann, p::ProjectorPoint)

Return the embedding of the [`ProjectorPoint`](@ref) representation of the [`Grassmann`](@ref)
manifold, i.e. the Euclidean space ``\mathbb F^{n×n}``.
"""
function get_embedding(
    ::Grassmann{TypeParameter{Tuple{n,k}},𝔽},
    ::ProjectorPoint,
) where {n,k,𝔽}
    return Euclidean(n, n; field=𝔽)
end
function get_embedding(M::Grassmann{Tuple{Int,Int},𝔽}, ::ProjectorPoint) where {𝔽}
    n, k = get_parameter(M.size)
    return Euclidean(n, n; field=𝔽, parameter=:field)
end

@doc raw"""
    representation_size(M::Grassmann, p::ProjectorPoint)

Return the represenation size or matrix dimension of a point on the [`Grassmann`](@ref)
`M` when using [`ProjectorPoint`](@ref)s, i.e. ``(n,n)``.
"""
function representation_size(M::Grassmann, p::ProjectorPoint)
    n, k = get_parameter(M.size)
    return (n, n)
end

@doc raw"""
    canonical_project!(M::Grassmann, q::ProjectorPoint, p)

Compute the canonical projection ``π(p)`` from the [`Stiefel`](@ref) manifold onto the [`Grassmann`](@ref)
manifold when represented as [`ProjectorPoint`](@ref), i.e.

```math
    π^{\mathrm{SG}}(p) = pp^{\mathrm{T}}
```
"""
function canonical_project!(::Grassmann, q::ProjectorPoint, p)
    q.value .= p * p'
    return q
end
function canonical_project!(M::Grassmann, q::ProjectorPoint, p::StiefelPoint)
    return canonical_project!(M, q, p.value)
end
function allocate_result(M::Grassmann, ::typeof(canonical_project), p::StiefelPoint)
    n, k = get_parameter(M.size)
    return ProjectorPoint(allocate(p.value, (n, n)))
end

@doc raw"""
    canonical_project!(M::Grassmann, q::ProjectorPoint, p)

Compute the canonical projection ``π(p)`` from the [`Stiefel`](@ref) manifold onto the [`Grassmann`](@ref)
manifold when represented as [`ProjectorPoint`](@ref), i.e.

```math
    Dπ^{\mathrm{SG}}(p)[X] = Xp^{\mathrm{T}} + pX^{\mathrm{T}}
```
"""
function differential_canonical_project!(::Grassmann, Y::ProjectorTangentVector, p, X)
    Xpt = X * p'
    Y.value .= Xpt .+ Xpt'
    return Y
end
function differential_canonical_project!(
    M::Grassmann,
    Y::ProjectorTangentVector,
    p::StiefelPoint,
    X::StiefelTangentVector,
)
    differential_canonical_project!(M, Y, p.value, X.value)
    return Y
end
function allocate_result(
    M::Grassmann,
    ::typeof(differential_canonical_project),
    p::StiefelPoint,
    X::StiefelTangentVector,
)
    n, k = get_parameter(M.size)
    return ProjectorTangentVector(allocate(p.value, (n, n)))
end
function allocate_result(M::Grassmann, ::typeof(differential_canonical_project), p, X)
    n, k = get_parameter(M.size)
    return ProjectorTangentVector(allocate(p, (n, n)))
end

@doc raw"""
    exp(M::Grassmann, p::ProjectorPoint, X::ProjectorTangentVector)

Compute the exponential map on the [`Grassmann`](@ref) as

```math
    \exp_pX = \operatorname{Exp}([X,p])p\operatorname{Exp}(-[X,p]),
```
where ``\operatorname{Exp}`` denotes the matrix exponential and ``[A,B] = AB-BA`` denotes the matrix commutator.

For details, see Proposition 3.2 in [BendokatZimmermannAbsil:2020](@cite).
"""
exp(M::Grassmann, p::ProjectorPoint, X::ProjectorTangentVector)

function exp!(::Grassmann, q::ProjectorPoint, p::ProjectorPoint, X::ProjectorTangentVector)
    xppx = X.value * p.value - p.value * X.value
    exp_xppx = exp(xppx)
    q.value .= exp_xppx * p.value / exp_xppx
    return q
end
@doc raw"""
    horizontal_lift(N::Stiefel{n,k}, q, X::ProjectorTangentVector)

Compute the horizontal lift of `X` from the tangent space at ``p=π(q)``
on the [`Grassmann`](@ref) manifold, i.e.

```math
Y = Xq ∈ T_q\mathrm{St}(n,k)
```

"""
horizontal_lift(::Stiefel, q, X::ProjectorTangentVector)

horizontal_lift!(::Stiefel, Y, q, X::ProjectorTangentVector) = copyto!(Y, X.value * q)

@doc raw"""
    parallel_transport_direction(
        M::Grassmann,
        p::ProjectorPoint,
        X::ProjectorTangentVector,
        d::ProjectorTangentVector
    )

Compute the parallel transport of `X` from the tangent space at `p` into direction `d`,
i.e. to ``q=\exp_pd``. The formula is given in Proposition 3.5 of [BendokatZimmermannAbsil:2020](@cite) as

```math
\mathcal{P}_{q ← p}(X) = \operatorname{Exp}([d,p])X\operatorname{Exp}(-[d,p]),
```

where ``\operatorname{Exp}`` denotes the matrix exponential and ``[A,B] = AB-BA`` denotes the matrix commutator.
"""
function parallel_transport_direction(
    M::Grassmann,
    p::ProjectorPoint,
    X::ProjectorTangentVector,
    d::ProjectorTangentVector,
)
    Y = allocate_result(M, vector_transport_direction, X, p, d)
    parallel_transport_direction!(M, Y, p, X, d)
    return Y
end

function parallel_transport_direction!(
    ::Grassmann,
    Y::ProjectorTangentVector,
    p::ProjectorPoint,
    X::ProjectorTangentVector,
    d::ProjectorTangentVector,
)
    dppd = d.value * p.value - p.value * d.value
    exp_dppd = exp(dppd)
    Y.value .= exp_dppd * X.value / exp_dppd
    return Y
end

Base.show(io::IO, p::ProjectorPoint) = print(io, "ProjectorPoint($(p.value))")
function Base.show(io::IO, X::ProjectorTangentVector)
    return print(io, "ProjectorTangentVector($(X.value))")
end
