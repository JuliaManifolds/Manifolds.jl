@doc raw"""
    Oblique{N,M} <: AbstractPowerManifold

The oblique manifold $\mathcal{OB}(n,m)$ is the set of real-valued matrices with unit norm
column endowed with the metric from the embedding. This yields exactly the same metric as
considering the product metric of the unit norm vectors, i.e. [`PowerManifold`](@ref) of the
$(n-1)$-dimensional [`Sphere`](@ref).

The [`Sphere`](@ref) is stored internally within `M.manifold`, such that all functions of
[`AbstractPowerManifold`](@ref) can be used directly.

# Constructor

    Oblique(n,m)

Generate the manifold of matrices $\mathbb R^{n × m}$ such that the $m$ columns are unit
vectors, i.e. from the [`Sphere`](@ref)`(n-1)`.
"""
struct Oblique{N,M} <: AbstractPowerManifold{Sphere{N},ArrayPowerRepresentation}
    manifold::Sphere{N}
end

Oblique(n::Int, m::Int) = Oblique{n - 1,m}(Sphere(n - 1))

^(M::Sphere{N}, m::Int) where {N} = Oblique{N,m}(M)

@doc raw"""
    check_manifold_point(M::Oblique{n,m},p)

Checks whether `p` is a valid point on the [`Oblique`](@ref)`{m,n}` `M`, i.e. is a matrix
of `m` unit columns from $\mathbb R^{n+1}$, i.e. each column is a point from
[`Sphere`](@ref)`(n)`.
"""
check_manifold_point(::Oblique, ::Any)
function check_manifold_point(M::Oblique{Ns,Ms}, p; kwargs...) where {Ns,Ms}
    if size(p) != (Ns + 1, Ms)
        return DomainError(
            length(p),
            "The matrix in `p` ($(size(p))) does not match the dimension of Oblique $((Ns,Ms)).",
        )
    end
    return check_manifold_point(PowerManifold(M.manifold, Ms), p; kwargs...)
end
@doc raw"""
    check_tangent_vector(M::Oblique p, X; check_base_point = true, kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`Oblique`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and columnswise
a tangent vector to the columns of `p` on the [`Sphere`](@ref).
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `p`.
"""
function check_tangent_vector(
    M::Oblique{Ns,Ms},
    p,
    X;
    check_base_point = true,
    kwargs...,
) where {Ns,Ms}
    if check_base_point && size(p) != (Ns + 1, Ms)
        return DomainError(
            length(p),
            "The matrix `p` ($(size(p))) does not match the dimension of Oblique $((Ns,Ms)).",
        )
    end
    if size(X) != (Ns + 1, Ms)
        return DomainError(
            length(X),
            "The matrix `X` ($(size(X))) does not match the dimension of Oblique $((Ns,Ms)).",
        )
    end
    return check_tangent_vector(
        PowerManifold(M.manifold, Ms),
        p,
        X;
        check_base_point = check_base_point,
        kwargs...,
    )
end

get_iterator(M::Oblique{Ns,Ms}) where {Ns,Ms} = 1:Ms

@generated manifold_dimension(::Oblique{N,M}) where {N,M} = (N) * M

@generated representation_size(::Oblique{N,M}) where {N,M} = (N + 1, M)

show(io::IO, ::Oblique{N,M}) where {N,M} = print(io, "Oblique($(N+1),$(M))")
