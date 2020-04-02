@doc raw"""
    Oblique{N,M,𝔽} <: AbstractPowerManifold{𝔽}

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
struct Oblique{N,M,𝔽,S} <: AbstractPowerManifold{𝔽,Sphere{S,𝔽},ArrayPowerRepresentation} where {N,M}
    manifold::Sphere{S,𝔽}
end

Oblique(n::Int, m::Int, field::AbstractNumbers = ℝ) = Oblique{n,m,field,n-1}(Sphere(n - 1,field))

^(M::Sphere{N,𝔽}, m::Int) where {N,𝔽} = Oblique{manifold_dimension(M)+1,m,𝔽,N}(M)

@doc raw"""
    check_manifold_point(M::Oblique{n,m},p)

Checks whether `p` is a valid point on the [`Oblique`](@ref)`{m,n}` `M`, i.e. is a matrix
of `m` unit columns from $\mathbb R^{n+1}$, i.e. each column is a point from
[`Sphere`](@ref)`(n)`.
"""
check_manifold_point(::Oblique, ::Any)
function check_manifold_point(M::Oblique{n,m}, p; kwargs...) where {n,m}
    if size(p) != (n, m)
        return DomainError(
            length(p),
            "The matrix in `p` ($(size(p))) does not match the dimension of Oblique $((n,m)).",
        )
    end
    return check_manifold_point(PowerManifold(M.manifold, m), p; kwargs...)
end
@doc raw"""
    check_tangent_vector(M::Oblique p, X; check_base_point = true, kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`Oblique`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and columnswise
a tangent vector to the columns of `p` on the [`Sphere`](@ref).
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `p`.
"""
function check_tangent_vector(
    M::Oblique{n,m},
    p,
    X;
    check_base_point = true,
    kwargs...,
) where {n,m}
    if check_base_point && size(p) != (n, m)
        return DomainError(
            length(p),
            "The matrix `p` ($(size(p))) does not match the dimension of Oblique $((n,m)).",
        )
    end
    if size(X) != (n, m)
        return DomainError(
            length(X),
            "The matrix `X` ($(size(X))) does not match the dimension of Oblique $((n,m)).",
        )
    end
    return check_tangent_vector(
        PowerManifold(M.manifold, m),
        p,
        X;
        check_base_point = check_base_point,
        kwargs...,
    )
end

get_iterator(M::Oblique{n,m}) where {n,m} = 1:m

@generated manifold_dimension(::Oblique{n,m,𝔽}) where {n,m,𝔽} = (n-1) * m * real_dimension(𝔽)

@generated representation_size(::Oblique{n,m}) where {n,m} = (n, m)

show(io::IO, ::Oblique{n,m}) where {n,m} = print(io, "Oblique($(n),$(m))")
