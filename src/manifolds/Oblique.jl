@doc raw"""
    Oblique{N,M,𝔽} <: AbstractPowerManifold{𝔽}

The oblique manifold $\mathcal{OB}(n,m)$ is the set of 𝔽-valued matrices with unit norm
column endowed with the metric from the embedding. This yields exactly the same metric as
considering the product metric of the unit norm vectors, i.e. `PowerManifold` of the
$(n-1)$-dimensional [`Sphere`](@ref).

The [`Sphere`](@ref) is stored internally within `M.manifold`, such that all functions of
`AbstractPowerManifold` can be used directly.

# Constructor

    Oblique(n,m)

Generate the manifold of matrices $\mathbb R^{n × m}$ such that the $m$ columns are unit
vectors, i.e. from the [`Sphere`](@ref)`(n-1)`.
"""
struct Oblique{N,M,𝔽,S} <:
       AbstractPowerManifold{𝔽,Sphere{S,𝔽},ArrayPowerRepresentation} where {N,M}
    manifold::Sphere{S,𝔽}
end

function Oblique(n::Int, m::Int, field::AbstractNumbers=ℝ)
    return Oblique{n,m,field,n - 1}(Sphere(n - 1, field))
end

Base.:^(M::Sphere{N,𝔽}, m::Int) where {N,𝔽} = Oblique{manifold_dimension(M) + 1,m,𝔽,N}(M)

@doc raw"""
    check_point(M::Oblique{n,m},p)

Checks whether `p` is a valid point on the [`Oblique`](@ref)`{m,n}` `M`, i.e. is a matrix
of `m` unit columns from $\mathbb R^{n}$, i.e. each column is a point from
[`Sphere`](@ref)`(n-1)`.
"""
check_point(::Oblique, ::Any)
function check_point(M::Oblique{n,m}, p; kwargs...) where {n,m}
    if size(p) != (n, m)
        return DomainError(
            length(p),
            "The matrix in `p` ($(size(p))) does not match the dimension of $(M).",
        )
    end
    return check_point(PowerManifold(M.manifold, m), p; kwargs...)
end
@doc raw"""
    check_vector(M::Oblique p, X; kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`Oblique`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and columnswise
a tangent vector to the columns of `p` on the [`Sphere`](@ref).
"""
function check_vector(M::Oblique{n,m}, p, X; kwargs...) where {n,m}
    if size(X) != (n, m)
        return DomainError(
            length(X),
            "The matrix `X` ($(size(X))) does not match the required dimension ($(representation_size(M))) for $(M).",
        )
    end
    return check_vector(PowerManifold(M.manifold, m), p, X; kwargs...)
end

get_iterator(::Oblique{n,m}) where {n,m} = Base.OneTo(m)

@generated function manifold_dimension(::Oblique{n,m,𝔽}) where {n,m,𝔽}
    return (n * real_dimension(𝔽) - 1) * m
end
power_dimensions(::Oblique{n,m}) where {n,m} = (m,)

@generated representation_size(::Oblique{n,m}) where {n,m} = (n, m)

@doc raw"""
    parallel_transport_to(M::Oblique, p, X, q)

Compute the parallel transport on the [`Oblique`](@ref) manifold by
doing a column wise parallel transport on the [`Sphere`](@ref)

"""
parallel_transport_to(::Oblique, p, X, q)

function Base.show(io::IO, ::Oblique{n,m,𝔽}) where {n,m,𝔽}
    return print(io, "Oblique($(n),$(m); field = $(𝔽))")
end
