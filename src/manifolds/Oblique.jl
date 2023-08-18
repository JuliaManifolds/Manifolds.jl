@doc raw"""
    Oblique{T,𝔽,S} <: AbstractPowerManifold{𝔽}

The oblique manifold $\mathcal{OB}(n,m)$ is the set of 𝔽-valued matrices with unit norm
column endowed with the metric from the embedding. This yields exactly the same metric as
considering the product metric of the unit norm vectors, i.e. [`PowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.PowerManifold) of the
$(n-1)$-dimensional [`Sphere`](@ref).

The [`Sphere`](@ref) is stored internally within `M.manifold`, such that all functions of
[`AbstractPowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractPowerManifold)  can be used directly.

# Constructor

    Oblique(n::Int, m::Int, field::AbstractNumbers=ℝ; parameter::Symbol=:field)

Generate the manifold of matrices $\mathbb R^{n × m}$ such that the $m$ columns are unit
vectors, i.e. from the [`Sphere`](@ref)`(n-1)`.
"""
struct Oblique{T,𝔽,S} <: AbstractPowerManifold{𝔽,Sphere{S,𝔽},ArrayPowerRepresentation}
    size::T
    manifold::Sphere{S,𝔽}
end

function Oblique(n::Int, m::Int, field::AbstractNumbers=ℝ; parameter::Symbol=:field)
    sphere = Sphere(n - 1, field; parameter=parameter)
    size = wrap_type_parameter(parameter, (n, m))
    return Oblique{typeof(size),field,typeof(sphere).parameters[1]}(size, sphere)
end

function Base.:^(::Sphere{TypeParameter{Tuple{N}},𝔽}, m::Int) where {N,𝔽}
    return Oblique(N + 1, m, 𝔽; parameter=:type)
end
function Base.:^(M::Sphere{Tuple{Int},𝔽}, m::Int) where {𝔽}
    N = M.size[1]
    return Oblique(N + 1, m, 𝔽)
end

@doc raw"""
    check_point(M::Oblique, p)

Checks whether `p` is a valid point on the [`Oblique`](@ref)`{m,n}` `M`, i.e. is a matrix
of `m` unit columns from $\mathbb R^{n}$, i.e. each column is a point from
[`Sphere`](@ref)`(n-1)`.
"""
check_point(::Oblique, ::Any)
function check_point(M::Oblique, p; kwargs...)
    n, m = get_nm(M)
    return check_point(PowerManifold(M.manifold, m), p; kwargs...)
end
@doc raw"""
    check_vector(M::Oblique p, X; kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`Oblique`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and columnswise
a tangent vector to the columns of `p` on the [`Sphere`](@ref).
"""
function check_vector(M::Oblique, p, X; kwargs...)
    n, m = get_nm(M)
    return check_vector(PowerManifold(M.manifold, m), p, X; kwargs...)
end

get_iterator(M::Oblique) = Base.OneTo(get_nm(M)[2])

get_nm(::Oblique{TypeParameter{Tuple{n,m}}}) where {n,m} = (n, m)
get_nm(M::Oblique{Tuple{Int,Int}}) = get_parameter(M.size)

function manifold_dimension(M::Oblique{<:Any,𝔽}) where {𝔽}
    n, m = get_nm(M)
    return (n * real_dimension(𝔽) - 1) * m
end
power_dimensions(M::Oblique) = get_nm(M)[2]

representation_size(M::Oblique) = get_nm(M)

@doc raw"""
    parallel_transport_to(M::Oblique, p, X, q)

Compute the parallel transport on the [`Oblique`](@ref) manifold by
doing a column wise parallel transport on the [`Sphere`](@ref)

"""
parallel_transport_to(::Oblique, p, X, q)

function Base.show(io::IO, ::Oblique{TypeParameter{Tuple{n,m}},𝔽}) where {n,m,𝔽}
    return print(io, "Oblique($(n), $(m); field = $(𝔽), parameter=:type)")
end
function Base.show(io::IO, M::Oblique{Tuple{Int,Int},𝔽}) where {𝔽}
    n, m = get_nm(M)
    return print(io, "Oblique($(n), $(m); field = $(𝔽))")
end
