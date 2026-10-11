@doc raw"""
    Oblique{𝔽, T, S} <: AbstractPowerManifold{𝔽}

The oblique manifold ``\mathcal{OB}(n,m)`` is the set of 𝔽-valued matrices with unit norm
column endowed with the metric from the embedding. This yields exactly the same metric as
considering the product metric of the unit norm vectors, i.e. [`PowerManifold`](@extref `ManifoldsBase.PowerManifold`) of the
``(n-1)``-dimensional [`Sphere`](@ref).

The [`Sphere`](@ref) is stored internally within `M.manifold`, such that all functions of
[`AbstractPowerManifold`](@extref `ManifoldsBase.AbstractPowerManifold`)  can be used directly.

# Constructor

    Oblique(n::Int, m::Int, field::AbstractNumbers=ℝ; parameter::Symbol=:type)

Generate the manifold of matrices ``\mathbb R^{n×m}`` such that the ``m`` columns are unit
vectors, i.e. from the [`Sphere`](@ref)`(n-1)`.
"""
struct Oblique{𝔽, T, S} <: AbstractPowerManifold{𝔽, Sphere{𝔽, S}, ArrayPowerRepresentation}
    size::T
    manifold::Sphere{𝔽, S}
end

function Oblique(n::Int, m::Int, field::AbstractNumbers = ℝ; parameter::Symbol = :type)
    sphere = Sphere(n - 1, field; parameter = parameter)
    size = wrap_type_parameter(parameter, (n, m))
    return Oblique{field, typeof(size), typeof(sphere).parameters[2]}(size, sphere)
end

function Base.:^(::Sphere{𝔽, TypeParameter{Tuple{N}}}, m::Int) where {N, 𝔽}
    return Oblique(N + 1, m, 𝔽)
end
function Base.:^(M::Sphere{𝔽, Tuple{Int}}, m::Int) where {𝔽}
    N = M.size[1]
    return Oblique(N + 1, m, 𝔽; parameter = :field)
end

@doc raw"""
    check_point(M::Oblique, p)

Checks whether `p` is a valid point on the [`Oblique`](@ref)`{m,n}` `M`, i.e. is a matrix
of `m` unit columns from ``\mathbb R^{n}``, i.e. each column is a point from
[`Sphere`](@ref)`(n-1)`.
"""
check_point(::Oblique, ::Any)
function check_point(M::Oblique, p; kwargs...)
    n, m = get_parameter(M.size)
    return check_point(PowerManifold(M.manifold, m), p; kwargs...)
end
@doc raw"""
    check_vector(M::Oblique p, X; kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`Oblique`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and columnswise
a tangent vector to the columns of `p` on the [`Sphere`](@ref).
"""
function check_vector(M::Oblique, p, X; kwargs...)
    n, m = get_parameter(M.size)
    return check_vector(PowerManifold(M.manifold, m), p, X; kwargs...)
end

get_iterator(M::Oblique) = Base.OneTo(get_parameter(M.size)[2])

@doc raw"""
    get_embedding(M::Oblique)

Return the embedding of the [`Oblique`](@ref) manifold `M`, the [`Euclidean`](@ref) ``𝔽^{n×m}``.
"""
function get_embedding(M::Oblique{𝔽}) where {𝔽}
    n, m = get_parameter(M.size)
    return Euclidean(n, m; field = 𝔽, parameter = :field)
end

function manifold_dimension(M::Oblique{𝔽}) where {𝔽}
    n, m = get_parameter(M.size)
    return (n * real_dimension(𝔽) - 1) * m
end
power_dimensions(M::Oblique) = (get_parameter(M.size)[2],)

representation_size(M::Oblique) = get_parameter(M.size)

@doc raw"""
    parallel_transport_to(M::Oblique, p, X, q)

Compute the parallel transport on the [`Oblique`](@ref) manifold by
doing a column wise parallel transport on the [`Sphere`](@ref)

"""
parallel_transport_to(::Oblique, p, X, q)

function Base.show(io::IO, ::Oblique{𝔽, TypeParameter{Tuple{n, m}}}) where {n, m, 𝔽}
    return print(io, "Oblique($(n), $(m), $(𝔽))")
end
function Base.show(io::IO, M::Oblique{𝔽, Tuple{Int, Int}}) where {𝔽}
    n, m = get_parameter(M.size)
    return print(io, "Oblique($(n), $(m), $(𝔽); parameter=:field)")
end
