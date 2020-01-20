@doc doc"""
    Torus{N} <: AbstractPowerManifold

The n-dimensionsl torus is the $n$-dimensional product of the [´Circle`](@ref).

The Circle is stored internally within `manifold`, such that all functions of
[`AbstractPowerManifold`](@ref) can be used directly.
"""
struct Torus{N} <: AbstractPowerManifold{Circle{ℝ},MultidimentionalArrayPowerRepresentation}
    manifold::Circle{ℝ}
end

Torus(n::Int) = Torus{n}(Circle())

@doc doc"""
    check_manifold_point(M::Torus{n},x)

check whether `x` is a valid point on the [`GraphManifold`](@ref)
"""
check_manifold_point(::Torus, ::Any)
function check_manifold_point(M::Torus{N}, x; kwargs...) where {N}
    if length(x) != N
        return DomainError(
            length(x),
            "The number of elements in `x` ($(length(x))) does not match the dimension of the torus ($(N)).",
        )
    end
    return check_manifold_point(PowerManifold(M.manifold, N), x; kwargs...)
end

function check_tangent_vector(M::Torus{N}, x, v; kwargs...) where {N}
    if length(x) != N
        return DomainError(
            length(x),
            "The number of elements in `x` ($(length(x))) does not match the dimension of the torus ($(N)).",
        )
    end
    if length(v) != N
        return DomainError(
            length(v),
            "The number of elements in `v` ($(length(v))) does not match the dimension of the torus ($(N)).",
        )
    end
    return check_tangent_vector(PowerManifold(M.manifold, N), x, v; kwargs...)
end

get_iterator(M::Torus{N}) where {N} = 1:N

@generated manifold_dimension(::Torus{N}) where {N} = N

@generated representation_size(::Torus{N}) where {N} = (N,)
