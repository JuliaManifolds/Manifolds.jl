@doc raw"""
    Torus{N} <: AbstractPowerManifold

The n-dimensional torus is the $n$-dimensional product of the [`Circle`](@ref).

The [`Circle`](@ref) is stored internally within `M.manifold`, such that all functions of
`AbstractPowerManifold` can be used directly.
"""
struct Torus{N} <: AbstractPowerManifold{ℝ,Circle{ℝ},ArrayPowerRepresentation}
    manifold::Circle{ℝ}
end

Torus(n::Int) = Torus{n}(Circle())

Base.:^(M::Circle, n::Int) = Torus{n}(M)

@doc raw"""
    check_point(M::Torus{n},p)

Checks whether `p` is a valid point on the [`Torus`](@ref) `M`, i.e. each of
its entries is a valid point on the [`Circle`](@ref) and the length of `x` is `n`.
"""
check_point(::Torus, ::Any)
function check_point(M::Torus{N}, p; kwargs...) where {N}
    if length(p) != N
        return DomainError(
            length(p),
            "The number of elements in `p` ($(length(p))) does not match the dimension of the torus ($(N)).",
        )
    end
    return check_point(PowerManifold(M.manifold, N), p; kwargs...)
end
@doc raw"""
    check_vector(M::Torus{n}, p, X; kwargs...)

Checks whether `X` is a valid tangent vector to `p` on the [`Torus`](@ref) `M`.
This means, that `p` is valid, that `X` is of correct dimension and elementwise
a tangent vector to the elements of `p` on the [`Circle`](@ref).
"""
function check_vector(M::Torus{N}, p, X; kwargs...) where {N}
    if length(X) != N
        return DomainError(
            length(X),
            "The number of elements in `X` ($(length(X))) does not match the dimension of the torus ($(N)).",
        )
    end
    return check_vector(PowerManifold(M.manifold, N), p, X; kwargs...)
end

get_iterator(::Torus{N}) where {N} = 1:N

manifold_dimension(::Torus{N}) where {N} = N

power_dimensions(::Torus{N}) where {N} = (N,)

representation_size(::Torus{N}) where {N} = (N,)

Base.show(io::IO, ::Torus{N}) where {N} = print(io, "Torus($(N))")
