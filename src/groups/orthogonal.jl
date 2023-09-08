@doc raw"""
    Orthogonal{T} = GeneralUnitaryMultiplicationGroup{T,ℝ,AbsoluteDeterminantOneMatrices}

Orthogonal group $\mathrm{O}(n)$ represented by [`OrthogonalMatrices`](@ref).

# Constructor

    Orthogonal(n::Int; parameter::Symbol=:field)
"""
const Orthogonal{T} = GeneralUnitaryMultiplicationGroup{T,ℝ,AbsoluteDeterminantOneMatrices}

function Orthogonal(n::Int; parameter::Symbol=:field)
    return GeneralUnitaryMultiplicationGroup(OrthogonalMatrices(n; parameter=parameter))
end

function Base.show(io::IO, ::Orthogonal{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "Orthogonal($(n); parameter=:type)")
end
function Base.show(io::IO, M::Orthogonal{Tuple{Int}})
    n = get_n(M)
    return print(io, "Orthogonal($(n))")
end
