@doc raw"""
    Orthogonal{T} = GeneralUnitaryMultiplicationGroup{T,ℝ,AbsoluteDeterminantOneMatrices}

Orthogonal group $\mathrm{O}(n)$ represented by [`OrthogonalMatrices`](@ref).

# Constructor

    Orthogonal(n::Int; parameter::Symbol=:type)
"""
const Orthogonal{T} = GeneralUnitaryMultiplicationGroup{T,ℝ,AbsoluteDeterminantOneMatrices}

function Orthogonal(n::Int; parameter::Symbol=:type)
    return GeneralUnitaryMultiplicationGroup(OrthogonalMatrices(n; parameter=parameter))
end

function Base.show(io::IO, ::Orthogonal{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "Orthogonal($(n))")
end
function Base.show(io::IO, M::Orthogonal{Tuple{Int}})
    n = get_parameter(M.manifold.size)[1]
    return print(io, "Orthogonal($(n); parameter=:field)")
end
