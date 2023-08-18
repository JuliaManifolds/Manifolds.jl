@doc raw"""
    Orthogonal{n} = GeneralUnitaryMultiplicationGroup{n,ℝ,AbsoluteDeterminantOneMatrices}

Orthogonal group $\mathrm{O}(n)$ represented by [`OrthogonalMatrices`](@ref).

# Constructor

    Orthogonal(n)
"""
const Orthogonal{n} = GeneralUnitaryMultiplicationGroup{n,ℝ,AbsoluteDeterminantOneMatrices}

function Orthogonal(n; parameter::Symbol=:field)
    return GeneralUnitaryMultiplicationGroup(OrthogonalMatrices(n; parameter=parameter))
end

function Base.show(io::IO, ::Orthogonal{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "Orthogonal($(n); parameter=:type)")
end
function Base.show(io::IO, M::Orthogonal{Tuple{Int}})
    n = get_n(M)
    return print(io, "Orthogonal($(n))")
end
