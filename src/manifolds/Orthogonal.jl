@doc raw"""
     OrthogonalMatrices{n} = GeneralUnitaryMatrices{n,ℝ,AbsoluteDeterminantOneMatrices}

The manifold of (real) orthogonal matrices ``\mathrm{O}(n)``.

    OrthogonalMatrices(n)
"""
const OrthogonalMatrices{n} = GeneralUnitaryMatrices{n,ℝ,AbsoluteDeterminantOneMatrices}

function OrthogonalMatrices(n::Int; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, (n,))
    return OrthogonalMatrices{typeof(size)}(size)
end

function Random.rand!(
    rng::AbstractRNG,
    M::OrthogonalMatrices,
    pX;
    vector_at=nothing,
    σ::Real=one(eltype(pX)),
)
    if vector_at === nothing
        # Special case: Rotations(1) is just zero-dimensional
        (manifold_dimension(M) == 0) && return fill!(pX, rand(rng, (-1, 1)))
        A = randn(rng, representation_size(M))
        s = diag(sign.(qr(A).R))
        D = Diagonal(s)
        pX .= qr(A).Q * D
    else
        # Special case: Rotations(1) is just zero-dimensional
        (manifold_dimension(M) == 0) && return fill!(pX, 0)
        A = σ .* randn(rng, representation_size(M))
        pX .= triu(A, 1) .- transpose(triu(A, 1))
        normalize!(pX)
    end
    return pX
end

function Base.show(io::IO, ::OrthogonalMatrices{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "OrthogonalMatrices($(n))")
end
function Base.show(io::IO, M::OrthogonalMatrices{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return print(io, "OrthogonalMatrices($n; parameter=:field)")
end
