@doc raw"""
     OrthogonalMatrices{n} = GeneralUnitaryMatrices{n,ℝ,AbsoluteDeterminantOneMatrices}

The manifold of (real) orthogonal matrices ``\mathrm{O}(n)``.

The tangent space at ``p`` are all matrices of the form ``X = pY``, where ``Y```
is skew symmetric.

If you prefer to use the representation of tangent vectors in the Lie algebra, see
the [`Orthogonal`](@ref)`(n)` group.

# Constructor

   OrthogonalMatrices(n)

Generate the ``n × x`` orthognoal matrices manifold.

## See also
[`Rotations`](@ref), [`UnitaryMatrices`](@ref), [`Orthogonal`](@ref).
"""
const OrthogonalMatrices{n} = GeneralUnitaryMatrices{n,ℝ,AbsoluteDeterminantOneMatrices}

OrthogonalMatrices(n) = OrthogonalMatrices{n}()

function Random.rand!(M::OrthogonalMatrices, pX; vector_at=nothing, σ::Real=one(eltype(pX)))
    if vector_at === nothing
        # Special case: Rotations(1) is just zero-dimensional
        (manifold_dimension(M) == 0) && return fill!(pX, rand((-1, 1)))
        A = randn(representation_size(M))
        s = diag(sign.(qr(A).R))
        D = Diagonal(s)
        pX .= qr(A).Q * D
    else
        # Special case: Rotations(1) is just zero-dimensional
        (manifold_dimension(M) == 0) && return fill!(pX, 0)
        A = σ .* randn(representation_size(M))
        pX .= triu(A, 1) .- transpose(triu(A, 1))
        normalize!(pX)
    end
    return pX
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

function Base.show(io::IO, ::OrthogonalMatrices{n}) where {n}
    return print(io, "OrthogonalMatrices($(n))")
end
