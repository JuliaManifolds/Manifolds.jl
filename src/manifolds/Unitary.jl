
@doc raw"""
    const UnitaryMatrices{n,𝔽} = AbstarctUnitaryMatrices{n,𝔽,AbsoluteDeterminantOneMatrices}

The manifold ``U(n,𝔽)`` of ``n×n`` complex matrices (when 𝔽=ℂ) or quaternionic matrices
(when 𝔽=ℍ) such that

``p^{\mathrm{H}}p = \mathrm{I}_n,``

where ``\mathrm{I}_n`` is the ``n×n`` identity matrix.
Such matrices `p` have a property that ``\lVert \det(p) \rVert = 1``.

The tangent spaces are given by

```math
    T_pU(n) \coloneqq \bigl\{
    X \big| pY \text{ where } Y \text{ is skew symmetric, i. e. } Y = -Y^{\mathrm{H}}
    \bigr\}
```

But note that tangent vectors are represented in the Lie algebra, i.e. just using ``Y`` in
the representation above.

# Constructor
    
    UnitaryMatrices(n, 𝔽::AbstractNumbers=ℂ)

see also [`OrthogonalMatrices`](@ref) for the real valued case.
"""
const UnitaryMatrices{n,𝔽} = GeneralUnitaryMatrices{n,𝔽,AbsoluteDeterminantOneMatrices}

UnitaryMatrices(n::Int, 𝔽::AbstractNumbers=ℂ) = UnitaryMatrices{n,𝔽}()

check_size(::UnitaryMatrices{1,ℍ}, p::Number) = nothing
check_size(::UnitaryMatrices{1,ℍ}, p, X::Number) = nothing

embed(::UnitaryMatrices{1,ℍ}, p::Number) = SMatrix{1,1}(p)

embed(::UnitaryMatrices{1,ℍ}, p, X::Number) = SMatrix{1,1}(X)

function exp(::UnitaryMatrices{1,ℍ}, p, X::Number)
    return p * exp(X)
end

function log(::UnitaryMatrices{1,ℍ}, p::Number, q::Number)
    return log(conj(p) * q)
end

project(::UnitaryMatrices{1,ℍ}, p) = normalize(p)

project(::UnitaryMatrices{1,ℍ}, p, X) = (X - conj(X)) / 2

show(io::IO, ::UnitaryMatrices{n,ℂ}) where {n} = print(io, "UnitaryMatrices($(n))")
show(io::IO, ::UnitaryMatrices{n,ℍ}) where {n} = print(io, "UnitaryMatrices($(n), ℍ)")
