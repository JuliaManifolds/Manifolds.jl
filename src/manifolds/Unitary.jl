
@doc raw"""
    const UnitaryMatrices{n} = AbstarctUnitaryMatrices{n,ℂ,AbsoluteDeterminantOneMatrices}

The manifold ``U(n)`` of ``n×n`` complex matrices such that

``p^{\mathrm{H}p = \mathrm{I}_n,``

where ``\mathrm{I}_n`` is the ``n×n`` identity matrix.
Such matrices `p` have a property that ``\lVert \det(p) \rVert = 1``.

The tangent spaces are given by

```math
    T_pU(n) \coloneqq \bigl\{
    X \big| pY \text{ where } Y \text{ is skew symmetric, i. e. } Y = -Y^{\mathrm{H}}
    \bigr\}
```

But note that tangent vectors are represented in the Lie algebra, i.e. just using ``Y`` in the representation above.

# Constructor
     UnitaryMatrices(n)

see also [`OrthogonalMatrices`](@ref) for the real valued case.
"""
const UnitaryMatrices{n} = GeneralUnitaryMatrices{n,ℂ,AbsoluteDeterminantOneMatrices}

UnitaryMatrices(n::Int) = UnitaryMatrices{n}()

show(io::IO, ::UnitaryMatrices{n}) where {n} = print(io, "UnitaryMatrices($(n))")
