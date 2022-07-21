
@doc raw"""
    const QuaternionicUnitaryMatrices{n} = AbstarctUnitaryMatrices{n,ℍ,AbsoluteDeterminantOneMatrices}

The manifold ``U(n, ℍ)`` of ``n×n`` quaterninic matrices such that

``p^* p = \mathrm{I}_n,``

where ``\mathrm{I}_n`` is the ``n×n`` identity matrix and ``p^*`` is the quaterninic
conjugate of ``p``.
Such matrices `p` have a property that ``\lVert \det(p) \rVert = 1``.

The tangent spaces are given by

```math
    T_pU(n, ℍ) \coloneqq \bigl\{
    X \big| pY \text{ where } Y \text{ is skew symmetric, i. e. } Y = -Y^*
    \bigr\}
```

But note that tangent vectors are represented in the Lie algebra, i.e. just using ``Y`` in the representation above.

# Constructor
    QuaternionicUnitaryMatrices(n)

See also [`OrthogonalMatrices`](@ref) for the real valued case and [`UnitaryMatrices`](@ref)
for the complex valued case.
"""
const QuaternionicUnitaryMatrices{n} =
    GeneralUnitaryMatrices{n,ℍ,AbsoluteDeterminantOneMatrices}

QuaternionicUnitaryMatrices(n::Int) = QuaternionicUnitaryMatrices{n}()

function show(io::IO, ::QuaternionicUnitaryMatrices{n}) where {n}
    return print(io, "QuaternionicUnitaryMatrices($(n))")
end

check_size(::QuaternionicUnitaryMatrices{1}, p::Number) = nothing
check_size(::QuaternionicUnitaryMatrices{1}, p, X::Number) = nothing

embed(::QuaternionicUnitaryMatrices{1}, p::Number) = SMatrix{1,1}(p)

embed(::QuaternionicUnitaryMatrices{1}, p, X::Number) = SMatrix{1,1}(X)

function exp(::QuaternionicUnitaryMatrices{1}, p, X::Number)
    return p * exp(X)
end

function log(::QuaternionicUnitaryMatrices{1}, p::Number, q::Number)
    return log(conj(p) * q)
end

project(::QuaternionicUnitaryMatrices{1}, p) = normalize(p)

project(::QuaternionicUnitaryMatrices{1}, p, X) = (X - conj(X)) / 2
