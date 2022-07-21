@doc raw"""
    QuaternionicUnitary{n} = GeneralUnitaryMultiplicationGroup{n,ℍ,AbsoluteDeterminantOneMatrices}

The group of unitary matrices ``\mathrm{U}(n, ℍ)``.

The group consists of all points ``p ∈ \mathbb H^{n × n}`` where ``p^*p = pp^* = I``,
and ``p^*`` is quaternionic conjugation.

The tangent spaces are if the form

```math
T_p\mathrm{U}(n, ℍ) = \bigl\{ X \in \mathbb C^{n×n} \big| X = pY \text{ where } Y = -Y^* \bigr\}
```

and we represent tangent vectors by just storing the quaternionic [`SkewHermitianMatrices`](@ref) ``Y``,
or in other words we represent the tangent spaces employing the Lie algebra ``\mathfrak{u}(n, ℍ)``.

# Constructor

    QuaternionicUnitary(n)

Construct ``\mathrm{U}(n, ℍ)``.
See also [`Orthogonal(n)`](@ref) for the real-valued case and [`Unitary(n)`](@ref) for
the complex-valued case.

"""
const QuaternionicUnitary{n} =
    GeneralUnitaryMultiplicationGroup{n,ℍ,AbsoluteDeterminantOneMatrices}

QuaternionicUnitary(n) = QuaternionicUnitary{n}(QuaternionicUnitaryMatrices(n))

function exp_lie(::QuaternionicUnitary{1}, X::Number)
    return exp(X)
end

function exp_lie!(::QuaternionicUnitary{1}, q, X)
    q[] = exp(X[])
    return q
end

identity_element(::QuaternionicUnitary{1}) = Quaternion(1.0)

Base.inv(::QuaternionicUnitary, p) = adjoint(p)

function log_lie(::QuaternionicUnitary{1}, q::Number)
    return log(q)
end

function log_lie!(::QuaternionicUnitary{1}, X, p)
    X[] = log(p[])
    return X
end

show(io::IO, ::QuaternionicUnitary{n}) where {n} = print(io, "QuaternionicUnitary($(n))")
