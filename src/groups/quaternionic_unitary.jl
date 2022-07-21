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
    q[1] = exp(X[1])
    return q
end

function log_lie(::QuaternionicUnitary{1}, q::Number)
    return log(q)
end

function log_lie!(::QuaternionicUnitary{1}, X, p)
    X[1] = log(p[1])
    return X
end

Base.inv(::QuaternionicUnitary, p) = adjoint(p)

show(io::IO, ::QuaternionicUnitary{n}) where {n} = print(io, "QuaternionicUnitary($(n))")
