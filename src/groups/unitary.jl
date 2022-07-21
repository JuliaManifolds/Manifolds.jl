@doc raw"""
     Unitary{n} = GeneralUnitaryMultiplicationGroup{n,ℂ,AbsoluteDeterminantOneMatrices}

The group of unitary matrices ``\mathrm{U}(n)``.

The group consists of all points ``p ∈ \mathbb C^{n × n}`` where ``p^\mathrm{H}p = pp^\mathrm{H} = I``.

The tangent spaces are if the form

```math
T_p\mathrm{U}(n) = \bigl\{ X \in \mathbb C^{n×n} \big| X = pY \text{ where } Y = -Y^{\mathrm{H}} \bigr\}
```

and we represent tangent vectors by just storing the [`SkewHermitianMatrices`](@ref) ``Y``,
or in other words we represent the tangent spaces employing the Lie algebra ``\mathfrak{u}(n)``.

# Constructor

    Unitary(n)

Construct ``\mathrm{U}(n)``.
See also [`Orthogonal(n)`](@ref) for the real-valued case.
"""
const Unitary{n} = GeneralUnitaryMultiplicationGroup{n,ℂ,AbsoluteDeterminantOneMatrices}

Unitary(n) = Unitary{n}(UnitaryMatrices(n))

@doc raw"""
    exp_lie(G::Unitary{2}, X)

Compute the group exponential map on the [`Unitary(2)`](@ref) group, which is

```math
\exp_e \colon X ↦ e^{\operatorname{tr}(X) / 2} \left(\cos θ I + \frac{\sin θ}{θ} \left(X - \frac{\operatorname{tr}(X)}{2} I\right)\right),
```

where ``θ = \frac{1}{2} \sqrt{4\det(X) - \operatorname{tr}(X)^2}``.
 """
exp_lie(::Unitary{2}, X)

function exp_lie!(::Unitary{1}, q, X)
    q[1] = exp(X[1])
    return q
end

function exp_lie!(::Unitary{2}, q, X)
    size(X) === (2, 2) && size(q) === (2, 2) || throw(DomainError())
    @inbounds a, d = imag(X[1, 1]), imag(X[2, 2])
    @inbounds b = (X[2, 1] - X[1, 2]') / 2
    θ = hypot((a - d) / 2, abs(b))
    sinθ, cosθ = sincos(θ)
    usincθ = ifelse(iszero(θ), one(sinθ) / one(θ), sinθ / θ)
    s = (a + d) / 2
    ciss = cis(s)
    α = ciss * complex(cosθ, -s * usincθ)
    β = ciss * usincθ
    @inbounds begin
        q[1, 1] = β * (im * a) + α
        q[2, 1] = β * b
        q[1, 2] = β * -b'
        q[2, 2] = β * (im * d) + α
    end
    return q
end

function exp_lie!(G::Unitary, q, X)
    copyto!(G, q, exp(X))
    return q
end

function log_lie!(::Unitary{1}, X, p)
    X[1] = log(p[1])
    return X
end
function log_lie!(G::Unitary, X, p)
    log_safe!(X, p)
    project!(G, X, Identity(G), X)
    return X
end

Base.inv(::Unitary, p) = adjoint(p)

show(io::IO, ::Unitary{n}) where {n} = print(io, "Unitary($(n))")
