@doc raw"""
     Unitary{n,𝔽} = GeneralUnitaryMultiplicationGroup{n,𝔽,AbsoluteDeterminantOneMatrices}

The group of unitary matrices ``\mathrm{U}(n, 𝔽)``, either complex (when 𝔽=ℂ) or quaternionic
(when 𝔽=ℍ)

The group consists of all points ``p ∈ 𝔽^{n × n}`` where ``p^{\mathrm{H}}p = pp^{\mathrm{H}} = I``.

The tangent spaces are if the form

```math
T_p\mathrm{U}(n) = \bigl\{ X \in 𝔽^{n×n} \big| X = pY \text{ where } Y = -Y^{\mathrm{H}} \bigr\}
```

and we represent tangent vectors by just storing the [`SkewHermitianMatrices`](@ref) ``Y``,
or in other words we represent the tangent spaces employing the Lie algebra ``\mathfrak{u}(n, 𝔽)``.

Quaternionic unitary group is isomorphic to the compact symplectic group of the same dimension.

# Constructor

    Unitary(n, 𝔽::AbstractNumbers=ℂ)

Construct ``\mathrm{U}(n, 𝔽)``.
See also [`Orthogonal(n)`](@ref) for the real-valued case.
"""
const Unitary{n,𝔽} = GeneralUnitaryMultiplicationGroup{n,𝔽,AbsoluteDeterminantOneMatrices}

function Unitary(n, 𝔽::AbstractNumbers=ℂ; parameter::Symbol=:type)
    return GeneralUnitaryMultiplicationGroup(UnitaryMatrices(n, 𝔽; parameter=parameter))
end

@doc raw"""
    exp_lie(G::Unitary{TypeParameter{Tuple{2}},ℂ}, X)

Compute the group exponential map on the [`Unitary(2)`](@ref) group, which is

```math
\exp_e \colon X ↦ e^{\operatorname{tr}(X) / 2} \left(\cos θ I + \frac{\sin θ}{θ} \left(X - \frac{\operatorname{tr}(X)}{2} I\right)\right),
```

where ``θ = \frac{1}{2} \sqrt{4\det(X) - \operatorname{tr}(X)^2}``.
"""
exp_lie(::Unitary{TypeParameter{Tuple{2}},ℂ}, X)

function exp_lie(::Unitary{TypeParameter{Tuple{1}},ℍ}, X::Number)
    return exp(X)
end

function exp_lie!(::Unitary{TypeParameter{Tuple{1}}}, q, X)
    q[] = exp(X[])
    return q
end

function exp_lie!(::Unitary{TypeParameter{Tuple{2}},ℂ}, q, X)
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

function log_lie!(::Unitary{TypeParameter{Tuple{1}}}, X, p)
    X[] = log(p[])
    return X
end
function log_lie!(::Unitary{TypeParameter{Tuple{1}}}, X::AbstractMatrix, p::AbstractMatrix)
    X[] = log(p[])
    return X
end
function log_lie!(G::Unitary, X, p)
    log_safe!(X, p)
    project!(G, X, Identity(G), X)
    return X
end

identity_element(::Unitary{TypeParameter{Tuple{1}},ℍ}) = Quaternions.quat(1.0)

function log_lie(::Unitary{TypeParameter{Tuple{1}}}, q::Number)
    return log(q)
end

Base.inv(::Unitary, p) = adjoint(p)

function Base.show(io::IO, ::Unitary{TypeParameter{Tuple{n}},ℂ}) where {n}
    return print(io, "Unitary($(n))")
end
function Base.show(io::IO, M::Unitary{Tuple{Int},ℂ})
    n = get_parameter(M.manifold.size)[1]
    return print(io, "Unitary($(n); parameter=:field)")
end
function Base.show(io::IO, ::Unitary{TypeParameter{Tuple{n}},ℍ}) where {n}
    return print(io, "Unitary($(n), ℍ)")
end
function Base.show(io::IO, M::Unitary{Tuple{Int},ℍ})
    n = get_parameter(M.manifold.size)[1]
    return print(io, "Unitary($(n), ℍ; parameter=:field)")
end
