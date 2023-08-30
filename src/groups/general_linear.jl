@doc raw"""
    GeneralLinear{n,𝔽} <:
        AbstractDecoratorManifold{𝔽}

The general linear group, that is, the group of all invertible matrices in ``𝔽^{n×n}``.

The default metric is the left-``\mathrm{GL}(n)``-right-``\mathrm{O}(n)``-invariant metric
whose inner product is
```math
⟨X_p,Y_p⟩_p = ⟨p^{-1}X_p,p^{-1}Y_p⟩_\mathrm{F} = ⟨X_e, Y_e⟩_\mathrm{F},
```
where ``X_p, Y_p ∈ T_p \mathrm{GL}(n, 𝔽)``,
``X_e = p^{-1}X_p ∈ 𝔤𝔩(n) = T_e \mathrm{GL}(n, 𝔽) = 𝔽^{n×n}`` is the corresponding
vector in the Lie algebra, and ``⟨⋅,⋅⟩_\mathrm{F}`` denotes the Frobenius inner product.

By default, tangent vectors ``X_p`` are represented with their corresponding Lie algebra
vectors ``X_e = p^{-1}X_p``.
"""
struct GeneralLinear{n,𝔽} <: AbstractDecoratorManifold{𝔽} end

function active_traits(f, ::GeneralLinear, args...)
    return merge_traits(
        IsGroupManifold(MultiplicationOperation()),
        IsEmbeddedManifold(),
        HasLeftInvariantMetric(),
        IsDefaultMetric(EuclideanMetric()),
    )
end

GeneralLinear(n, 𝔽::AbstractNumbers=ℝ) = GeneralLinear{n,𝔽}()

function allocation_promotion_function(::GeneralLinear{n,ℂ}, f, ::Tuple) where {n}
    return complex
end

function check_point(G::GeneralLinear, p; kwargs...)
    detp = det(p)
    if iszero(detp)
        return DomainError(
            detp,
            "The matrix $(p) does not lie on $(G), since it is not invertible.",
        )
    end
    return nothing
end
check_point(::GeneralLinear, ::Identity{MultiplicationOperation}) = nothing

function check_vector(G::GeneralLinear, p, X; kwargs...)
    return nothing
end

distance(G::GeneralLinear, p, q) = norm(G, p, log(G, p, q))

embed(::GeneralLinear, p) = p
embed(::GeneralLinear, p, X) = X

@doc raw"""
    exp(G::GeneralLinear, p, X)

Compute the exponential map on the [`GeneralLinear`](@ref) group.

The exponential map is
````math
\exp_p \colon X ↦ p \operatorname{Exp}(X^\mathrm{H}) \operatorname{Exp}(X - X^\mathrm{H}),
````

where ``\operatorname{Exp}(⋅)`` denotes the matrix exponential, and ``⋅^\mathrm{H}`` is
the conjugate transpose [AndruchowLarotondaRechtVarela:2014](@cite) [MartinNeff:2016](@cite).
"""
function exp(M::GeneralLinear, p, X)
    q = similar(p)
    return exp!(M, q, p, X)
end
function exp(M::GeneralLinear, p, X, t::Number)
    q = similar(p)
    return exp!(M, q, p, t * X)
end

function exp!(G::GeneralLinear, q, p, X)
    expX = exp(X)
    if isnormal(X; atol=sqrt(eps(real(eltype(X)))))
        return compose!(G, q, p, expX)
    end
    compose!(G, q, expX', exp(X - X'))
    compose!(G, q, p, q)
    return q
end
function exp!(G::GeneralLinear, q, p, X, t::Number)
    return exp!(G, q, p, t * X)
end
function exp!(::GeneralLinear{1}, q, p, X)
    p1 = p isa Identity ? p : p[1]
    q[1] = p1 * exp(X[1])
    return q
end
function exp!(G::GeneralLinear{2}, q, p, X)
    if isnormal(X; atol=sqrt(eps(real(eltype(X)))))
        return compose!(G, q, p, exp(SizedMatrix{2,2}(X)))
    end
    A = SizedMatrix{2,2}(X')
    B = SizedMatrix{2,2}(X) - A
    compose!(G, q, exp(A), exp(B))
    compose!(G, q, p, q)
    return q
end

function get_coordinates(
    ::GeneralLinear{n,ℝ},
    p,
    X,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
) where {n}
    return vec(X)
end

function get_coordinates!(
    ::GeneralLinear{n,ℝ},
    Xⁱ,
    p,
    X,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
) where {n}
    return copyto!(Xⁱ, X)
end

get_embedding(::GeneralLinear{n,𝔽}) where {n,𝔽} = Euclidean(n, n; field=𝔽)

function get_vector(
    ::GeneralLinear{n,ℝ},
    p,
    Xⁱ,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
) where {n}
    return reshape(Xⁱ, n, n)
end

function get_vector!(
    ::GeneralLinear{n,ℝ},
    X,
    p,
    Xⁱ,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
) where {n}
    return copyto!(X, Xⁱ)
end

function exp_lie!(::GeneralLinear{1}, q, X)
    q[1] = exp(X[1])
    return q
end
exp_lie!(::GeneralLinear{2}, q, X) = copyto!(q, exp(SizedMatrix{2,2}(X)))

inner(::GeneralLinear, p, X, Y) = dot(X, Y)

inverse_translate_diff(::GeneralLinear, p, q, X, ::LeftForwardAction) = X
inverse_translate_diff(::GeneralLinear, p, q, X, ::RightBackwardAction) = p * X / p

function inverse_translate_diff!(G::GeneralLinear, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, inverse_translate_diff(G, p, q, X, conv))
end

# find sU for s ∈ S⁺ and U ∈ U(n, 𝔽) that minimizes ‖sU - p‖²
function _project_Un_S⁺(p)
    n = LinearAlgebra.checksquare(p)
    F = svd(p)
    s = mean(F.S)
    U = F.U * F.Vt
    return rmul!(U, s)
end

@doc raw"""
    log(G::GeneralLinear, p, q)

Compute the logarithmic map on the [`GeneralLinear(n)`](@ref) group.

The algorithm proceeds in two stages. First, the point ``r = p^{-1} q`` is projected to the
nearest element (under the Frobenius norm) of the direct product subgroup
``\mathrm{O}(n) × S^+``, whose logarithmic map is exactly computed using the matrix
logarithm. This initial tangent vector is then refined using the
[`NLSolveInverseRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.NLSolveInverseRetraction).

For `GeneralLinear(n, ℂ)`, the logarithmic map is instead computed on the realified
supergroup `GeneralLinear(2n)` and the resulting tangent vector is then complexified.

Note that this implementation is experimental.
"""
function log(M::GeneralLinear, p, q)
    X = similar(p)
    return log!(M, X, p, q)
end

function log!(G::GeneralLinear{n,𝔽}, X, p, q) where {n,𝔽}
    pinvq = inverse_translate(G, p, q, LeftForwardAction())
    𝔽 === ℝ && det(pinvq) ≤ 0 && throw(OutOfInjectivityRadiusError())
    if isnormal(pinvq; atol=sqrt(eps(real(eltype(pinvq)))))
        log_safe!(X, pinvq)
    else
        # compute the equivalent logarithm on GL(dim(𝔽) * n, ℝ)
        # this is significantly more stable than computing the complex algorithm
        Gᵣ = GeneralLinear(real_dimension(𝔽) * n, ℝ)
        pinvqᵣ = realify(pinvq, 𝔽)
        Xᵣ = realify(X, 𝔽)
        log_safe!(Xᵣ, _project_Un_S⁺(pinvqᵣ))
        inverse_retraction = NLSolveInverseRetraction(ExponentialRetraction(), Xᵣ)
        inverse_retract!(Gᵣ, Xᵣ, Identity(G), pinvqᵣ, inverse_retraction)
        unrealify!(X, Xᵣ, 𝔽, n)
    end
    translate_diff!(G, X, p, Identity(G), X, LeftForwardAction())
    return X
end
function log!(::GeneralLinear{1}, X, p, q)
    p1 = p isa Identity ? p : p[1]
    X[1] = log(p1 \ q[1])
    return X
end

function _log_lie!(::GeneralLinear{1}, X, p)
    X[1] = log(p[1])
    return X
end

manifold_dimension(G::GeneralLinear) = manifold_dimension(get_embedding(G))

LinearAlgebra.norm(::GeneralLinear, p, X) = norm(X)

parallel_transport_to(::GeneralLinear, p, X, q) = X

parallel_transport_to!(::GeneralLinear, Y, p, X, q) = copyto!(Y, X)

project(::GeneralLinear, p) = p
project(::GeneralLinear, p, X) = X

project!(::GeneralLinear, q, p) = copyto!(q, p)
project!(::GeneralLinear, Y, p, X) = copyto!(Y, X)

@doc raw"""
    Random.rand(G::GeneralLinear; vector_at=nothing, kwargs...)

If `vector_at` is `nothing`, return a random point on the [`GeneralLinear`](@ref) group `G`
by using `rand` in the embedding.

If `vector_at` is not `nothing`, return a random tangent vector from the tangent space of
the point `vector_at` on the [`GeneralLinear`](@ref) by using by using `rand` in the embedding.
"""
rand(G::GeneralLinear; kwargs...)

function Random.rand!(G::GeneralLinear, pX; kwargs...)
    rand!(get_embedding(G), pX; kwargs...)
    return pX
end
function Random.rand!(rng::AbstractRNG, G::GeneralLinear, pX; kwargs...)
    rand!(rng, get_embedding(G), pX; kwargs...)
    return pX
end

Base.show(io::IO, ::GeneralLinear{n,𝔽}) where {n,𝔽} = print(io, "GeneralLinear($n, $𝔽)")

translate_diff(::GeneralLinear, p, q, X, ::LeftForwardAction) = X
translate_diff(::GeneralLinear, p, q, X, ::RightBackwardAction) = p \ X * p

function translate_diff!(G::GeneralLinear, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end
