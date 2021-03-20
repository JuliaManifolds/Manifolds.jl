@doc raw"""
    GeneralLinear{n,𝔽} <:
        AbstractGroupManifold{𝔽,MultiplicationOperation,DefaultEmbeddingType}

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
struct GeneralLinear{n,𝔽} <:
       AbstractGroupManifold{𝔽,MultiplicationOperation,DefaultEmbeddingType} end

GeneralLinear(n, 𝔽::AbstractNumbers=ℝ) = GeneralLinear{n,𝔽}()

function allocation_promotion_function(::GeneralLinear{n,ℂ}, f, ::Tuple) where {n}
    return complex
end

function check_manifold_point(G::GeneralLinear, p; kwargs...)
    mpv = check_manifold_point(decorated_manifold(G), p; kwargs...)
    mpv === nothing || return mpv
    detp = det(p)
    if iszero(detp)
        return DomainError(
            detp,
            "The matrix $(p) does not lie on $(G), since it is not invertible.",
        )
    end
    return nothing
end
check_manifold_point(::GT, ::Identity{GT}; kwargs...) where {GT<:GeneralLinear} = nothing
function check_manifold_point(G::GeneralLinear, e::Identity; kwargs...)
    return DomainError(e, "The identity element $(e) does not belong to $(G).")
end

function check_tangent_vector(G::GeneralLinear, p, X; check_base_point=true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(G, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = check_tangent_vector(decorated_manifold(G), p, X; kwargs...)
    mpv === nothing || return mpv
    return nothing
end

decorated_manifold(::GeneralLinear{n,𝔽}) where {n,𝔽} = Euclidean(n, n; field=𝔽)

default_metric_dispatch(::GeneralLinear, ::EuclideanMetric) = Val(true)
default_metric_dispatch(::GeneralLinear, ::LeftInvariantMetric{EuclideanMetric}) = Val(true)

distance(G::GeneralLinear, p, q) = norm(G, p, log(G, p, q))

@doc raw"""
    exp(G::GeneralLinear, p, X)

Compute the exponential map on the [`GeneralLinear`](@ref) group.

The exponential map is
````math
\exp_p \colon X ↦ p \operatorname{Exp}(X^\mathrm{H}) \operatorname{Exp}(X - X^\mathrm{H}),
````

where ``\operatorname{Exp}(⋅)`` denotes the matrix exponential, and ``⋅^\mathrm{H}`` is
the conjugate transpose. [^AndruchowLarotondaRechtVarela2014][^MartinNeff2016]

[^AndruchowLarotondaRechtVarela2014]:
    > Andruchow E., Larotonda G., Recht L., and Varela A.:
    > “The left invariant metric in the general linear group”,
    > Journal of Geometry and Physics 86, pp. 241-257, 2014.
    > doi: [10.1016/j.geomphys.2014.08.009](https://doi.org/10.1016/j.geomphys.2014.08.009),
    > arXiv: [1109.0520v1](https://arxiv.org/abs/1109.0520v1).
[^MartinNeff2016]:
    > Martin, R. J. and Neff, P.:
    > “Minimal geodesics on GL(n) for left-invariant, right-O(n)-invariant Riemannian metrics”,
    > Journal of Geometric Mechanics 8(3), pp. 323-357, 2016.
    > doi: [10.3934/jgm.2016010](https://doi.org/10.3934/jgm.2016010),
    > arXiv: [1409.7849v2](https://arxiv.org/abs/1409.7849v2).
"""
exp(::GeneralLinear, p, X)

function exp!(G::GeneralLinear, q, p, X)
    expX = exp(X)
    if isnormal(X; atol=sqrt(eps(real(eltype(X)))))
        return compose!(G, q, p, expX)
    end
    compose!(G, q, expX', exp(X - X'))
    compose!(G, q, p, q)
    return q
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

flat!(::GeneralLinear, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

get_coordinates(::GeneralLinear{n,ℝ}, p, X, ::DefaultOrthonormalBasis) where {n} = vec(X)

function get_coordinates!(
    ::GeneralLinear{n,ℝ},
    Xⁱ,
    p,
    X,
    ::DefaultOrthonormalBasis,
) where {n}
    return copyto!(Xⁱ, X)
end

function get_vector(::GeneralLinear{n,ℝ}, p, Xⁱ, ::DefaultOrthonormalBasis) where {n}
    return reshape(Xⁱ, n, n)
end

function get_vector!(::GeneralLinear{n,ℝ}, X, p, Xⁱ, ::DefaultOrthonormalBasis) where {n}
    return copyto!(X, Xⁱ)
end

function group_exp!(::GeneralLinear{1}, q, X)
    q[1] = exp(X[1])
    return q
end
group_exp!(::GeneralLinear{2}, q, X) = copyto!(q, exp(SizedMatrix{2,2}(X)))

function group_log!(::GeneralLinear{1}, X::AbstractMatrix, p::AbstractMatrix)
    X[1] = log(p[1])
    return X
end

inner(::GeneralLinear, p, X, Y) = dot(X, Y)

invariant_metric_dispatch(::GeneralLinear, ::LeftAction) = Val(true)

inverse_translate_diff(::GeneralLinear, p, q, X, ::LeftAction) = X
inverse_translate_diff(::GeneralLinear, p, q, X, ::RightAction) = p * X / p

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
[`NLsolveInverseRetraction`](@ref).

For `GeneralLinear(n, ℂ)`, the logarithmic map is instead computed on the realified
supergroup `GeneralLinear(2n)` and the resulting tangent vector is then complexified.

Note that this implementation is experimental.
"""
log(::GeneralLinear, p, q)

function log!(G::GeneralLinear{n,𝔽}, X, p, q) where {n,𝔽}
    pinvq = inverse_translate(G, p, q, LeftAction())
    𝔽 === ℝ && det(pinvq) ≤ 0 && throw(OutOfInjectivityRadiusError())
    e = Identity(G, pinvq)
    if isnormal(pinvq; atol=sqrt(eps(real(eltype(pinvq)))))
        log_safe!(X, pinvq)
    else
        # compute the equivalent logarithm on GL(dim(𝔽) * n, ℝ)
        # this is significantly more stable than computing the complex algorithm
        Gᵣ = GeneralLinear(real_dimension(𝔽) * n, ℝ)
        pinvqᵣ = realify(pinvq, 𝔽)
        Xᵣ = realify(X, 𝔽)
        eᵣ = Identity(Gᵣ, pinvqᵣ)
        log_safe!(Xᵣ, _project_Un_S⁺(pinvqᵣ))
        inverse_retraction = NLsolveInverseRetraction(ExponentialRetraction(), Xᵣ)
        inverse_retract!(Gᵣ, Xᵣ, eᵣ, pinvqᵣ, inverse_retraction)
        unrealify!(X, Xᵣ, 𝔽, n)
    end
    translate_diff!(G, X, p, e, X, LeftAction())
    return X
end
function log!(::GeneralLinear{1}, X, p, q)
    p1 = p isa Identity ? p : p[1]
    X[1] = log(p1 \ q[1])
    return X
end

manifold_dimension(G::GeneralLinear) = manifold_dimension(decorated_manifold(G))

LinearAlgebra.norm(::GeneralLinear, p, X) = norm(X)

project(::GeneralLinear, p) = p
project(::GeneralLinear, p, X) = X

project!(::GeneralLinear, q, p) = copyto!(q, p)
project!(::GeneralLinear, Y, p, X) = copyto!(Y, X)

sharp!(::GeneralLinear, X::TFVector, p, ξ::CoTFVector) = copyto!(X, ξ)

Base.show(io::IO, ::GeneralLinear{n,𝔽}) where {n,𝔽} = print(io, "GeneralLinear($n, $𝔽)")

translate_diff(::GeneralLinear, p, q, X, ::LeftAction) = X
translate_diff(::GeneralLinear, p, q, X, ::RightAction) = p \ X * p

function translate_diff!(G::GeneralLinear, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end

vector_transport_to(::GeneralLinear, p, X, q, ::ParallelTransport) = X

vector_transport_to!(::GeneralLinear, Y, p, X, q, ::ParallelTransport) = copyto!(Y, X)
