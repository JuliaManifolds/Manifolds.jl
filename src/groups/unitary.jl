@doc raw"""
    Unitary{n,𝔽} <: AbstractGroupManifold{𝔽,MultiplicationOperation,DefaultEmbeddingType}

The group of unitary matrices ``\mathrm{U}(n, 𝔽)``.

The group consists of all points ``p ∈ 𝔽^{n × n}`` where
``p^\mathrm{H}p = pp^\mathrm{H} = I``. All such points satisfy the property ``|\det(p)|=1``.

The tangent vectors ``X_p ∈ T_p \mathrm{U}(n, 𝔽)`` are represented instead as the
corresponding element ``X_e = p^\mathrm{H} X_p`` of the Lie algebra ``𝔲(n, 𝔽)``, which
consists of the skew-hermitian matrices, that is, all ``X_e ∈ 𝔽^{n × n}`` where
``X_e = -X_e^\mathrm{H}``.

# Constructor

    Unitary(n, 𝔽=ℂ)

Constructs ``\mathrm{U}(n, 𝔽)``. See also [`Orthogonal(n)`](@ref) for the special case
``\mathrm{O}(n)=\mathrm{U}(n, ℝ)``.
"""
struct Unitary{n,𝔽} <: AbstractGroupManifold{𝔽,MultiplicationOperation,DefaultEmbeddingType} end

Unitary(n, 𝔽::AbstractNumbers=ℂ) = Unitary{n,𝔽}()

function allocation_promotion_function(::Unitary{n,ℂ}, f, ::Tuple) where {n}
    return complex
end

function check_manifold_point(G::Unitary, p; kwargs...)
    mpv = check_manifold_point(decorated_manifold(G), p; kwargs...)
    mpv === nothing || return mpv
    if !isapprox(p' * p, one(p); kwargs...)
        return DomainError(
            norm(p' * p - one(p)),
            "$p must be unitary but it's not at kwargs $kwargs",
        )
    end
    return nothing
end
check_manifold_point(::GT, ::Identity{GT}; kwargs...) where {GT<:Unitary} = nothing
function check_manifold_point(G::Unitary, e::Identity; kwargs...)
    return DomainError(e, "The identity element $(e) does not belong to $(G).")
end

function check_tangent_vector(G::Unitary, p, X; check_base_point=true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(G, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = check_tangent_vector(decorated_manifold(G), p, X; kwargs...)
    mpv === nothing || return mpv
    if !isapprox(X', -X; kwargs...)
        return DomainError(
            norm(X' + X),
            "$p must be skew-Hermitian but it's not at kwargs $kwargs",
        )
    end
    return nothing
end

decorated_manifold(::Unitary{n,𝔽}) where {n,𝔽} = Euclidean(n, n; field=𝔽)

default_metric_dispatch(::Unitary, ::EuclideanMetric) = Val(true)
default_metric_dispatch(::Unitary, ::InvariantMetric{EuclideanMetric}) = Val(true)

exp!(G::Unitary, q, p, X) = compose!(G, q, p, group_exp(G, X))

flat!(::Unitary, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

function group_exp!(::Unitary{1}, q, X)
    q[1] = exp(X[1])
    return q
end

@doc raw"""
    group_exp(G::Unitary{2,ℂ}, X)

Compute the group exponential map on the [`Unitary(2,ℂ)`](@ref) group, which is

````math
\exp_e \colon X ↦ e^{\operatorname{tr}(X) / 2} \left(\cos θ I + \frac{\sin θ}{θ} \left(X - \frac{\operatorname{tr}(X)}{2} I\right)\right),
````
where ``θ = \frac{1}{2} \sqrt{4\det(X) - \operatorname{tr}(X)^2}``.
"""
group_exp(::Unitary{2,ℂ}, X)

function group_exp!(::Unitary{2,ℂ}, q, X)
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

function group_log!(::Unitary{1}, X::AbstractMatrix, p::AbstractMatrix)
    X[1] = log(p[1])
    return X
end

@doc raw"""
    injectivity_radius(G::Unitary)
    injectivity_radius(G::Unitary, p)

Return the injectivity radius on the ``\mathrm{U}(n,𝔽)=``[`Unitary`](@ref) group `G`, which
is globally ``π \sqrt{2}`` for ``𝔽=ℝ`` and ``π`` for ``𝔽=ℂ`` or ``𝔽=ℍ``.
"""
function injectivity_radius(::Unitary, p)
    T = float(real(eltype(p)))
    return T(π)
end
function injectivity_radius(::Unitary, p, ::ExponentialRetraction)
    T = float(real(eltype(p)))
    return T(π)
end

inner(::Unitary, p, X, Y) = dot(X, Y)

Base.inv(::Unitary, p) = adjoint(p)

invariant_metric_dispatch(::Unitary, ::LeftAction) = Val(true)
invariant_metric_dispatch(::Unitary, ::RightAction) = Val(true)

inverse_translate(G::Unitary, p, q, ::LeftAction) = inv(G, p) * q
inverse_translate(G::Unitary, p, q, ::RightAction) = q * inv(G, p)

inverse_translate!(G::Unitary, x, p, q, ::LeftAction) = mul!(x, inv(G, p), q)
inverse_translate!(G::Unitary, x, p, q, ::RightAction) = mul!(x, q, inv(G, p))

function inverse_translate_diff(G::Unitary, p, q, X, conv::ActionDirection)
    return translate_diff(G, inv(G, p), q, X, conv)
end

function inverse_translate_diff!(G::Unitary, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, inverse_translate_diff(G, p, q, X, conv))
end

function log!(G::Unitary, X, p, q)
    pinvq = inverse_translate(G, p, q)
    Xₑ = group_log!(G, X, pinvq)
    e = Identity(G, pinvq)
    copyto!(X, translate_diff(G, p, e, Xₑ, LeftAction()))
    return X
end

function manifold_dimension(G::Unitary{n,𝔽}) where {n,𝔽}
    return real_dimension(𝔽) * div(n * (n + 1), 2) - n
end

"""
    mean(
        G::Orthogonal,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/4);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(G::Manifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::Unitary, ::Any)

function Statistics.mean!(G::Unitary, q, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(G, q, x, w, GeodesicInterpolationWithinRadius(π / 4); kwargs...)
end

LinearAlgebra.norm(::Unitary, p, X) = norm(X)

@doc raw"""
    project(G::Unitary{n,𝔽}, p)

Project the point ``p ∈ 𝔽^{n × n}`` to the nearest point in
``\mathrm{U}(n,𝔽)=``[`Unitary(n,𝔽)`](@ref) under the Frobenius norm. If
``p = U S V^\mathrm{H}`` is the singular value decomposition of ``p``, then the projection
is
````math
\operatorname{proj}_{\mathrm{U}(n,𝔽)} \colon p ↦ U V^\mathrm{H}.
````
"""
project(::Unitary, p)

function project!(::Unitary, q, p)
    F = svd(p)
    mul!(q, F.U, F.Vt)
    return q
end

@doc raw"""
    project(G::Unitary{n,𝔽}, p, X)

Orthogonally project the tangent vector ``X ∈ 𝔽^{n × n}`` to the tangent space of
[`Unitary(n,𝔽)`](@ref) at ``p``, represented as the Lie algebra ``𝔲(n, 𝔽)``. The projection
removes the Hermitian part of ``X``:

````math
\operatorname{proj}_{p} \colon X ↦ \frac{1}{2}(X - X^\mathrm{H}).
````
"""
project(::Unitary, p, X)

function project!(::Unitary, Y, p, X)
    Y .= (X .- X') ./ 2
    return Y
end

sharp!(::Unitary, X::TFVector, p, ξ::CoTFVector) = copyto!(X, ξ)

Base.show(io::IO, ::Unitary{n,𝔽}) where {n,𝔽} = print(io, "Unitary($n, $𝔽)")

translate_diff(::Unitary, p, q, X, ::LeftAction) = X
translate_diff(G::Unitary, p, q, X, ::RightAction) = inv(G, p) * X * p

function translate_diff!(G::Unitary, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end
