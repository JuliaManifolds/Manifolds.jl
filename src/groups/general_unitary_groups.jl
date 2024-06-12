@doc raw"""
    GeneralUnitaryMultiplicationGroup{T,𝔽,S} <: AbstractDecoratorManifold{𝔽}

A generic type for Lie groups based on a unitary property and matrix multiplication,
see e.g. [`Orthogonal`](@ref), [`SpecialOrthogonal`](@ref), [`Unitary`](@ref), and [`SpecialUnitary`](@ref)
"""
struct GeneralUnitaryMultiplicationGroup{T,𝔽,S} <: AbstractDecoratorManifold{𝔽}
    manifold::GeneralUnitaryMatrices{T,𝔽,S}
end

@inline function active_traits(f, ::GeneralUnitaryMultiplicationGroup, args...)
    if is_metric_function(f)
        #pass to Rotations by default - but keep Group Decorator for the retraction
        return merge_traits(
            IsGroupManifold(MultiplicationOperation()),
            IsExplicitDecorator(),
        )
    else
        return merge_traits(
            IsGroupManifold(MultiplicationOperation()),
            HasBiinvariantMetric(),
            IsDefaultMetric(EuclideanMetric()),
            IsExplicitDecorator(), #pass to the inner M by default/last fallback
        )
    end
end

function allocate_result(
    ::GeneralUnitaryMultiplicationGroup,
    ::typeof(exp),
    ::Identity{MultiplicationOperation},
    X,
)
    return allocate(X)
end
function allocate_result(
    M::GeneralUnitaryMultiplicationGroup,
    ::typeof(log),
    ::Identity{MultiplicationOperation},
    q,
)
    return allocate(q)
end
function allocate_result(M::Rotations, ::typeof(rand), ::Identity{MultiplicationOperation})
    return similar(Matrix{Float64}, representation_size(M)...)
end

function allocation_promotion_function(
    ::GeneralUnitaryMultiplicationGroup{<:Any,ℂ},
    ::typeof(identity_element),
    args::Tuple,
)
    return complex
end

decorated_manifold(G::GeneralUnitaryMultiplicationGroup) = G.manifold

@doc raw"""
     exp_lie(G::Orthogonal{TypeParameter{Tuple{2}}}, X)
     exp_lie(G::SpecialOrthogonal{TypeParameter{Tuple{2}}}, X)

Compute the Lie group exponential map on the [`Orthogonal`](@ref)`(2)` or [`SpecialOrthogonal`](@ref)`(2)` group.
Given ``X = \begin{pmatrix} 0 & -θ \\ θ & 0 \end{pmatrix}``, the group exponential is

```math
\exp_e \colon X ↦ \begin{pmatrix} \cos θ & -\sin θ \\ \sin θ & \cos θ \end{pmatrix}.
```
"""
exp_lie(::GeneralUnitaryMultiplicationGroup{TypeParameter{Tuple{2}},ℝ}, X)

@doc raw"""
     exp_lie(G::Orthogonal{TypeParameter{Tuple{4}}}, X)
     exp_lie(G::SpecialOrthogonal{TypeParameter{Tuple{4}}}, X)

Compute the group exponential map on the [`Orthogonal`](@ref)`(4)` or the [`SpecialOrthogonal`](@ref) group.
The algorithm used is a more numerically stable form of those proposed in [GallierXu:2002](@cite), [AndricaRohan:2013](@cite).
"""
exp_lie(::GeneralUnitaryMultiplicationGroup{TypeParameter{Tuple{4}},ℝ}, X)

function exp_lie!(::GeneralUnitaryMultiplicationGroup{TypeParameter{Tuple{2}},ℝ}, q, X)
    @assert size(X) == (2, 2)
    @inbounds θ = (X[2, 1] - X[1, 2]) / 2
    sinθ, cosθ = sincos(θ)
    @inbounds begin
        q[1, 1] = cosθ
        q[2, 1] = sinθ
        q[1, 2] = -sinθ
        q[2, 2] = cosθ
    end
    return q
end
function exp_lie!(::GeneralUnitaryMultiplicationGroup{TypeParameter{Tuple{3}},ℝ}, q, X)
    θ = norm(X) / sqrt(2)
    if θ ≈ 0
        a = 1 - θ^2 / 6
        b = θ / 2
    else
        a = sin(θ) / θ
        b = (1 - cos(θ)) / θ^2
    end
    copyto!(q, I)
    q .+= a .* X
    mul!(q, X, X, b, true)
    return q
end
function exp_lie!(::GeneralUnitaryMultiplicationGroup{TypeParameter{Tuple{4}},ℝ}, q, X)
    T = eltype(X)
    α, β = angles_4d_skew_sym_matrix(X)
    sinα, cosα = sincos(α)
    sinβ, cosβ = sincos(β)
    α² = α^2
    β² = β^2
    Δ = β² - α²
    if !isapprox(Δ, 0; atol=1e-6)  # Case α > β ≥ 0
        sincα = sinα / α
        sincβ = β == 0 ? one(T) : sinβ / β
        a₀ = (β² * cosα - α² * cosβ) / Δ
        a₁ = (β² * sincα - α² * sincβ) / Δ
        a₂ = (cosα - cosβ) / Δ
        a₃ = (sincα - sincβ) / Δ
    elseif α == 0 # Case α = β = 0
        a₀ = a₁ = one(T)
        a₂ = inv(T(2))
        a₃ = inv(T(6))
    else  # Case α ⪆ β ≥ 0, α ≠ 0
        sincα = sinα / α
        r = β / α
        c = 1 / (1 + r)
        d = α * (α - β) / 2
        if α < 1e-2
            e = evalpoly(α², (inv(T(3)), inv(T(-30)), inv(T(840)), inv(T(-45360))))
        else
            e = (sincα - cosα) / α²
        end
        a₀ = (α * sinα + (1 + r - d) * cosα) * c
        a₁ = ((3 - d) * sincα - (2 - r) * cosα) * c
        a₂ = (sincα - (1 - r) / 2 * cosα) * c
        a₃ = (e + (1 - r) * (e - sincα / 2)) * c
    end

    X² = X * X
    X³ = X² * X
    q = a₀ * I + a₁ .* X .+ a₂ .* X² .+ a₃ .* X³
    return q
end

function get_embedding(G::GeneralUnitaryMultiplicationGroup)
    return get_embedding(G.manifold)
end

function inverse_translate(G::GeneralUnitaryMultiplicationGroup, p, q, ::LeftForwardAction)
    return inv(G, p) * q
end
function inverse_translate(
    G::GeneralUnitaryMultiplicationGroup,
    p,
    q,
    ::RightBackwardAction,
)
    return q * inv(G, p)
end

function inverse_translate!(
    G::GeneralUnitaryMultiplicationGroup,
    x,
    p,
    q,
    ::LeftForwardAction,
)
    return mul!(x, inv(G, p), q)
end
function inverse_translate!(
    G::GeneralUnitaryMultiplicationGroup,
    x,
    p,
    q,
    ::RightBackwardAction,
)
    return mul!(x, q, inv(G, p))
end

function inverse_translate_diff(
    G::GeneralUnitaryMultiplicationGroup,
    p,
    q,
    X,
    conv::ActionDirectionAndSide,
)
    return translate_diff(G, inv(G, p), q, X, conv)
end
function inverse_translate_diff!(
    G::GeneralUnitaryMultiplicationGroup,
    Y,
    p,
    q,
    X,
    conv::ActionDirectionAndSide,
)
    return copyto!(Y, inverse_translate_diff(G, p, q, X, conv))
end

function log(G::GeneralUnitaryMultiplicationGroup, ::Identity{MultiplicationOperation}, q)
    return log_lie(G, q)
end

function log!(
    G::GeneralUnitaryMultiplicationGroup,
    X,
    ::Identity{MultiplicationOperation},
    q,
)
    return log_lie!(G, X, q)
end

function log_lie!(
    G::GeneralUnitaryMultiplicationGroup{<:Any,ℝ},
    X::AbstractMatrix,
    q::AbstractMatrix,
)
    log_safe!(X, q)
    return project!(G, X, Identity(G), X)
end
function log_lie!(
    ::GeneralUnitaryMultiplicationGroup{<:Any,ℝ},
    X,
    ::Identity{MultiplicationOperation},
)
    fill!(X, 0)
    return X
end
function log_lie!(
    G::GeneralUnitaryMultiplicationGroup{TypeParameter{Tuple{2}},ℝ},
    X::AbstractMatrix,
    q::AbstractMatrix,
)
    @assert size(q) == (2, 2)
    @inbounds θ = atan(q[2, 1], q[1, 1])
    return get_vector!(G, X, Identity(G), θ, DefaultOrthogonalBasis())
end

function log_lie!(
    G::GeneralUnitaryMultiplicationGroup{TypeParameter{Tuple{4}},ℝ},
    X::AbstractMatrix,
    q::AbstractMatrix,
)
    cosα, cosβ = cos_angles_4d_rotation_matrix(q)
    α = acos(clamp(cosα, -1, 1))
    β = acos(clamp(cosβ, -1, 1))
    if α ≈ 0 && β ≈ π
        A² = Symmetric((q - I) ./ 2)
        P = eigvecs(A²)
        E = similar(q)
        fill!(E, 0)
        @inbounds begin
            E[2, 1] = -β
            E[1, 2] = β
        end
        copyto!(X, P * E * transpose(P))
    else
        det(q) < 0 && throw(
            DomainError(
                "The Lie group logarithm is not defined for $q with a negative determinant ($(det(q)) < 0). Point `q` is in a different connected component of the manifold $G",
            ),
        )
        log_safe!(X, q)
    end
    return project!(G, X, Identity(G), X)
end

function manifold_volume(M::GeneralUnitaryMultiplicationGroup)
    return manifold_volume(M.manifold)
end

function translate_diff!(
    G::GeneralUnitaryMultiplicationGroup,
    Y,
    p,
    q,
    X,
    ::LeftForwardAction,
)
    return copyto!(G, Y, X)
end
function translate_diff!(
    G::GeneralUnitaryMultiplicationGroup,
    Y,
    p,
    q,
    X,
    ::RightForwardAction,
)
    copyto!(G, Y, X)
    return Y
end
function translate_diff!(
    G::GeneralUnitaryMultiplicationGroup,
    Y,
    p,
    q,
    X,
    ::LeftBackwardAction,
)
    copyto!(G, Y, p * X * inv(G, p))
    return Y
end
function translate_diff!(
    G::GeneralUnitaryMultiplicationGroup,
    Y,
    p,
    q,
    X,
    ::RightBackwardAction,
)
    return copyto!(G, Y, inv(G, p) * X * p)
end

function volume_density(M::GeneralUnitaryMultiplicationGroup, p, X)
    return volume_density(M.manifold, p, X)
end
