@doc raw"""
    has_approx_invariant_metric(
        G::AbstractDecoratorManifold,
        p,
        X,
        Y,
        qs::AbstractVector,
        conv::ActionDirection = LeftAction();
        kwargs...,
    ) -> Bool

Check whether the metric on the group $\mathcal{G}$ is (approximately) invariant using a set of predefined
points. Namely, for $p ∈ \mathcal{G}$, $X,Y ∈ T_p \mathcal{G}$, a metric $g$, and a
translation map $τ_q$ in the specified direction, check for each $q ∈ \mathcal{G}$ that the
following condition holds:

````math
g_p(X, Y) ≈ g_{τ_q p}((\mathrm{d}τ_q)_p X, (\mathrm{d}τ_q)_p Y).
````

This is necessary but not sufficient for invariance.

Optionally, `kwargs` passed to `isapprox` may be provided.
"""
has_approx_invariant_metric(::AbstractDecoratorManifold, p, X, Y, qs, ::ActionDirection)
@trait_function has_approx_invariant_metric(
    M::AbstractDecoratorManifold,
    p,
    X,
    Y,
    qs,
    conv::ActionDirection=LeftAction();
    kwargs...,
)
function has_approx_invariant_metric(
    ::TraitList{<:IsGroupManifold},
    M::AbstractDecoratorManifold,
    p,
    X,
    Y,
    qs,
    conv::ActionDirection;
    kwargs...,
)
    gpXY = inner(M, p, X, Y)
    for q in qs
        τq = translate(M, q, p, conv)
        dτqX = translate_diff(M, q, p, X, conv)
        dτqY = translate_diff(M, q, p, Y, conv)
        isapprox(gpXY, inner(M, τq, dτqX, dτqY); kwargs...) || return false
    end
    return true
end

"""
    direction(::AbstractDecoratorManifold) -> AD

Get the direction of the action a certain Lie group with its implicit metric has
"""
direction(::AbstractDecoratorManifold)

@trait_function direction(M::AbstractDecoratorManifold)

direction(::TraitList{HasLeftInvariantMetric}, ::AbstractDecoratorManifold) = LeftAction()

direction(::TraitList{HasRightInvariantMetric}, ::AbstractDecoratorManifold) = RightAction()

function exp(::TraitList{HasLeftInvariantMetric}, M::AbstractDecoratorManifold, p, X)
    return retract(M, p, X, GroupExponentialRetraction(LeftAction()))
end
function exp!(::TraitList{HasLeftInvariantMetric}, M::AbstractDecoratorManifold, q, p, X)
    return retract!(M, q, p, X, GroupExponentialRetraction(LeftAction()))
end
function exp(::TraitList{HasRightInvariantMetric}, M::AbstractDecoratorManifold, p, X)
    return retract(M, p, X, GroupExponentialRetraction(RightAction()))
end
function exp!(::TraitList{HasRightInvariantMetric}, M::AbstractDecoratorManifold, q, p, X)
    return retract!(M, q, p, X, GroupExponentialRetraction(RightAction()))
end
function exp(::TraitList{HasBiinvariantMetric}, M::MetricManifold, p, X)
    return exp(M.manifold, p, X)
end
function exp!(::TraitList{HasBiinvariantMetric}, M::MetricManifold, q, p, X)
    return exp!(M.manifold, q, p, X)
end

function get_coordinates(
    t::TraitList{IT},
    M::MetricManifold,
    p,
    X,
    B::AbstractBasis,
) where {IT<:AbstractInvarianceTrait}
    conv = direction(t, M)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    return get_coordinates_lie(next_trait(t), M, Xₑ, B)
end
function get_coordinates!(
    t::TraitList{IT},
    M::MetricManifold,
    c,
    p,
    X,
    B::AbstractBasis,
) where {IT<:AbstractInvarianceTrait}
    conv = direction(t, M)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    return get_coordinates_lie!(next_trait(t), M, c, Xₑ, B)
end

function get_vector(
    t::TraitList{IT},
    M::MetricManifold,
    p,
    c,
    B::AbstractBasis,
) where {IT<:AbstractInvarianceTrait}
    conv = direction(t, M)
    Xₑ = get_vector_lie(next_trait(t), M, c, B)
    return translate_diff(M, p, Identity(M), Xₑ, conv)
end
function get_vector!(
    t::TraitList{IT},
    M::MetricManifold,
    X,
    p,
    c,
    B::AbstractBasis,
) where {IT<:AbstractInvarianceTrait}
    conv = direction(t, M)
    Xₑ = get_vector_lie(next_trait(t), M, c, B)
    return translate_diff!(M, X, p, Identity(M), Xₑ, conv)
end

@trait_function has_invariant_metric(M::AbstractDecoratorManifold, op::ActionDirection)

# Fallbacks / false
has_invariant_metric(::AbstractManifold, op::ActionDirection) = false
function has_invariant_metric(
    ::TraitList{<:HasLeftInvariantMetric},
    ::AbstractDecoratorManifold,
    ::LeftAction,
)
    return true
end
function has_invariant_metric(
    ::TraitList{<:HasRightInvariantMetric},
    ::AbstractDecoratorManifold,
    ::RightAction,
)
    return true
end

@trait_function has_biinvariant_metric(M::AbstractDecoratorManifold)

# fallback / default: false
has_biinvariant_metric(::AbstractManifold) = false
function has_biinvariant_metric(
    ::TraitList{<:HasBiinvariantMetric},
    ::AbstractDecoratorManifold,
)
    return true
end

function inner(
    t::TraitList{IT},
    M::AbstractDecoratorManifold,
    p,
    X,
    Y,
) where {IT<:AbstractInvarianceTrait}
    conv = direction(t, M)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    Yₑ = inverse_translate_diff(M, p, p, Y, conv)
    return inner(next_trait(t), M, Identity(M), Xₑ, Yₑ)
end
function inner(
    t::TraitList{<:IsGroupManifold},
    M::AbstractDecoratorManifold,
    ::Identity,
    X,
    Y,
)
    return inner(next_trait(t), M, identity_element(M), X, Y)
end

function inverse_translate_diff(
    ::TraitList{IsMetricManifold},
    M::MetricManifold,
    p,
    q,
    X,
    conv::ActionDirection,
)
    return inverse_translate_diff(M.manifold, p, q, X, conv)
end
function inverse_translate_diff!(
    ::TraitList{IsMetricManifold},
    M::MetricManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection,
)
    return inverse_translate_diff!(M.manifold, Y, p, q, X, conv)
end

function log(::TraitList{HasLeftInvariantMetric}, M::AbstractDecoratorManifold, p, q)
    return inverse_retract(M, p, q, GroupLogarithmicInverseRetraction(LeftAction()))
end
function log!(::TraitList{HasLeftInvariantMetric}, M::AbstractDecoratorManifold, X, p, q)
    return inverse_retract!(M, X, p, q, GroupLogarithmicInverseRetraction(LeftAction()))
end
function log(::TraitList{HasRightInvariantMetric}, M::AbstractDecoratorManifold, p, q)
    return inverse_retract(M, p, q, GroupLogarithmicInverseRetraction(RightAction()))
end
function log!(::TraitList{HasRightInvariantMetric}, M::AbstractDecoratorManifold, X, p, q)
    return inverse_retract!(M, X, p, q, GroupLogarithmicInverseRetraction(RightAction()))
end
function log(::TraitList{HasBiinvariantMetric}, M::MetricManifold, p, q)
    return log(M.manifold, p, q)
end
function log!(::TraitList{HasBiinvariantMetric}, M::MetricManifold, X, p, q)
    return log!(M.manifold, X, p, q)
end

function LinearAlgebra.norm(
    t::TraitList{IT},
    M::AbstractDecoratorManifold,
    p,
    X,
) where {IT<:AbstractInvarianceTrait}
    conv = direction(t, M)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    return norm(next_trait(t), M, Identity(M), Xₑ)
end
function LinearAlgebra.norm(
    t::TraitList{<:IsGroupManifold},
    M::AbstractDecoratorManifold,
    ::Identity,
    X,
)
    return norm(next_trait(t), M, identity_element(M), X)
end

function translate_diff!(
    ::TraitList{IsMetricManifold},
    M::MetricManifold,
    p,
    q,
    X,
    conv::ActionDirection,
)
    return translate_diff(M.manifold, p, q, X, conv)
end
function translate_diff!(
    ::TraitList{IsMetricManifold},
    M::MetricManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection,
)
    return translate_diff!(M.manifold, Y, p, q, X, conv)
end

"""
    LeftInvariantMetric <: AbstractMetric

An [`AbstractMetric`](@ref) that changes the metric of a Lie group to the left-invariant
metric obtained by left-translations to the identity. Adds the
[`HasLeftInvariantMetric`](@ref) trait.
"""
struct LeftInvariantMetric <: AbstractMetric end

"""
    RightInvariantMetric <: AbstractMetric

An [`AbstractMetric`](@ref) that changes the metric of a Lie group to the right-invariant
metric obtained by right-translations to the identity. Adds the
[`HasRightInvariantMetric`](@ref) trait.
"""
struct RightInvariantMetric <: AbstractMetric end

@inline function active_traits(
    f,
    ::MetricManifold{<:Any,<:AbstractManifold,LeftInvariantMetric},
    args...,
)
    return merge_traits(HasLeftInvariantMetric(), IsExplicitDecorator())
end

@inline function active_traits(
    f,
    ::MetricManifold{<:Any,<:AbstractManifold,RightInvariantMetric},
    args...,
)
    return merge_traits(HasRightInvariantMetric(), IsExplicitDecorator())
end
