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

function exp(::TraitList{<:HasBiinvariantMetric}, M::MetricManifold, p, X)
    return retract(base_group(M), p, X, GroupExponentialRetraction(direction(M)))
end
function exp!(::TraitList{<:HasBiinvariantMetric}, M::MetricManifold, q, p, X)
    return retract!(base_group(M), q, p, X, GroupExponentialRetraction(direction(M)))
end

@trait_function has_biinvariant_metric(M::AbstractDecoratorManifold)

has_biinvariant_metric(::TraitList{EmptyTrait}, ::AbstractDecoratorManifold) = false
function has_biinvariant_metric(
    ::TraitList{HasBiinvariantMetric},
    ::AbstractDecoratorManifold,
)
    return true
end

function inner(
    t::TraitList{<:AbstractInvarianceTrait},
    M::AbstractDecoratorManifold,
    p,
    X,
    Y,
) where {𝔽}
    conv = direction(M)
    Xₑ = inverse_translate_diff(M.manifold, p, p, X, conv)
    Yₑ = inverse_translate_diff(M.manifold, p, p, Y, conv)
    return inner(next_trait(t), M, Identity(M), Xₑ, Yₑ)
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

function log(
    ::TraitList{<:HasBiinvariantMetric},
    M::AbstractDecoratorManifold,
    p,
    q,
) where {𝔽}
    conv = direction(M)
    return inverse_retract(base_group(M), p, q, GroupLogarithmicInverseRetraction(conv))
end
function log!(
    ::TraitList{<:HasBiinvariantMetric},
    M::AbstractDecoratorManifold,
    X,
    p,
    q,
) where {𝔽}
    conv = direction(M)
    return inverse_retract!(base_group(M), X, p, q, GroupLogarithmicInverseRetraction(conv))
end

function LinearAlgebra.norm(
    t::TraitList{<:AbstractInvarianceTrait},
    M::AbstractDecoratorManifold,
    p,
    X,
) where {𝔽}
    conv = direction(M)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    return norm(next_trait(t), M, Identity(M), Xₑ)
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
