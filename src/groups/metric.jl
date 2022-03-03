@doc raw"""
    has_approx_invariant_metric(
        G::AbstractGroupManifold,
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
has_approx_invariant_metric(
    ::AbstractGroupManifold,
    ::Any,
    ::Any,
    ::Any,
    ::Any,
    ::ActionDirection,
)
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
    M::AbstractGroupManifold,
    p,
    X,
    Y,
    qs,
    conv::ActionDirection=LeftAction();
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

Get the direction of the action a certain [`AbstractGroupManifold`](@ref) with its implicit metric has
"""
direction(::AbstractDecoratorManifold)

@trait_function direction(M::AbstractDecoratorManifold)

direction(::TraitList{HasLeftInvariantMetric}, ::AbstractDecoratorManifold) = LeftAction()

direction(::TraitList{HasRightInvariantMetric}, ::AbstractDecoratorManifold) = RightAction()

function exp!(
    ::TraitList{<:AbstractInvarianceTrait},
    M::MetricManifold{𝔽,<:AbstractGroupManifold},
    q,
    p,
    X,
) where {𝔽}
    if has_biinvariant_metric(M)
        conv = direction(M.manifold)
        return retract!(base_group(M), q, p, X, GroupExponentialRetraction(conv))
    end
    return invoke(exp!, Tuple{MetricManifold,typeof(q),typeof(p),typeof(X)}, M, q, p, X)
end

@trait_function has_biinvariant_metric(M::AbstractDecoratorManifold)

has_biinvariant_metric(::TraitList{EmptyTrait}, ::AbstractGroupManifold) = false
has_biinvariant_metric(::TraitList{HasBiinvariantMetric}, ::AbstractGroupManifold) = true

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
    ::TraitList{<:AbstractInvarianceTrait},
    M::AbstractDecoratorManifold,
    p,
    X,
) where {𝔽}
    conv = direction(M)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    return norm(base_group(M), Identity(M), Xₑ)
end
