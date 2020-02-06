@doc doc"""
    has_invariant_metric(G::AbstractGroupManifold, conv::ActionDirection) -> Val

Return `Val(true)` if the metric on the group $\mathcal{G}$ is invariant under translations
by the specified direction, that is, given a group $\mathcal{G}$, a left- or right group
translation map $τ$, and a metric $g_e$ on the Lie algebra, a $τ$-invariant metric at any
point $p ∈ \mathcal{G}$ is defined as a metric with the inner product

````math
g_p(X, Y) = g_{τ_q p}((\mathrm{d}τ_q)_p X, (\mathrm{d}τ_q)_p Y),
````

for $X,Y ∈ T_q \mathcal{G}$ and all $q ∈ \mathcal{G}$, where $(\mathrm{d}τ_q)_p$ is the
action of the differential of translation by $q$ evaluated at $p$ (see
[`translate_diff`](@ref)).
"""
function has_invariant_metric(M::Manifold, conv::ActionDirection)
    return has_invariant_metric(M, conv, is_decorator_manifold(M))
end
function has_invariant_metric(M::Manifold, conv::ActionDirection, ::Val{true})
    return has_invariant_metric(M.manifold, conv)
end
has_invariant_metric(::Manifold, ::ActionDirection, ::Val{false}) = Val(false)

"""
    has_biinvariant_metric(G::AbstractGroupManifold) -> Val

Return `Val(true)` if the metric on the manifold is bi-invariant, that is, if the metric
is both left- and right-invariant (see [`has_invariant_metric`](@ref)).
"""
function has_biinvariant_metric(M::Manifold)
    return Val(
        (
            has_invariant_metric(M, LeftAction()) === Val(true) &&
            has_invariant_metric(M, RightAction()) === Val(true)
        ),
    )
end

@doc doc"""
    check_has_invariant_metric(
        G::AbstractGroupManifold,
        p,
        X,
        Y,
        qs::AbstractVector,
        conv::ActionDirection = LeftAction();
        kwargs...,
    ) -> Bool

Check whether the metric on the group $\mathcal{G}$ is invariant using a set of predefined
points. Namely, for $p ∈ \mathcal{G}$, $X,Y \in T_p \mathcal{G}$, a metric $g$, and a
translation map $τ_q$ in the specified direction, check for each $q ∈ \mathcal{G}$ that the
following condition holds:

````math
g_p(X, Y) ≈ g_{τ_q p}((\mathrm{d}τ_q)_p X, (\mathrm{d}τ_q)_p Y).
````

This is necessary but not sufficient for invariance.

Optionally, `kwargs` passed to `isapprox` may be provided.
"""
function check_has_invariant_metric(
    M::Manifold,
    p,
    X,
    Y,
    qs,
    conv::ActionDirection = LeftAction();
    kwargs...,
)
    ip = inner(M, p, X, Y)
    for q in qs
        τp = translate(M, q, p, conv)
        dτX = translate_diff(M, q, p, X, conv)
        dτY = translate_diff(M, q, p, Y, conv)
        isapprox(ip, inner(M, τp, dτX, dτY); kwargs...) || return false
    end
    return true
end

@doc doc"""
    InvariantMetric{G<:Metric,D<:ActionDirection} <: Metric

Extend a metric on the Lie algebra of an [`AbstractGroupManifold`](@ref) to the whole group
via translation in the specified direction.

Given a group $\mathcal{G}$ and a left- or right group translation map $τ$ on the group, a
metric $g$ is $τ$-invariant if it has the inner product

````math
g_p(X, Y) = g_{τ_q p}((\mathrm{d}τ_q)_p X, (\mathrm{d}τ_q)_p Y),
````

for all $p,q ∈ \mathcal{G}$ and $X,Y ∈ T_p \mathcal{G}$, where $(\mathrm{d}τ_q)_p$ is the
action of the differential of translation by $q$ evaluated at $p$ (see
[`translate_diff`](@ref)).

`InvariantMetric` constructs an (assumed) $τ$-invariant metric by extending the inner
product of a metric $h_e$ on the Lie algebra to the whole group:

````math
g_p(X, Y) = h_e((\mathrm{d}τ_p)_p^{-1} X, (\mathrm{d}τ_p)_p^{-1} Y).
````

!!! warning
    The invariance condition is not checked and must be verified for the entire group.
    To check the condition for a set of points, use [`check_has_invariant_metric`](@ref).

The convenient aliases [`LeftInvariantMetric`](@ref) and [`RightInvariantMetric`](@ref) are
provided.

# Constructor

    InvariantMetric(metric::Metric, conv::ActionDirection)
"""
struct InvariantMetric{G<:Metric,D<:ActionDirection} <: Metric
    metric::G
end

function InvariantMetric(metric, conv = LeftAction())
    return InvariantMetric{typeof(metric),typeof(conv)}(metric)
end

const LeftInvariantMetric{G} = InvariantMetric{G,LeftAction}

"""
    LeftInvariantMetric(metric::Metric)

Alias for a left-[`InvariantMetric`](@ref).
"""
LeftInvariantMetric(metric) = InvariantMetric{LeftAction}(metric)

const RightInvariantMetric{G} = InvariantMetric{G,RightAction}

"""
    RightInvariantMetric(metric::Metric)

Alias for a right-[`InvariantMetric`](@ref).
"""
RightInvariantMetric(metric) = InvariantMetric{RightAction}(metric)

direction(::InvariantMetric{G,D}) where {G,D} = D()

function has_invariant_metric(
    ::MetricManifold{<:Manifold,InvariantMetric{<:Metric,D}},
    ::D,
) where {D}
    return Val(true)
end

function exp!(M::MetricManifold{<:AbstractGroupManifold}, ::Val{false}, q, p, X)
    if has_biinvariant_metric(M) === Val(true)
        return retract!(M, q, p, X, GroupExponentialRetraction(LeftAction()))
    end
    return invoke(
        exp!,
        Tuple{MetricManifold,Val{false},typeof(q),typeof(p),typeof(X)},
        M,
        Val(false),
        q,
        p,
        X,
    )
end

function log!(M::MetricManifold{<:AbstractGroupManifold}, ::Val{false}, X, p, q)
    if has_biinvariant_metric(M) === Val(true)
        return inverse_retract!(M, X, p, q, GroupLogarithmicInverseRetraction(LeftAction()))
    end
    return invoke(
        log!,
        Tuple{MetricManifold,Val{false},typeof(X),typeof(p),typeof(q)},
        M,
        Val(false),
        X,
        p,
        q,
    )
end
