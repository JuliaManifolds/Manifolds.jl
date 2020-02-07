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
differential of translation by $q$ evaluated at $p$ (see [`translate_diff`](@ref)).

`InvariantMetric` constructs an (assumed) $τ$-invariant metric by extending the inner
product of a metric $h_e$ on the Lie algebra to the whole group:

````math
g_p(X, Y) = h_e((\mathrm{d}τ_p^{-1})_p X, (\mathrm{d}τ_p^{-1})_p Y).
````

!!! warning
    The invariance condition is not checked and must be verified for the entire group.
    To check the condition for a set of points, use [`check_has_invariant_metric`](@ref).

The convenient aliases [`LeftInvariantMetric`](@ref) and [`RightInvariantMetric`](@ref) are
provided.

# Constructor

    InvariantMetric(metric::Metric, conv::ActionDirection = LeftAction())
"""
struct InvariantMetric{G<:Metric,D<:ActionDirection} <: Metric
    metric::G
end

function InvariantMetric(metric, conv = LeftAction())
    return InvariantMetric{typeof(metric),typeof(conv)}(metric)
end

const LeftInvariantMetric{G} = InvariantMetric{G,LeftAction} where {G<:Metric}

"""
    LeftInvariantMetric(metric::Metric)

Alias for a left-[`InvariantMetric`](@ref).
"""
LeftInvariantMetric(metric) = InvariantMetric{typeof(metric),LeftAction}(metric)

const RightInvariantMetric{G} = InvariantMetric{G,RightAction} where {G<:Metric}

"""
    RightInvariantMetric(metric::Metric)

Alias for a right-[`InvariantMetric`](@ref).
"""
RightInvariantMetric(metric) = InvariantMetric{typeof(metric),RightAction}(metric)

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
points. Namely, for $p ∈ \mathcal{G}$, $X,Y ∈ T_p \mathcal{G}$, a metric $g$, and a
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
    gpXY = inner(M, p, X, Y)
    for q in qs
        τq = translate(M, q, p, conv)
        dτqX = translate_diff(M, q, p, X, conv)
        dτqY = translate_diff(M, q, p, Y, conv)
        isapprox(gpXY, inner(M, τq, dτqX, dτqY); kwargs...) || return false
    end
    return true
end

direction(::InvariantMetric{G,D}) where {G,D} = D()

function exp!(M::MetricManifold{<:Manifold,<:InvariantMetric}, ::Val{false}, q, p, X)
    if has_biinvariant_metric(M) === Val(true)
        conv = direction(metric(M))
        return retract!(M, q, p, X, GroupExponentialRetraction(conv))
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

"""
    has_biinvariant_metric(G::AbstractGroupManifold) -> Val

Return `Val(true)` if the metric on the manifold is bi-invariant, that is, if the metric
is both left- and right-invariant (see [`has_invariant_metric`](@ref)).
"""
function has_biinvariant_metric(M::Manifold)
    return Val(
        has_invariant_metric(M, LeftAction()) === Val(true) &&
        has_invariant_metric(M, RightAction()) === Val(true),
    )
end

@doc doc"""
    has_invariant_metric(G::AbstractGroupManifold, conv::ActionDirection) -> Val

Return `Val(true)` if the metric on the group $\mathcal{G}$ is invariant under translations
by the specified direction, that is, given a group $\mathcal{G}$, a left- or right group
translation map $τ$, and a metric $g_e$ on the Lie algebra, a $τ$-invariant metric at
any point $p ∈ \mathcal{G}$ is defined as a metric with the inner product

````math
g_p(X, Y) = g_{τ_q p}((\mathrm{d}τ_q)_p X, (\mathrm{d}τ_q)_p Y),
````

for $X,Y ∈ T_q \mathcal{G}$ and all $q ∈ \mathcal{G}$, where $(\mathrm{d}τ_q)_p$ is the
differential of translation by $q$ evaluated at $p$ (see [`translate_diff`](@ref)).
"""
function has_invariant_metric(M::Manifold, conv::ActionDirection)
    return has_invariant_metric(M, conv, is_decorator_manifold(M))
end
function has_invariant_metric(M::MetricManifold, conv::ActionDirection)
    return has_invariant_metric(M, conv, is_default_metric(M))
end
function has_invariant_metric(M::Manifold, conv::ActionDirection, ::Val{true})
    return has_invariant_metric(M.manifold, conv)
end
has_invariant_metric(::Manifold, ::ActionDirection, ::Val{false}) = Val(false)
function has_invariant_metric(
    M::MetricManifold{<:Manifold,<:InvariantMetric},
    conv::ActionDirection,
)
    direction(metric(M)) === conv && return Val(true)
    return invoke(has_invariant_metric, Tuple{MetricManifold,typeof(conv)}, M, conv)
end

function inner(M::MetricManifold{<:Manifold,<:InvariantMetric}, ::Val{false}, p, X, Y)
    imetric = metric(M)
    conv = direction(imetric)
    N = MetricManifold(M.manifold, imetric.metric)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    Yₑ = inverse_translate_diff(M, p, p, Y, conv)
    return inner(N, Identity(N), Xₑ, Yₑ)
end

function is_default_metric(M::MetricManifold{<:Manifold,<:InvariantMetric})
    imetric = metric(M)
    N = MetricManifold(M.manifold, imetric.metric)
    is_default_metric(N) === Val(true) || return Val(false)
    return has_invariant_metric(N, direction(imetric))
end

function log!(M::MetricManifold{<:Manifold,<:InvariantMetric}, ::Val{false}, X, p, q)
    if has_biinvariant_metric(M) === Val(true)
        imetric = metric(M)
        conv = direction(imetric)
        return inverse_retract!(M, X, p, q, GroupLogarithmicInverseRetraction(conv))
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

function norm(M::MetricManifold{<:Manifold,<:InvariantMetric}, ::Val{false}, p, X)
    imetric = metric(M)
    conv = direction(imetric)
    N = MetricManifold(M.manifold, imetric.metric)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    return norm(N, Identity(N), Xₑ)
end

function show(io::IO, metric::LeftInvariantMetric)
    print(io, "LeftInvariantMetric($(metric.metric))")
end
function show(io::IO, metric::RightInvariantMetric)
    print(io, "RightInvariantMetric($(metric.metric))")
end
