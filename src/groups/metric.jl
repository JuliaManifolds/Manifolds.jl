@doc raw"""
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
    To check the condition for a set of points, use [`has_approx_invariant_metric`](@ref).

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
function has_approx_invariant_metric(
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
    if biinvariant_metric_dispatch(M) === Val(true)
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
    biinvariant_metric_dispatch(G::AbstractGroupManifold) -> Val

Return `Val(true)` if the metric on the manifold is bi-invariant, that is, if the metric
is both left- and right-invariant (see [`invariant_metric_dispatch`](@ref)).
"""
function biinvariant_metric_dispatch(M::Manifold)
    return Val(
        invariant_metric_dispatch(M, LeftAction()) === Val(true) &&
        invariant_metric_dispatch(M, RightAction()) === Val(true),
    )
end

has_biinvariant_metric(M::Manifold) = _extract_val(biinvariant_metric_dispatch(M))

@doc raw"""
    invariant_metric_dispatch(G::AbstractGroupManifold, conv::ActionDirection) -> Val

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
invariant_metric_dispatch(::AbstractGroupManifold, ::ActionDirection)

@decorator_transparent_function invariant_metric_dispatch(
    M::AbstractDecoratorManifold,
    conv::ActionDirection
)
function invariant_metric_dispatch(M::MetricManifold, conv::ActionDirection)
    return default_metric_dispatch(M)
end
function invariant_metric_dispatch(
    M::MetricManifold{<:Manifold,<:InvariantMetric},
    conv::ActionDirection,
)
    direction(metric(M)) === conv && return Val(true)
    return invoke(invariant_metric_dispatch, Tuple{MetricManifold,typeof(conv)}, M, conv)
end

function has_invariant_metric(M::Manifold, conv::ActionDirection)
    return _extract_val(invariant_metric_dispatch(M, conv))
end

function inner(M::MetricManifold{<:Manifold,<:InvariantMetric}, ::Val{false}, p, X, Y)
    imetric = metric(M)
    conv = direction(imetric)
    N = MetricManifold(M.manifold, imetric.metric)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    Yₑ = inverse_translate_diff(M, p, p, Y, conv)
    return inner(N, Identity(N), Xₑ, Yₑ)
end

function default_metric_dispatch(M::MetricManifold{<:Manifold,<:InvariantMetric})
    imetric = metric(M)
    N = MetricManifold(M.manifold, imetric.metric)
    default_metric_dispatch(N) !== Val{true} && return Val(false)
    return invariant_metric_dispatch(N, direction(imetric))
end

function log!(M::MetricManifold{<:Manifold,<:InvariantMetric}, ::Val{false}, X, p, q)
    if biinvariant_metric_dispatch(M) === Val(true)
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

function norm(M::MetricManifold{<:Manifold,<:InvariantMetric}, p, X)
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
