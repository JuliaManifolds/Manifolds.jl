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

direction(::InvariantMetric{G,D}) where {G,D} = D()

function exp!(
    M::MetricManifold{𝔽,<:AbstractGroupManifold,<:InvariantMetric},
    q,
    p,
    X,
) where {𝔽}
    if has_biinvariant_metric(M)
        conv = direction(metric(M))
        return retract!(base_group(M), q, p, X, GroupExponentialRetraction(conv))
    end
    return invoke(exp!, Tuple{MetricManifold,typeof(q),typeof(p),typeof(X)}, M, q, p, X)
end

@trait_function is_biinvariant_metric(M::AbstractManifold)

function inner(M::MetricManifold{𝔽,<:AbstractManifold,<:InvariantMetric}, p, X, Y) where {𝔽}
    imetric = metric(M)
    conv = direction(imetric)
    N = MetricManifold(M.manifold, imetric.metric)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    Yₑ = inverse_translate_diff(M, p, p, Y, conv)
    return inner(N, Identity(N), Xₑ, Yₑ)
end

function log!(
    M::MetricManifold{𝔽,<:AbstractGroupManifold,<:InvariantMetric},
    X,
    p,
    q,
) where {𝔽}
    if has_biinvariant_metric(M)
        imetric = metric(M)
        conv = direction(imetric)
        return inverse_retract!(
            base_group(M),
            X,
            p,
            q,
            GroupLogarithmicInverseRetraction(conv),
        )
    end
    return invoke(log!, Tuple{MetricManifold,typeof(X),typeof(p),typeof(q)}, M, X, p, q)
end

function LinearAlgebra.norm(
    M::MetricManifold{𝔽,<:AbstractManifold,<:InvariantMetric},
    p,
    X,
) where {𝔽}
    imetric = metric(M)
    conv = direction(imetric)
    N = MetricManifold(M.manifold, imetric.metric)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    return norm(N, Identity(N), Xₑ)
end

function Base.show(io::IO, metric::LeftInvariantMetric)
    return print(io, "LeftInvariantMetric($(metric.metric))")
end
function Base.show(io::IO, metric::RightInvariantMetric)
    return print(io, "RightInvariantMetric($(metric.metric))")
end
