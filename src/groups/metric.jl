@doc doc"""
    has_invariant_metric(G::AbstractGroupManifold, conv::ActionDirection) -> Val

Return `Val(true)` if the metric on the group $G$ is invariant under translations by the
specified direction, that is, given a group $G$, a left- or right group translation map $τ$,
and a metric $g_e$ on the Lie algebra, a $τ$-invariant metric at any point $x ∈ G$ is
defined as

````math
g_x(v, w) = g_{τ_y x}((\mathrm{d}τ_y)_x v, (\mathrm{d}τ_y)_x w),
````

for $v,w ∈ T_x G$ and all $y ∈ G$, where $(\mathrm{d}τ_y)_x$ is the action of the
differential of translation by $y$ evaluated at $x$ (see [`translate_diff`](@ref)).
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
    InvariantMetric{G<:Metric,D<:ActionDirection} <: Metric

Extend a metric on the Lie algebra of an [`AbstractGroupManifold`](@ref) to the whole group
via translation in the specified direction.

Given a group $G$, a left- or right group translation map $τ$, and a metric $g_e$ on the Lie
algebra, a $τ$-invariant metric at any point $x ∈ G$ is defined as

````math
g_x(v, w) = g_{τ_y x}((\mathrm{d}τ_y)_x v, (\mathrm{d}τ_y)_x w),
````

for $v,w ∈ T_x G$ and all $y ∈ G$, where $(\mathrm{d}τ_y)_x$ is the action of the
differential of translation by $y$ evaluated at $x$ (see [`translate_diff`](@ref)).
Setting $y = x^{-1}$ extends a metric on the Lie algebra to the whole group:

````math
g_x(v, w) = g_e((\mathrm{d}τ_x)_x^{-1} v, (\mathrm{d}τ_x)_x^{-1} w).
````

Note that the above condition is not checked and must be verified for the entire group.

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

function exp!(M::MetricManifold{<:AbstractGroupManifold}, ::Val{false}, y, x, v)
    if has_biinvariant_metric(M) === Val(true)
        return retract!(M, y, x, v, GroupExponentialRetraction(LeftAction()))
    end
    return invoke(
        exp!,
        Tuple{MetricManifold,Val{false},typeof(y),typeof(x),typeof(v)},
        M,
        Val(false),
        y,
        x,
        v,
    )
end

function log!(M::MetricManifold{<:AbstractGroupManifold}, ::Val{false}, v, x, y)
    if has_biinvariant_metric(M) === Val(true)
        return inverse_retract!(M, v, x, y, GroupLogarithmicInverseRetraction(LeftAction()))
    end
    return invoke(
        log!,
        Tuple{MetricManifold,Val{false},typeof(v),typeof(x),typeof(y)},
        M,
        Val(false),
        v,
        x,
        y,
    )
end
