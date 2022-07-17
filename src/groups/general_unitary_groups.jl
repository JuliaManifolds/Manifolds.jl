@doc raw"""
    GeneralUnitaryMultiplicationGroup{n,𝔽,M} = GroupManifold{𝔽,M,MultiplicationOperation}

A generic type for Lie groups based on a unitary property and matrix multiplcation,
see e.g. [`Orthogonal`](@ref), [`SpecialOrthogonal`](@ref), [`Unitary`](@ref), and [`SpecialUnitary`](@ref)
"""
struct GeneralUnitaryMultiplicationGroup{n,𝔽,S} <: AbstractDecoratorManifold{𝔽}
    manifold::GeneralUnitaryMatrices{n,𝔽,S}
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
    ::GeneralUnitaryMultiplicationGroup,
    ::typeof(log),
    ::Identity{MultiplicationOperation},
    q,
)
    return allocate(q)
end

decorated_manifold(G::GeneralUnitaryMultiplicationGroup) = G.manifold

function Random.rand!(G::GeneralUnitaryMultiplicationGroup, pX; kwargs...)
    rand!(G.manifold, pX; kwargs...)
    return pX
end
function Random.rand!(rng::AbstractRNG, G::GeneralUnitaryMultiplicationGroup, pX; kwargs...)
    rand!(rng, G.manifold, pX; kwargs...)
    return pX
end
function translate_diff!(G::GeneralUnitaryMultiplicationGroup, Y, p, q, X, ::LeftAction)
    return copyto!(G, Y, p, X)
end
function translate_diff!(G::GeneralUnitaryMultiplicationGroup, Y, p, q, X, ::RightAction)
    return copyto!(G, Y, p, inv(G, p) * X * p)
end
