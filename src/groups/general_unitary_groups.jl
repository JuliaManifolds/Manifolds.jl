@doc raw"""
    GeneralUnitaryMultiplicationGroup{n,𝔽,M} = GroupManifold{𝔽,M,MultiplicationOperation}

A generic type for Lie groups based on a unitary property and matrix multiplcation,
see e.g. [`Orthogonal`](@ref), [`SpecialOrthogonal`](@ref), [`Unitary`](@ref), and [`SpecialUnitary`](@ref)
"""
const GeneralUnitaryMultiplicationGroup{n,𝔽,M<:AbstractManifold{𝔽}} =
    GroupManifold{𝔽,M,MultiplicationOperation}

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
