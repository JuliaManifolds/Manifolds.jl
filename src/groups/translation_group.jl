@doc raw"""
    TranslationGroup{T<:Tuple,𝔽} <: GroupManifold{Euclidean{T,𝔽},AdditionOperation}

Translation group $\mathrm{T}(n)$ represented by translation arrays.

# Constructor
    TranslationGroup(n₁,...,nᵢ; field = 𝔽)

Generate the translation group on
$𝔽^{n₁,…,nᵢ}$ = `Euclidean(n₁,...,nᵢ; field = 𝔽)`, which is isomorphic to the group itself.
"""
const TranslationGroup{T<:Tuple,𝔽} = GroupManifold{𝔽,Euclidean{T,𝔽},AdditionOperation}

function TranslationGroup(n::Int...; field::AbstractNumbers=ℝ)
    return TranslationGroup{Tuple{n...},field}(
        Euclidean(n...; field=field),
        AdditionOperation(),
    )
end

@inline function active_traits(f, M::TranslationGroup, args...)
    if is_metric_function(f)
        #pass to Euclidean by default - but keep Group Decorator for the retraction
        return merge_traits(IsGroupManifold(M.op), IsExplicitDecorator())
    else
        return merge_traits(
            IsGroupManifold(M.op),
            HasBiinvariantMetric(),
            IsDefaultMetric(EuclideanMetric()),
            active_traits(f, M.manifold, args...),
            IsExplicitDecorator(), #pass to Euclidean by default/last fallback
        )
    end
end

identity_element!(::TranslationGroup, p) = fill!(p, 0)

function Manifolds.log!(::Euclidean, X, p::Identity{AdditionOperation}, q)
    copyto!(X, q)
    return X
end

function Base.show(io::IO, ::TranslationGroup{N,𝔽}) where {N,𝔽}
    return print(io, "TranslationGroup($(join(N.parameters, ", ")); field = $(𝔽))")
end
