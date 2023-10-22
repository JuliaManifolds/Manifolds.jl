@doc raw"""
    TranslationGroup{T,𝔽} <: GroupManifold{Euclidean{T,𝔽},AdditionOperation}

Translation group $\mathrm{T}(n)$ represented by translation arrays.

# Constructor
    TranslationGroup(n₁,...,nᵢ; field=𝔽, parameter::Symbol=:type)

Generate the translation group on
$𝔽^{n₁,…,nᵢ}$ = `Euclidean(n₁,...,nᵢ; field=𝔽)`, which is isomorphic to the group itself.
"""
const TranslationGroup{T,𝔽} = GroupManifold{𝔽,Euclidean{T,𝔽},AdditionOperation}

function TranslationGroup(n::Int...; field::AbstractNumbers=ℝ, parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, n)
    return TranslationGroup{typeof(size),field}(
        Euclidean(n...; field=field, parameter=parameter),
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
            IsDefaultMetric(EuclideanMetric()),
            active_traits(f, M.manifold, args...),
            IsExplicitDecorator(), #pass to Euclidean by default/last fallback
        )
    end
end

exp!(::TranslationGroup, q, ::Identity{AdditionOperation}, X) = copyto!(q, X)

has_biinvariant_metric(::TranslationGroup) = true

has_invariant_metric(::TranslationGroup, ::ActionDirectionAndSide) = true

identity_element!(::TranslationGroup, p) = fill!(p, 0)

log(::TranslationGroup, ::Identity{AdditionOperation}, q) = q

function log!(::TranslationGroup, X, p::Identity{AdditionOperation}, q)
    copyto!(X, q)
    return X
end

function Base.show(io::IO, M::TranslationGroup{N,𝔽}) where {N<:Tuple,𝔽}
    size = get_parameter(M.manifold.size)
    return print(io, "TranslationGroup($(join(size, ", ")); field=$(𝔽), parameter=:field)")
end
function Base.show(io::IO, M::TranslationGroup{N,𝔽}) where {N<:TypeParameter,𝔽}
    size = get_parameter(M.manifold.size)
    return print(io, "TranslationGroup($(join(size, ", ")); field=$(𝔽))")
end
