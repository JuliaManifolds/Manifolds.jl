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
    return merge_traits(
        IsGroupManifold(M.op),
        HasBiinvariantMetric(),
        active_traits(f, M.manifold, args...),
        IsExplicitDecorator(),
    )
end
# for exp! just pass down, the M.manifold traits would introduce MetricManifold which triggers
# the ODE solver and yields a domain error, cf.
# https://github.com/JuliaManifolds/Manifolds.jl/issues/483
@inline function active_traits(::typeof(exp!), M::TranslationGroup, args...)
    return merge_traits(IsExplicitDecorator())
end

identity_element!(::TranslationGroup, p) = fill!(p, 0)

function Base.show(io::IO, ::TranslationGroup{N,𝔽}) where {N,𝔽}
    return print(io, "TranslationGroup($(join(N.parameters, ", ")); field = $(𝔽))")
end
