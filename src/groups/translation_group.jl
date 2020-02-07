@doc raw"""
    TranslationGroup{T<:Tuple,𝔽} <: GroupManifold{Euclidean{T,𝔽},AdditionOperation}

Translation group $\mathrm{T}(n)$ represented by translation arrays.

# Constructor
    TranslationGroup(n₁,...,nᵢ; field = 𝔽)

Generate the translation group on
$𝔽^{n₁,…,nᵢ}$ = `Euclidean(n₁,...,nᵢ; field = 𝔽)`, which is isomorphic to the group itself.
"""
const TranslationGroup{T<:Tuple,𝔽} = GroupManifold{Euclidean{T,𝔽},AdditionOperation}

function TranslationGroup(n::Int...; field::AbstractNumbers = ℝ)
    return TranslationGroup{Tuple{n...},field}(
        Euclidean(n...; field = field),
        AdditionOperation(),
    )
end

has_invariant_metric(::TranslationGroup, ::ActionDirection) = Val(true)

is_default_metric(::MetricManifold{<:TranslationGroup,EuclideanMetric}) = Val(true)

function show(io::IO, ::TranslationGroup{N,𝔽}) where {N,𝔽}
    print(io, "TranslationGroup($(join(N.parameters, ", ")); field = $(𝔽))")
end
