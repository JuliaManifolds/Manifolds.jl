@doc raw"""
    TranslationGroup{T<:Tuple,F} <: GroupManifold{Euclidean{T,F},AdditionOperation}

Translation group $\mathrm{T}(n)$ represented by translation arrays.

# Constructor
    TranslationGroup(n₁,...,nᵢ; field = 𝔽)

Generate the translation group on
$𝔽^{n₁,…,nᵢ}$ = `Euclidean(n₁,...,nᵢ; field = 𝔽)`, which is isomorphic to group itself.
"""
const TranslationGroup{T<:Tuple,F} = GroupManifold{Euclidean{T,F},AdditionOperation}

function TranslationGroup(n::Int...; field::AbstractNumbers = ℝ)
    return TranslationGroup{Tuple{n...},field}(
        Euclidean(n...; field = field),
        AdditionOperation(),
    )
end

function show(io::IO, ::TranslationGroup{N,F}) where {N,F}
    print(io, "TranslationGroup($(join(N.parameters, ", ")); field = $(F))")
end
