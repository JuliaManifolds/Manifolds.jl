@doc doc"""
    TranslationGroup{N} <: GroupManifold{Euclidean{N},AdditionOperation}

Translation group `\mathrm{T}(N)` represented by translation arrays.

# Constructor
    TranslationGroup(n)
"""
const TranslationGroup{T<:Tuple} = GroupManifold{Euclidean{T},AdditionOperation}

function TranslationGroup(n...)
    return TranslationGroup{Tuple{n...}}(Euclidean(n...), AdditionOperation())
end

function show(io::IO, ::TranslationGroup{N}) where {N}
    print(io, "TranslationGroup($(N))")
end
