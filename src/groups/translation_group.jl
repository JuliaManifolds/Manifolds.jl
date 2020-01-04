@doc doc"""
    TranslationGroup{N} <: GroupManifold{Euclidean{N},AdditionOperation}

Translation group $\mathrm{T}(N)$ represented by translation arrays.

# Constructor
    TranslationGroup(n)
"""
const TranslationGroup{T<:Tuple,F} = GroupManifold{Euclidean{T,F},AdditionOperation}

function TranslationGroup(n::Int...; field::AbstractField=ℝ)
    return TranslationGroup{Tuple{n...},field}(Euclidean(n...; field=field), AdditionOperation())
end

function show(io::IO, ::TranslationGroup{N,F}) where {N,F}
    print(io, "TranslationGroup($(join(N.parameters, ", ")); field = $(F))")
end
