@doc doc"""
    EuclideanGroup{N} <: GroupManifold{Euclidean{N},AdditionOperation}

Euclidean group `\mathrm{R}(N)` represented by translation arrays.

# Constructor
    EuclideanGroup(n)
"""
const EuclideanGroup{T<:Tuple} = GroupManifold{Euclidean{T},AdditionOperation}

function EuclideanGroup(n...)
    return EuclideanGroup{Tuple{n...}}(Euclidean(n...), AdditionOperation())
end

function show(io::IO, ::EuclideanGroup{N}) where {N}
    print(io, "EuclideanGroup($(N))")
end
