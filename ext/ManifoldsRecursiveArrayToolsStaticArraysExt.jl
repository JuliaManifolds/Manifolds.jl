
module ManifoldsRecursiveArrayToolsStaticArraysExt

if isdefined(Base, :get_extension)
    using Manifolds
    using RecursiveArrayTools
    using StaticArrays
    import Manifolds: allocate
else
    using ..Manifolds
    using ..RecursiveArrayTools
    using ..StaticArrays

    import ..Manifolds: allocate
end

function allocate(
    ::PowerManifoldNestedReplacing,
    x::AbstractArray{<:ArrayPartition{T,<:NTuple{N,SArray}}},
) where {T,N}
    return similar(x)
end

end
