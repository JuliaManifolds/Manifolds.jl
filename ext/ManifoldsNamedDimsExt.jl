module ManifoldsNamedDimsExt

using Manifolds
using ManifoldsBase
using NamedDims: NamedDimsArray

using Manifolds: PowerManifoldMultidimensional, _write

# resolve ambiguities with the indexing of NamedDims.jl arrays
Base.@propagate_inbounds function Base.getindex(
        p::NamedDimsArray,
        M::PowerManifoldMultidimensional,
        I::Integer...,
    )
    return collect(get_component(M, p, I...))
end
Base.@propagate_inbounds function Base.getindex(
        p::NamedDimsArray{L, T, N},
        M::PowerManifoldMultidimensional,
        I::Vararg{Integer, N},
    ) where {L, T, N}
    return get_component(M, p, I...)
end
function Base.view(
        p::NamedDimsArray,
        M::PowerManifoldMultidimensional,
        I::Union{Integer, Colon, AbstractVector}...,
    )
    return _write(M, representation_size(M.manifold), p, I...)
end

end
