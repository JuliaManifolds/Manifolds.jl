module ManifoldsFillArraysExt

using Manifolds
using ManifoldsBase
using FillArrays: AbstractFill

using Manifolds: PowerManifoldMultidimensional, _write

# resolve the ambiguity with the view of FillArrays.jl arrays
function Base.view(
        p::AbstractFill,
        M::PowerManifoldMultidimensional,
        I::Union{Integer, Colon, AbstractVector}...,
    )
    return _write(M, representation_size(M.manifold), p, I...)
end

end
