module ManifoldsHybridArraysExt

if isdefined(Base, :get_extension)
    using Manifolds
    using ManifoldsBase

    using Manifolds: PowerManifoldMultidimensional
    using Manifolds: rep_size_to_colons

    using HybridArrays

    import Manifolds: _read
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..Manifolds
    using ..ManifoldsBase

    using ..Manifolds: PowerManifoldMultidimensional
    using ..Manifolds: rep_size_to_colons

    using ..HybridArrays

    import ..Manifolds: _read
end

Base.@propagate_inbounds @inline function _read(
    ::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::HybridArray,
    i::Tuple,
)
    return x[rep_size_to_colons(rep_size)..., i...]
end
Base.@propagate_inbounds @inline function _read(
    ::PowerManifoldMultidimensional,
    rep_size::Tuple{},
    x::HybridArray,
    i::NTuple{N,Int},
) where {N}
    # disambiguation
    return x[i...]
end

end
