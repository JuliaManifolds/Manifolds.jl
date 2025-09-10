module ManifoldsHybridArraysExt

using Manifolds
using ManifoldsBase

using Manifolds: PowerManifoldMultidimensional
using Manifolds: rep_size_to_colons

using HybridArrays

import Manifolds: _read

Base.@propagate_inbounds @inline function _read(
        ::PowerManifoldMultidimensional, rep_size::Tuple, x::HybridArray, i::Tuple
    )
    return x[rep_size_to_colons(rep_size)..., i...]
end
Base.@propagate_inbounds @inline function _read(
        ::PowerManifoldMultidimensional, rep_size::Tuple{}, x::HybridArray, i::NTuple{N, Int}
    ) where {N}
    # disambiguation
    return x[i...]
end

end
