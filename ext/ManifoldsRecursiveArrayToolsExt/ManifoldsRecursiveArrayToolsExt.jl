module ManifoldsRecursiveArrayToolsExt

using Manifolds
using ManifoldsBase
using RecursiveArrayTools: AbstractVectorOfArray, ArrayPartition
import RecursiveArrayTools: recursive_bottom_eltype
using StaticArrays

using Manifolds:
    FiberBundleBasisData,
    FiberBundleProductVectorTransport,
    PowerManifoldNestedReplacing

using Manifolds: bundle_transport_tangent_direction, bundle_transport_tangent_to
using Manifolds: PowerManifoldMultidimensional, _write

import Manifolds:
    adjoint_Jacobi_field,
    allocate,
    allocate_result,
    get_coordinates,
    get_vector,
    get_vectors,
    jacobi_field,
    riemannian_gradient,
    _vector_transport_direction,
    _vector_transport_to

function allocate(
        ::PowerManifoldNestedReplacing,
        x::AbstractArray{<:ArrayPartition{T, <:NTuple{N, SArray}}},
    ) where {T, N}
    return similar(x)
end

for MT in [AbstractDecoratorManifold, ProductManifold]
    @eval begin
        function allocate_result(M::$MT, f::typeof(get_parameters), p::ArrayPartition)
            return @invoke allocate_result(M, f, p::Any)
        end
    end
end

# resolve ambiguities with the indexing of RecursiveArrayTools.jl arrays
Base.@propagate_inbounds function Base.getindex(
        p::AbstractVectorOfArray, M::PowerManifoldMultidimensional, I::Integer...
    )
    return collect(get_component(M, p, I...))
end
Base.@propagate_inbounds function Base.getindex(
        p::AbstractVectorOfArray{T, N}, M::PowerManifoldMultidimensional, I::Vararg{Integer, N}
    ) where {T, N}
    return get_component(M, p, I...)
end
function Base.view(
        p::AbstractVectorOfArray,
        M::PowerManifoldMultidimensional,
        I::Union{Integer, Colon, AbstractVector}...,
    )
    return _write(M, representation_size(M.manifold), p, I...)
end
function Base.view(
        p::AbstractVectorOfArray{T, N, <:AbstractVector{T}},
        M::PowerManifoldMultidimensional,
        I::Union{Integer, Colon, AbstractVector}...,
    ) where {T, N}
    return _write(M, representation_size(M.manifold), p, I...)
end

recursive_bottom_eltype(::TuckerPoint{T}) where {T} = T
recursive_bottom_eltype(::TuckerTangentVector{T}) where {T} = T

include("bundles_rat.jl")
include("ProductManifoldRAT.jl")

end
