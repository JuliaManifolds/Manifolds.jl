module ManifoldsRecursiveArrayToolsExt

using Manifolds
using RecursiveArrayTools: ArrayPartition
import RecursiveArrayTools: recursive_bottom_eltype
using StaticArrays
using LinearAlgebra

using Base.Iterators: repeated

using Manifolds:
    FiberBundleBasisData,
    FiberBundleProductVectorTransport,
    PowerManifoldNestedReplacing,
    TypeParameter

using Manifolds: bundle_transport_tangent_direction, bundle_transport_tangent_to

import Manifolds:
    adjoint_Jacobi_field,
    allocate,
    allocate_result,
    get_coordinates,
    get_vector,
    get_vectors,
    hat,
    isapprox,
    jacobi_field,
    project,
    riemannian_gradient,
    _vector_transport_direction,
    _vector_transport_to,
    vee

function allocate(
        ::PowerManifoldNestedReplacing,
        x::AbstractArray{<:ArrayPartition{T, <:NTuple{N, SArray}}},
    ) where {T, N}
    return similar(x)
end

function allocate_result(M::AbstractDecoratorManifold, f::typeof(get_parameters), p::ArrayPartition)
    return @invoke allocate_result(M, f, p::Any)
end

recursive_bottom_eltype(::TuckerPoint{T}) where {T} = T
recursive_bottom_eltype(::TuckerTangentVector{T}) where {T} = T

include("bundles_rat.jl")
include("ProductManifoldRAT.jl")

end
