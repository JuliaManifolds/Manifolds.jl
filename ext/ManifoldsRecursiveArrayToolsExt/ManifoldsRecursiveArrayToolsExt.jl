module ManifoldsRecursiveArrayToolsExt

using Manifolds
using RecursiveArrayTools: ArrayPartition
using StaticArrays
using LinearAlgebra

using Base.Iterators: repeated

using Manifolds:
    ActionDirectionAndSide,
    ColumnwiseMultiplicationAction,
    FiberBundleBasisData,
    FiberBundleProductVectorTransport,
    LeftColumnwiseSpecialEuclideanAction,
    LeftInvariantRepresentation,
    PowerManifoldNestedReplacing,
    SpecialEuclideanIdentity,
    SpecialEuclideanInGeneralLinear,
    TangentVectorRepresentation,
    TypeParameter

using Manifolds:
    bundle_transport_tangent_direction, bundle_transport_tangent_to, _get_parameter

import Manifolds:
    adjoint_Jacobi_field,
    allocate,
    allocate_result,
    apply,
    apply!,
    apply_diff,
    apply_diff!,
    apply_diff_group!,
    _common_product_translate_diff,
    compose,
    _compose,
    exp_lie,
    get_coordinates,
    get_vector,
    get_vectors,
    hat,
    identity_element,
    inverse_apply,
    inverse_apply_diff,
    inverse_translate,
    inverse_translate_diff,
    isapprox,
    jacobi_field,
    lie_bracket,
    optimal_alignment,
    project,
    translate,
    translate_diff,
    vector_representation,
    _vector_transport_direction,
    _vector_transport_to,
    vee

function allocate(
    ::PowerManifoldNestedReplacing,
    x::AbstractArray{<:ArrayPartition{T,<:NTuple{N,SArray}}},
) where {T,N}
    return similar(x)
end

include("bundles_rat.jl")
include("ProductManifoldRAT.jl")
include("ProductGroupRAT.jl")
include("special_euclidean_rat.jl")
include("rotation_translation_action_rat.jl")

end
