module ManifoldsRecursiveArrayToolsExt

if isdefined(Base, :get_extension)
    using Manifolds
    using RecursiveArrayTools: ArrayPartition
    using StaticArrays
    using LinearAlgebra

    using Manifolds:
        ActionDirectionAndSide,
        ColumnwiseMultiplicationAction,
        FiberBundleBasisData,
        FiberBundleProductVectorTransport,
        LeftColumnwiseSpecialEuclideanAction,
        PowerManifoldNestedReplacing,
        SpecialEuclideanIdentity,
        SpecialEuclideanInGeneralLinear,
        TypeParameter

    using Manifolds: bundle_transport_tangent_direction, _get_parameter

    import Manifolds:
        adjoint_Jacobi_field,
        allocate,
        allocate_result,
        apply,
        apply!,
        apply_diff,
        apply_diff!,
        apply_diff_group!,
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
        jacobi_field,
        lie_bracket,
        optimal_alignment,
        project,
        translate,
        translate_diff,
        _vector_transport_direction,
        vee
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..Manifolds
    using ..RecursiveArrayTools: ArrayPartition
    using ..StaticArrays
    using ..LinearAlgebra

    using ..Manifolds: bundle_transport_tangent_direction, _get_parameter

    using ..Manifolds:
        ActionDirectionAndSide,
        ColumnwiseMultiplicationAction,
        FiberBundleBasisData,
        FiberBundleProductVectorTransport,
        LeftColumnwiseSpecialEuclideanAction,
        PowerManifoldNestedReplacing,
        SpecialEuclideanIdentity,
        SpecialEuclideanInGeneralLinear,
        TypeParameter

    import ..Manifolds:
        adjoint_Jacobi_field,
        allocate,
        allocate_result,
        apply,
        apply!,
        apply_diff,
        apply_diff!,
        apply_diff_group!,
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
        jacobi_field,
        lie_bracket,
        optimal_alignment,
        project,
        translate,
        translate_diff,
        _vector_transport_direction,
        vee
end

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
