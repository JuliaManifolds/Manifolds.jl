module ManifoldsRecursiveArrayToolsExt

if isdefined(Base, :get_extension)
    using Base: @propagate_inbounds
    using Manifolds
    using Manifolds: submanifold_components
    using RecursiveArrayTools: ArrayPartition
    import Base: getindex, setindex!, view
    import Manifolds:
        ProductFVectorDistribution,
        adjoint_Jacobi_field,
        allocate,
        allocate_result,
        apply,
        apply!,
        apply_diff,
        apply_diff_group!,
        compose,
        _compose,
        exp,
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
        log,
        optimal_alignment,
        project,
        rand,
        _rand!,
        translate,
        translate_diff,
        _vector_transport_direction,
        _vector_transport_to,
        vee,
        else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..Manifolds
    using ..RecursiveArrayTools
    import Base: getindex, setindex!, view
    import Manifolds:
        ProductFVectorDistribution,
        adjoint_Jacobi_field,
        allocate,
        allocate_result,
        apply,
        apply!,
        apply_diff,
        apply_diff_group!,
        _compose,
        exp,
        exp_lie,
        get_vector,
        get_vectors,
        identity_element,
        inverse_apply,
        inverse_apply_diff,
        inverse_translate,
        inverse_translate_diff,
        isapprox,
        jacobi_field,
        log,
        optimal_alignment,
        rand,
        _rand!,
        translate,
        translate_diff,
        _vector_transport_direction,
        _vector_transport_to
end

include("FiberBundleRATExt.jl")
include("ProductGroupRATExt.jl")
include("ProductManifoldRATExt.jl")
include("rotation_translation_actionRATExt.jl")
include("semidirect_product_groupRATExt.jl")
include("VectorBundleRATExt.jl")
end
