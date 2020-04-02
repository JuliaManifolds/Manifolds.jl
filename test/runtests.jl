include("autodiff.jl")
include("utils.jl")

@info "Manifolds.jl Test settings:\n\n"*
    "Testing Float32:  $(TEST_FLOAT32)\n"*
    "Testing Double64: $(TEST_DOUBLE64)\n"*
    "Testing Static:   $(TEST_STATIC_SIZED)\n"

# starting with tests of simple manifolds
if testing_group("basic")
    @testset "Basic manifolds" begin
        include("circle.jl")
        include("cholesky_space.jl")
        include("euclidean.jl")
        include("fixed_rank.jl")
        include("generalized_grassmann.jl")
        include("generalized_stiefel.jl")
        include("grassmann.jl")
        include("hyperbolic.jl")
        include("rotations.jl")
        include("skewsymmetric.jl")
        include("sphere.jl")
        include("stiefel.jl")
        include("symmetric.jl")
        include("symmetric_positive_definite.jl")
        include("oblique.jl")
        include("torus.jl")
    end
end

# meta manifolds
if testing_group("combined")
    @testset "Combined manifolds" begin
        include("product_manifold.jl")
        include("power_manifold.jl")
        include("vector_bundle.jl")
        include("graph.jl")
        include("metric.jl")
    end
end

if testing_group("groupmisc")
    @testset "Group manifolds/miscellaneous" begin
        @testset "Ambiguities" begin
            # TODO: reduce the number of ambiguities
            length(Test.detect_ambiguities(ManifoldsBase)) <= 13
            length(Test.detect_ambiguities(Manifolds)) == 0
        end

        include("sized_abstract_array.jl")
        include("statistics.jl")

        # Lie groups and actions
        include("groups/group_utils.jl")
        include("groups/groups_general.jl")
        include("groups/array_manifold.jl")
        include("groups/circle_group.jl")
        include("groups/translation_group.jl")
        include("groups/special_orthogonal.jl")
        include("groups/product_group.jl")
        include("groups/semidirect_product_group.jl")
        include("groups/special_euclidean.jl")
        include("groups/group_operation_action.jl")
        include("groups/rotation_action.jl")
        include("groups/translation_action.jl")
        include("groups/metric.jl")
    end
end
