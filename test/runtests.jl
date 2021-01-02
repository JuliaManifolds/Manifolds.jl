include("utils.jl")
@info "Manifolds.jl Test settings:\n\n" *
      "Testing Float32:  $(TEST_FLOAT32)\n" *
      "Testing Double64: $(TEST_DOUBLE64)\n" *
      "Testing Static:   $(TEST_STATIC_SIZED)\n\n" *
      "Check test/utils.jl if you wish to change these settings."

@testset "Manifolds.jl" begin
    include_test("differentiation.jl")

    @testset "Ambiguities" begin
        # TODO: reduce the number of ambiguities
        if VERSION.prerelease == () #
            @test length(Test.detect_ambiguities(ManifoldsBase)) <= 17
            @test length(Test.detect_ambiguities(Manifolds)) == 0
            @test length(our_base_ambiguities()) <= 24
        else
            @info "Skipping Ambiguity tests for pre-release versions"
        end
    end

    @testset "utils test" begin
        @test Manifolds.usinc_from_cos(-1) == 0
        @test Manifolds.usinc_from_cos(-1.0) == 0.0
    end

    include_test("groups/group_utils.jl")
    include_test("notation.jl")
    # starting with tests of simple manifolds
    include_test("centered_matrices.jl")
    include_test("circle.jl")
    include_test("cholesky_space.jl")
    include_test("elliptope.jl")
    include_test("euclidean.jl")
    include_test("fixed_rank.jl")
    include_test("generalized_grassmann.jl")
    include_test("generalized_stiefel.jl")
    include_test("grassmann.jl")
    include_test("hyperbolic.jl")
    include_test("multinomial_doubly_stochastic.jl")
    include_test("multinomial_symmetric.jl")
    include_test("positive_numbers.jl")
    include_test("probability_simplex.jl")
    include_test("projective_space.jl")
    include_test("rotations.jl")
    include_test("skewsymmetric.jl")
    include_test("spectrahedron.jl")
    include_test("sphere.jl")
    include_test("sphere_symmetric_matrices.jl")
    include_test("stiefel.jl")
    include_test("symmetric.jl")
    include_test("symmetric_positive_definite.jl")
    include_test("symmetric_positive_semidefinite_fixed_rank.jl")

    include_test("multinomial_matrices.jl")
    include_test("oblique.jl")
    include_test("torus.jl")

    #meta manifolds
    include_test("product_manifold.jl")
    include_test("power_manifold.jl")
    include_test("vector_bundle.jl")
    include_test("graph.jl")

    include_test("metric.jl")
    include_test("statistics.jl")

    # Lie groups and actions
    include_test("groups/groups_general.jl")
    include_test("groups/array_manifold.jl")
    include_test("groups/circle_group.jl")
    include_test("groups/translation_group.jl")
    include_test("groups/special_orthogonal.jl")
    include_test("groups/product_group.jl")
    include_test("groups/semidirect_product_group.jl")
    include_test("groups/special_euclidean.jl")
    include_test("groups/group_operation_action.jl")
    include_test("groups/rotation_action.jl")
    include_test("groups/translation_action.jl")
    include_test("groups/metric.jl")

    include_test("recipes.jl")
end
