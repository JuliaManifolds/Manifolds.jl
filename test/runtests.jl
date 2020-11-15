include("differentiation.jl")

# Ambiguity detection is so much slower on Julia 1.6.0-DEV.430, so this reduces
# the number of packages involved to a reasonable minimum
using LinearAlgebra, Manifolds, ManifoldsBase, StaticArrays, Statistics, StatsBase

function our_base_ambiguities()
    ambigs = Test.detect_ambiguities(Base)
    modules_we_care_about =
        [Base, LinearAlgebra, Manifolds, ManifoldsBase, StaticArrays, Statistics, StatsBase]
    our_ambigs = filter(ambigs) do (m1, m2)
        we_care = m1.module in modules_we_care_about && m2.module in modules_we_care_about
        return we_care && (m1.module === Manifolds || m2.module === Manifolds)
    end
    return our_ambigs
end

(VERSION >= v"1.1") && @testset "Ambiguities" begin
    # TODO: reduce the number of ambiguities
    if VERSION >= v"1.6-DEV"
        @test length(Test.detect_ambiguities(ManifoldsBase)) <= 3
        @test length(Test.detect_ambiguities(Manifolds)) <= 101
        @test length(our_base_ambiguities()) <= 4
    else
        @test length(Test.detect_ambiguities(ManifoldsBase)) <= 17
        @test length(Test.detect_ambiguities(Manifolds)) == 0
        @test length(our_base_ambiguities()) <= 21
    end
end

include("utils.jl")

@info "Manifolds.jl Test settings:\n\n" *
      "Testing Float32:  $(TEST_FLOAT32)\n" *
      "Testing Double64: $(TEST_DOUBLE64)\n" *
      "Testing Static:   $(TEST_STATIC_SIZED)\n"

include("groups/group_utils.jl")
include("sized_abstract_array.jl")
include("errors.jl")
include("notation.jl")
# starting with tests of simple manifolds
include("centered_matrices.jl")
include("circle.jl")
include("cholesky_space.jl")
include("elliptope.jl")
include("euclidean.jl")
include("fixed_rank.jl")
include("generalized_grassmann.jl")
include("generalized_stiefel.jl")
include("grassmann.jl")
include("hyperbolic.jl")
include("multinomial_doubly_stochastic.jl")
include("multinomial_symmetric.jl")
include("positive_numbers.jl")
include("probability_simplex.jl")
include("projective_space.jl")
include("rotations.jl")
include("skewsymmetric.jl")
include("spectrahedron.jl")
include("sphere.jl")
include("sphere_symmetric_matrices.jl")
include("stiefel.jl")
include("symmetric.jl")
include("symmetric_positive_definite.jl")
include("symmetric_positive_semidefinite_fixed_rank.jl")

include("multinomial_matrices.jl")
include("oblique.jl")
include("torus.jl")

#meta manifolds
include("product_manifold.jl")
include("power_manifold.jl")
include("vector_bundle.jl")
include("graph.jl")

include("metric.jl")
include("statistics.jl")

# Lie groups and actions
include("groups/groups_general.jl")
include("groups/array_manifold.jl")
include("groups/circle_group.jl")
include("groups/translation_group.jl")
include("groups/general_linear.jl")
include("groups/special_orthogonal.jl")
include("groups/product_group.jl")
include("groups/semidirect_product_group.jl")
include("groups/special_euclidean.jl")
include("groups/group_operation_action.jl")
include("groups/rotation_action.jl")
include("groups/translation_action.jl")
include("groups/metric.jl")
