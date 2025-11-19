using Manifolds, Test

TEST_SET = get(ENV, "MANIFOLDS_TEST_SET", "all")

@info "=== Manifolds.jl Test Suite ===\n\n" *
    "Test set       $(TEST_SET)\n\n" *
    "These settings are stored in environment variables."

camel_to_snake(x) = lowercase(replace(string(x), r"(?<!^)(?=[A-Z])" => "_"))

function include_test(path)
    @info "Running $path"
    return @time include(path)  # show basic timing, (this will print a newline at end)
end

@testset "Manifolds.jl" begin
    (TEST_SET ∈ ["all", "utilities", "manifolds"]) && Test.@testset "Utilities" begin
        include_test("test_ambiguities.jl")
        include_test("test_deprecated.jl")
        include_test("test_differentiation.jl")
        include_test("test_notation.jl")
        include_test("test_utils.jl")
        include_test("test_metric.jl")
        include_test("test_statistics.jl")
    end
    if TEST_SET ∈ ["all", "integration"]
        include_test("approx_inverse_retraction.jl")
        # manifolds requiring ODE solvers
        include_test("manifolds-old/embedded_torus.jl")
        # If we are not on apple, test recipes
        Sys.isapple() || include_test("test_recipes.jl")
    end
    (TEST_SET ∈ ["all", "manifolds"]) && Test.@testset "Manifolds Test Suite" begin
        include_test("manifolds/test_sphere.jl")
        include_test("manifolds/test_general_unitary.jl")
    end
    (TEST_SET ∈ ["all", "manifolds"]) && Test.@testset "Manifolds.jl Old Tests" begin
        # starting with tests of simple manifolds
        include_test("manifolds-old/centered_matrices.jl")
        include_test("manifolds-old/circle.jl")
        include_test("manifolds-old/cholesky_space.jl")
        include_test("manifolds-old/determinant_one_matrices.jl")
        include_test("manifolds-old/elliptope.jl")
        include_test("manifolds-old/euclidean.jl")
        include_test("manifolds-old/fixed_rank.jl")
        include_test("manifolds-old/flag.jl")
        include_test("manifolds-old/generalized_grassmann.jl")
        include_test("manifolds-old/generalized_stiefel.jl")
        include_test("manifolds-old/grassmann.jl")
        include_test("manifolds-old/hamiltonian.jl")
        include_test("manifolds-old/heisenberg_matrices.jl")
        include_test("manifolds-old/hyperbolic.jl")
        include_test("manifolds-old/hyperrectangle.jl")
        include_test("manifolds-old/invertible_matrices.jl")
        include_test("manifolds-old/lorentz.jl")
        include_test("manifolds-old/multinomial_doubly_stochastic.jl")
        include_test("manifolds-old/multinomial_symmetric.jl")
        include_test("manifolds-old/multinomial_spd.jl")
        include_test("manifolds-old/positive_numbers.jl")
        include_test("manifolds-old/probability_simplex.jl")
        include_test("manifolds-old/projective_space.jl")
        include_test("manifolds-old/rotations.jl")
        include_test("manifolds-old/segre.jl")
        include_test("manifolds-old/shape_space.jl")
        include_test("manifolds-old/skewhermitian.jl")
        include_test("manifolds-old/spectrahedron.jl")
        include_test("manifolds-old/sphere_symmetric_matrices.jl")
        include_test("manifolds-old/stiefel.jl")
        include_test("manifolds-old/symmetric.jl")
        include_test("manifolds-old/symmetric_positive_definite.jl")
        include_test("manifolds-old/spd_fixed_determinant.jl")
        include_test("manifolds-old/symmetric_positive_semidefinite_fixed_rank.jl")
        include_test("manifolds-old/symplectic.jl")
        include_test("manifolds-old/symplecticgrassmann.jl")
        include_test("manifolds-old/symplecticstiefel.jl")
        include_test("manifolds-old/tucker.jl")
        include_test("manifolds-old/unitary_matrices.jl")

        include_test("manifolds-old/essential_manifold.jl")
        include_test("manifolds-old/multinomial_matrices.jl")
        include_test("manifolds-old/oblique.jl")
        include_test("manifolds-old/torus.jl")

        #meta manifolds
        include_test("manifolds-old/product_manifold.jl")
        include_test("manifolds-old/power_manifold.jl")
        include_test("manifolds-old/fiber_bundle.jl")
        include_test("manifolds-old/vector_bundle.jl")
        include_test("manifolds-old/graph.jl")
    end
end
