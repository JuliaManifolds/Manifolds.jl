using Manifolds, Pkg, Test

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
        include_test("test_equality.jl")
        include_test("test_notation.jl")
        include_test("test_utils.jl")
        include_test("test_metric.jl")
        include_test("test_statistics.jl")
    end
    if TEST_SET ∈ ["all", "integration"]
        include_test("test_atlases.jl")
        include_test("approx_inverse_retraction.jl")
        # manifolds requiring ODE solvers
        include_test("manifolds-old/embedded_torus.jl")
        # If we are not on apple, test recipes
        Sys.isapple() || include_test("test_recipes.jl")
    end
    (TEST_SET ∈ ["all", "manifolds"]) && Test.@testset "Manifolds Test Suite" begin
        include_test("manifolds/test_centered_matrices.jl")
        include_test("manifolds/test_determinant_one_matrices.jl")
        include_test("manifolds/test_elliptope.jl")
        include_test("manifolds/test_essential_manifold.jl")
        include_test("manifolds/test_euclidean.jl")
        include_test("manifolds/test_fixed_rank.jl")
        include_test("manifolds/test_flag.jl")
        include_test("manifolds/test_generalized_grassmann.jl")
        include_test("manifolds/test_generalized_stiefel.jl")
        include_test("manifolds/test_cholesky_space.jl")
        include_test("manifolds/test_circle.jl")
        include_test("manifolds/test_parametric_surface.jl")
        include_test("manifolds/test_projective_space.jl")
        include_test("manifolds/test_segre.jl")
        include_test("manifolds/test_shape_space.jl")
        include_test("manifolds/test_orthogonal_matrices.jl")
        include_test("manifolds/test_rotations.jl")
        include_test("manifolds/test_sphere.jl")
        include_test("manifolds/test_sphere_symmetric_matrices.jl")
        include_test("manifolds/test_special_unitary_matrices.jl")
        include_test("manifolds/test_spd_fixed_determinant.jl")
        include_test("manifolds/test_spectrahedron.jl")
        include_test("manifolds/test_veronese.jl")
        include_test("manifolds/test_general_unitary.jl")
        include_test("manifolds/test_grassmann.jl")
        include_test("manifolds/test_hamiltonian.jl")
        include_test("manifolds/test_heisenberg_matrices.jl")
        include_test("manifolds/test_hyperbolic.jl")
        include_test("manifolds/test_hyperrectangle.jl")
        include_test("manifolds/test_invertible_matrices.jl")
        include_test("manifolds/test_multinomial_doubly_stochastic.jl")
        include_test("manifolds/test_multinomial_matrices.jl")
        include_test("manifolds/test_multinomial_spd.jl")
        include_test("manifolds/test_multinomial_symmetric.jl")
        include_test("manifolds/test_oblique.jl")
        include_test("manifolds/test_probability_simplex.jl")
        include_test("manifolds/test_power_manifold.jl")
        include_test("manifolds/test_product_manifold.jl")
        include_test("manifolds/test_skew_hermitian_matrices.jl")
        include_test("manifolds/test_stiefel.jl")
        include_test("manifolds/test_symmetric.jl")
        include_test("manifolds/test_symmetric_positive_definite.jl")
        include_test("manifolds/test_symmetric_positive_semidefinite_fixed_rank.jl")
        include_test("manifolds/test_symplectic.jl")
        include_test("manifolds/test_symplectic_grassmann.jl")
        include_test("manifolds/test_symplectic_stiefel.jl")
        include_test("manifolds/test_unitary_matrices.jl")
        include_test("manifolds/test_vector_bundle.jl")
    end
    (TEST_SET ∈ ["all", "manifolds"]) && Test.@testset "Manifolds.jl Old Tests" begin
        # starting with tests of simple manifolds
        include_test("manifolds-old/lorentz.jl")
        include_test("manifolds-old/positive_numbers.jl")
        include_test("manifolds-old/tucker.jl")

        include_test("manifolds-old/torus.jl")

        #meta manifolds
        include_test("manifolds-old/fiber.jl")
        include_test("manifolds-old/fiber_bundle.jl")
        include_test("manifolds-old/graph.jl")
    end
    #
    # On CI, JET runs in its own job for simplicity; it comes last, since JET.jl 0.9 on
    # Julia 1.10 redefines methods of the package it analyzes.
    # When running the tests locally the following still activates the env and runs JET.
    if TEST_SET == "all" && get(ENV, "CI", "false") != "true"
        @testset "JET.jl" begin
            Pkg.activate(joinpath(@__DIR__, "jet"))
            Pkg.instantiate()
            include_test("jet/runtests.jl")
        end
    end
end
