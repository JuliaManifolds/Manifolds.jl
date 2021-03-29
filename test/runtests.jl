include("utils.jl")
@info "Manifolds.jl Test settings:\n\n" *
      "Testing Float32:  $(TEST_FLOAT32)\n" *
      "Testing Double64: $(TEST_DOUBLE64)\n" *
      "Testing Static:   $(TEST_STATIC_SIZED)\n\n" *
      "Check test/utils.jl if you wish to change these settings."

@testset "Manifolds.jl" begin
    include_test("differentiation.jl")

    @testset "Ambiguities" begin
        # TODO: reduce the number of ambiguities for Julia 1.6
        if VERSION.prerelease == () && !Sys.iswindows() &&  VERSION < v"1.6.0"
            @test length(Test.detect_ambiguities(ManifoldsBase)) <= 18
            @test length(Test.detect_ambiguities(Manifolds)) == 0
            @test length(our_base_ambiguities()) <= 24
        else
            @info "Skipping Ambiguity tests for pre-release versions"
        end
    end

    @testset "utils test" begin
        Random.seed!(42)
        @testset "usinc_from_cos" begin
            @test Manifolds.usinc_from_cos(-1) == 0
            @test Manifolds.usinc_from_cos(-1.0) == 0.0
        end
        @testset "log_safe!" begin
            n = 8
            Q = qr(randn(n, n)).Q
            A1 = Matrix(Hermitian(Q * Diagonal(rand(n)) * Q'))
            @test exp(Manifolds.log_safe!(similar(A1), A1)) ≈ A1 atol = 1e-6
            A1_fail = Matrix(Hermitian(Q * Diagonal([-1; rand(n - 1)]) * Q'))
            @test_throws DomainError Manifolds.log_safe!(similar(A1_fail), A1_fail)

            T = triu!(randn(n, n))
            T[diagind(T)] .= rand.()
            @test exp(Manifolds.log_safe!(similar(T), T)) ≈ T atol = 1e-6
            T_fail = copy(T)
            T_fail[1] = -1
            @test_throws DomainError Manifolds.log_safe!(similar(T_fail), T_fail)

            A2 = Q * T * Q'
            @test exp(Manifolds.log_safe!(similar(A2), A2)) ≈ A2 atol = 1e-6
            A2_fail = Q * T_fail * Q'
            @test_throws DomainError Manifolds.log_safe!(similar(A2_fail), A2_fail)

            A3 = exp(randn(n, n))
            @test exp(Manifolds.log_safe!(similar(A3), A3)) ≈ A3 atol = 1e-6

            A3_fail = Float64[1 2; 3 1]
            @test_throws DomainError Manifolds.log_safe!(similar(A3_fail), A3_fail)

            A4 = randn(ComplexF64, n, n)
            @test exp(Manifolds.log_safe!(similar(A4), A4)) ≈ A4 atol = 1e-6
        end
        @testset "isnormal" begin
            @test !Manifolds.isnormal([1.0 2.0; 3.0 4.0])
            @test !Manifolds.isnormal(complex.(reshape(1:4, 2, 2), reshape(5:8, 2, 2)))

            # diagonal
            @test Manifolds.isnormal(diagm(randn(5)))
            @test Manifolds.isnormal(diagm(randn(ComplexF64, 5)))
            @test Manifolds.isnormal(Diagonal(randn(5)))
            @test Manifolds.isnormal(Diagonal(randn(ComplexF64, 5)))

            # symmetric/hermitian
            @test Manifolds.isnormal(Symmetric(randn(3, 3)))
            @test Manifolds.isnormal(Hermitian(randn(3, 3)))
            @test Manifolds.isnormal(Hermitian(randn(ComplexF64, 3, 3)))
            x = Matrix(Symmetric(randn(3, 3)))
            x[3, 1] += eps()
            @test !Manifolds.isnormal(x)
            @test Manifolds.isnormal(x; atol=sqrt(eps()))

            # skew-symmetric/skew-hermitian
            skew(x) = x - x'
            @test Manifolds.isnormal(skew(randn(3, 3)))
            @test Manifolds.isnormal(skew(randn(ComplexF64, 3, 3)))

            # orthogonal/unitary
            @test Manifolds.isnormal(Matrix(qr(randn(3, 3)).Q); atol=sqrt(eps()))
            @test Manifolds.isnormal(
                Matrix(qr(randn(ComplexF64, 3, 3)).Q);
                atol=sqrt(eps()),
            )
        end
        @testset "realify/unrealify!" begin
            # round trip real
            x = randn(3, 3)
            @test Manifolds.realify(x, ℝ) === x
            @test Manifolds.unrealify!(similar(x), x, ℝ) == x

            # round trip complex
            x2 = randn(ComplexF64, 3, 3)
            x2r = Manifolds.realify(x2, ℂ)
            @test eltype(x2r) <: Real
            @test size(x2r) == (6, 6)
            x2c = Manifolds.unrealify!(similar(x2), x2r, ℂ)
            @test x2c ≈ x2

            # matrix multiplication is preserved
            x3 = randn(ComplexF64, 3, 3)
            x3r = Manifolds.realify(x3, ℂ)
            @test x2 * x3 ≈ Manifolds.unrealify!(similar(x2), x2r * x3r, ℂ)
        end
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

    include_test("essential_manifold.jl")
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
    include_test("approx_inverse_retraction.jl")

    # Lie groups and actions
    include_test("groups/groups_general.jl")
    include_test("groups/array_manifold.jl")
    include_test("groups/circle_group.jl")
    include_test("groups/translation_group.jl")
    include_test("groups/general_linear.jl")
    include_test("groups/special_linear.jl")
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
