include("utils.jl")

@testset "Grassmann" begin
    @testset "Real" begin
        M = Grassmann(3, 2)
        @testset "Basics" begin
            @test repr(M) == "Grassmann(3, 2, ℝ)"
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 2
            @test !is_manifold_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_tangent_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0.0, 0.0, 1.0, 0.0])
            @test_throws DomainError is_manifold_point(M, [2.0 0.0; 0.0 1.0; 0.0 0.0], true)
            @test_throws DomainError is_tangent_vector(
                M,
                [2.0 0.0; 0.0 1.0; 0.0 0.0],
                zeros(3, 2),
                true,
            )
            @test_throws DomainError is_tangent_vector(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                ones(3, 2),
                true,
            )
            @test is_manifold_point(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], true)
            @test_throws DomainError is_manifold_point(
                M,
                1im * [1.0 0.0; 0.0 1.0; 0.0 0.0],
                true,
            )
            @test is_tangent_vector(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                zero_tangent_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0]),
                true,
            )
            @test_throws DomainError is_tangent_vector(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                1im * zero_tangent_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0]),
                true,
            )
            @test injectivity_radius(M) == π / 2
            @test injectivity_radius(M, ExponentialRetraction()) == π / 2
            @test injectivity_radius(M, [1.0 0.0; 0.0 1.0; 0.0 0.0]) == π / 2
            @test injectivity_radius(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                ExponentialRetraction(),
            ) == π / 2
        end
        types = [Matrix{Float64}]
        TEST_STATIC_SIZED && push!(types, MMatrix{3,2,Float64,6})

        TEST_FLOAT32 && push!(types, Matrix{Float32})
        basis_types = (ProjectedOrthonormalBasis(:gram_schmidt),)
        @testset "Type $T" for T in types
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            v = [0.0 0.0; 0.0 0.0; 0.0 1.0]
            y = exp(M, x, v)
            w = [0.0 1.0; -1.0 0.0; 1.0 0.0]
            z = exp(M, x, w)
            pts = convert.(T, [x, y, z])
            test_manifold(
                M,
                pts,
                test_exp_log = true,
                test_injectivity_radius = false,
                test_project_tangent = true,
                test_default_vector_transport = false,
                point_distributions = [Manifolds.uniform_distribution(M, pts[1])],
                test_forward_diff = false,
                test_reverse_diff = false,
                test_vee_hat = false,
                retraction_methods = [PolarRetraction(), QRRetraction()],
                inverse_retraction_methods = [
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                ],
                #basis_types_vecs = basis_types,
                exp_log_atol_multiplier = 10.0,
                is_tangent_atol_multiplier = 20.0,
            )

            @testset "inner/norm" begin
                v1 = inverse_retract(M, pts[1], pts[2], PolarInverseRetraction())
                v2 = inverse_retract(M, pts[1], pts[3], PolarInverseRetraction())

                @test real(inner(M, pts[1], v1, v2)) ≈ real(inner(M, pts[1], v2, v1))
                @test imag(inner(M, pts[1], v1, v2)) ≈ -imag(inner(M, pts[1], v2, v1))
                @test imag(inner(M, pts[1], v1, v1)) ≈ 0

                @test norm(M, pts[1], v1) isa Real
                @test norm(M, pts[1], v1) ≈ sqrt(inner(M, pts[1], v1, v1))
            end
        end

        @testset "Distribution tests" begin
            ugd_mmatrix = Manifolds.uniform_distribution(M, @MMatrix [
                1.0 0.0
                0.0 1.0
                0.0 0.0
            ])
            @test isa(rand(ugd_mmatrix), MMatrix)
        end

        @testset "vector transport" begin
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            v = [0.0 0.0; 0.0 0.0; 0.0 1.0]
            y = exp(M, x, v)
            @test vector_transport_to(M, x, v, y, ProjectionTransport()) == project(M, y, v)
            @test is_tangent_vector(
                M,
                y,
                vector_transport_to(M, x, v, y, ProjectionTransport()),
                true;
                atol = 10^-15,
            )
        end
    end

    @testset "Complex" begin
        M = Grassmann(3, 2, ℂ)
        @testset "Basics" begin
            @test repr(M) == "Grassmann(3, 2, ℂ)"
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 4
            @test !is_manifold_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_tangent_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0.0, 0.0, 1.0, 0.0])
            @test Manifolds.allocation_promotion_function(M, exp!, (1,)) == complex
            @test_throws DomainError is_manifold_point(M, [2.0 0.0; 0.0 1.0; 0.0 0.0], true)
            @test_throws DomainError is_tangent_vector(
                M,
                [2.0 0.0; 0.0 1.0; 0.0 0.0],
                zeros(3, 2),
                true,
            )
            @test_throws DomainError is_tangent_vector(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                ones(3, 2),
                true,
            )
            @test is_tangent_vector(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                1im * zero_tangent_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0]),
            )
            @test is_manifold_point(M, [1.0 0.0; 0.0 1.0; 0.0 0.0])
            @test injectivity_radius(M) == π / 2
        end
        types = [Matrix{ComplexF64}]
        @testset "Type $T" for T in types
            x = [0.5 + 0.5im 0.5 + 0.5im; 0.5 + 0.5im -0.5 - 0.5im; 0.0 0.0]
            v = [0.0 0.0; 0.0 0.0; 0.0 1.0]
            y = exp(M, x, v)
            w = [0.0 1.0; -1.0 0.0; 1.0 0.0]
            z = exp(M, x, w)
            pts = convert.(T, [x, y, z])
            test_manifold(
                M,
                pts,
                test_exp_log = true,
                test_injectivity_radius = false,
                test_project_tangent = true,
                test_default_vector_transport = false,
                test_forward_diff = false,
                test_reverse_diff = false,
                test_vee_hat = false,
                retraction_methods = [PolarRetraction(), QRRetraction()],
                inverse_retraction_methods = [
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                ],
                exp_log_atol_multiplier = 10.0,
                is_tangent_atol_multiplier = 20.0,
            )

            @testset "inner/norm" begin
                v1 = inverse_retract(M, pts[1], pts[2], PolarInverseRetraction())
                v2 = inverse_retract(M, pts[1], pts[3], PolarInverseRetraction())

                @test real(inner(M, pts[1], v1, v2)) ≈ real(inner(M, pts[1], v2, v1))
                @test imag(inner(M, pts[1], v1, v2)) ≈ -imag(inner(M, pts[1], v2, v1))
                @test imag(inner(M, pts[1], v1, v1)) ≈ 0

                @test norm(M, pts[1], v1) isa Real
                @test norm(M, pts[1], v1) ≈ sqrt(inner(M, pts[1], v1, v1))
            end
        end
    end
end
