include("utils.jl")

@testset "Projective" begin
    @testset "Real" begin
        M = Projective(2)
        @testset "Basics" begin
            @test repr(M) == "Projective(2, ℝ)"
            @test representation_size(M) == (3,)
            @test manifold_dimension(M) == 2
            @test !is_manifold_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_tangent_vector(M, [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            @test injectivity_radius(M) == π / 2
            @test injectivity_radius(M, ExponentialRetraction()) == π / 2
            @test injectivity_radius(M, [1.0, 0.0, 0.0]) == π / 2
            @test injectivity_radius(M, [1.0, 0.0, 0.0], ExponentialRetraction()) == π / 2
        end
        types = [Vector{Float64}]
        TEST_STATIC_SIZED && push!(types, MVector{3,Float64})

        TEST_FLOAT32 && push!(types, Vector{Float32})
        basis_types = (ProjectedOrthonormalBasis(:gram_schmidt),)
        @testset "Type $T" for T in types
            x = [1.0, 0.0, 0.0]
            v = [0.0, 1.0, 0.0]
            y = exp(M, x, v)
            w = [0.0, 1.0, -1.0]
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
                retraction_methods = [
                    ProjectionRetraction(),
                    PolarRetraction(),
                    QRRetraction(),
                ],
                inverse_retraction_methods = [
                    ProjectionInverseRetraction(),
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
            ugd_mmatrix = Manifolds.uniform_distribution(M, @MVector [1.0, 0.0, 0.0])
            @test isa(rand(ugd_mmatrix), MVector)
        end
    end
end
