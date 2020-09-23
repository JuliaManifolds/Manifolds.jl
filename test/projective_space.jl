include("utils.jl")

@testset "ProjectiveSpace" begin
    @testset "Real" begin
        M = ProjectiveSpace(2)
        @testset "Basics" begin
            @test repr(M) == "ProjectiveSpace(2, ℝ)"
            @test representation_size(M) == (3,)
            @test manifold_dimension(M) == 2
            @test !is_manifold_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_tangent_vector(M, [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            @test_throws DomainError is_manifold_point(M, [2.0, 0.0, 0.0], true)
            @test !is_manifold_point(M, [2.0, 0.0, 0.0])
            @test !is_tangent_vector(M, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            @test_throws DomainError is_tangent_vector(
                M,
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                true,
            )
            @test injectivity_radius(M) == π / 2
            @test injectivity_radius(M, ExponentialRetraction()) == π / 2
            @test injectivity_radius(M, [1.0, 0.0, 0.0]) == π / 2
            @test injectivity_radius(M, [1.0, 0.0, 0.0], ExponentialRetraction()) == π / 2
        end
        types = [Vector{Float64}]
        TEST_STATIC_SIZED && push!(types, MVector{3,Float64})

        TEST_FLOAT32 && push!(types, Vector{Float32})
        basis_types = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))
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
                test_injectivity_radius = false,
                test_project_point = true,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_default_vector_transport = true,
                vector_transport_methods = [
                    ParallelTransport(),
                    ProjectionTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                point_distributions = [Manifolds.uniform_distribution(M, pts[1])],
                tvector_distributions = [Manifolds.normal_tvector_distribution(
                    M,
                    pts[1],
                    1.0,
                )],
                test_forward_diff = false,
                test_reverse_diff = false,
                basis_types_vecs = (
                    DiagonalizingOrthonormalBasis([0.0, 1.0, 2.0]),
                    basis_types...,
                ),
                basis_types_to_from = basis_types,
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
                is_tangent_atol_multiplier = 1,
            )
        end

        @testset "equivalence" begin
            x = [1.0, 0.0, 0.0]
            v = [0.0, 1.0, 0.0]
            @test isapprox(M, x, -x)
            @test isapprox(M, x, exp(M, x, π * v))
            @test log(M, x, -x) ≈ zero(v)
            @test isapprox(M, -x, vector_transport_to(M, x, v, -x), -v)
        end

        @testset "Distribution MVector tests" begin
            upd_mvector = Manifolds.uniform_distribution(M, @MVector [1.0, 0.0, 0.0])
            @test isa(rand(upd_mvector), MVector)

            gtpd_mvector =
                Manifolds.normal_tvector_distribution(M, (@MVector [1.0, 0.0, 0.0]), 1.0)
            @test isa(rand(gtpd_mvector), MVector)
        end
    end

    @testset "Complex" begin
        M = ProjectiveSpace(2, ℂ)
        @testset "Basics" begin
            @test repr(M) == "ProjectiveSpace(2, ℂ)"
            @test representation_size(M) == (3,)
            @test manifold_dimension(M) == 5
            @test Manifolds.allocation_promotion_function(M, exp!, (1,)) == complex
            @test !is_manifold_point(M, [1.0 + 0im, 0.0, 0.0, 0.0])
            @test !is_tangent_vector(M, [1.0 + 0im, 0.0, 0.0, 0.0], [0.0 + 0im, 1.0, 0.0])
            @test_throws DomainError is_manifold_point(M, [1.0, im, 0.0], true)
            @test !is_manifold_point(M, [1.0, im, 0.0])
            @test !is_tangent_vector(M, [1.0 + 0im, 0.0, 0.0], [1.0 + 0im, 0.0, 0.0])
            @test !is_tangent_vector(M, [1.0 + 0im, 0.0, 0.0], [-0.5im, 0.0, 0.0])
            @test_throws DomainError is_tangent_vector(
                M,
                [1.0 + 0im, 0.0, 0.0],
                [1.0 + 0im, 0.0, 0.0],
                true,
            )
            @test_throws DomainError is_tangent_vector(
                M,
                [1.0 + 0im, 0.0, 0.0],
                [-0.5im, 0.0, 0.0],
                true,
            )
            @test injectivity_radius(M) == π / 2
            @test injectivity_radius(M, ExponentialRetraction()) == π / 2
            @test injectivity_radius(M, [1.0 + 0im, 0.0, 0.0]) == π / 2
            @test injectivity_radius(M, [1.0 + 0im, 0.0, 0.0], ExponentialRetraction()) == π / 2
        end
        types = [Vector{ComplexF64}]
        @testset "Type $T" for T in types
            x = [0.5 + 0.5im, 0.5 + 0.5im, 0]
            v = [0.0, 0.0, 1.0 - im]
            y = im * exp(M, x, v)
            w = [0.5, -0.5, 0.5im]
            z = (sqrt(0.5) - sqrt(0.5) * im) * exp(M, x, w)
            pts = convert.(T, [x, y, z])
            test_manifold(
                M,
                pts,
                test_injectivity_radius = false,
                test_project_point = true,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_default_vector_transport = true,
                vector_transport_methods = [
                    ParallelTransport(),
                    ProjectionTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
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
                is_tangent_atol_multiplier = 1,
                exp_log_atol_multiplier = 10.0^3,
            )
        end

        @testset "equivalence" begin
            x = [1.0 + 0im, 0.0, 0.0]
            v = [0.0, im, 0.0]
            s = sqrt(0.5) - sqrt(0.5)*im
            @test isapprox(M, x, s * x)
            @test isapprox(M, x, exp(M, x, π * v))
            @test log(M, x, s * x) ≈ zero(v)
            @test isapprox(M, s * x, vector_transport_to(M, x, v, s * x), s * v)
        end
    end

    @testset "ArrayProjectiveSpace" begin
        M = ArrayProjectiveSpace(2, 2; field = ℝ)
        @test repr(M) == "ArrayProjectiveSpace(2, 2; field = ℝ)"
        @test typeof(get_embedding(M)) === Euclidean{Tuple{2,2},ℝ}
        @test representation_size(M) == (2, 2)
        p = ones(2, 2)
        q = project(M, p)
        @test is_manifold_point(M, q)
        Y = [1.0 0.0; 0.0 1.1]
        X = project(M, q, Y)
        @test is_tangent_vector(M, q, X)

        M = ArrayProjectiveSpace(2, 2; field = ℂ)
        @test repr(M) == "ArrayProjectiveSpace(2, 2; field = ℂ)"
        @test typeof(get_embedding(M)) === Euclidean{Tuple{2,2},ℂ}
        @test representation_size(M) == (2, 2)
    end
end
