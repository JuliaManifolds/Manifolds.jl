include("../utils.jl")

@testset "Rotations" begin
    M = Manifolds.Rotations(2)
    @test repr(M) == "Rotations(2)"
    @test representation_size(M) == (2, 2)
    @test injectivity_radius(M) == π * sqrt(2.0)
    @test injectivity_radius(M, [1.0 0.0; 0.0 1.0]) == π * sqrt(2.0)
    @test injectivity_radius(M, ExponentialRetraction()) == π * sqrt(2.0)
    @test injectivity_radius(M, [1.0 0.0; 0.0 1.0], ExponentialRetraction()) ==
          π * sqrt(2.0)
    @test injectivity_radius(M, PolarRetraction()) ≈ π / sqrt(2)
    @test injectivity_radius(M, [1.0 0.0; 0.0 1.0], PolarRetraction()) ≈ π / sqrt(2)
    @test get_embedding(M) == Euclidean(2, 2)
    types = [Matrix{Float64}]
    TEST_FLOAT32 && push!(types, Matrix{Float32})
    TEST_STATIC_SIZED && push!(types, MMatrix{2,2,Float64,4})
    retraction_methods = [Manifolds.PolarRetraction(), Manifolds.QRRetraction()]

    inverse_retraction_methods =
        [Manifolds.PolarInverseRetraction(), Manifolds.QRInverseRetraction()]

    basis_types = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))

    @testset "vee/hat" begin
        M = Manifolds.Rotations(2)
        Xf = [1.23]
        p = Matrix{Float64}(I, 2, 2)
        X = Manifolds.hat(M, p, Xf)
        @test isa(X, AbstractMatrix)
        @test norm(M, p, X) / sqrt(2) ≈ norm(Xf)
        @test Manifolds.vee(M, p, X) == Xf

        X = project(M, p, randn(2, 2))
        Xf = Manifolds.vee(M, p, X)
        @test isa(Xf, AbstractVector)
        @test Manifolds.hat(M, p, Xf) == X
    end

    for T in types
        angles = (0.0, π / 2, 2π / 3, π / 4)
        pts = [convert(T, [cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)]) for ϕ in angles]
        test_manifold(
            M,
            pts;
            test_injectivity_radius=false,
            test_project_tangent=true,
            test_musical_isomorphisms=true,
            test_default_vector_transport=true,
            retraction_methods=retraction_methods,
            inverse_retraction_methods=inverse_retraction_methods,
            point_distributions=[Manifolds.normal_rotation_distribution(M, pts[1], 1.0)],
            tvector_distributions=[Manifolds.normal_tvector_distribution(M, pts[1], 1.0)],
            basis_types_to_from=basis_types,
            test_inplace=true,
            test_rand_point=true,
        )

        @testset "log edge cases" begin
            X = Manifolds.hat(M, pts[1], [Float64(π)])
            p = exp(M, pts[1], X)
            @test isapprox(p, exp(M, pts[1], log(M, pts[1], p)))
        end

        X = log(M, pts[1], pts[2])
        @test norm(M, pts[1], X) ≈ (angles[2] - angles[1]) * sqrt(2)

        # check that exp! does not have a side effect
        q = allocate(pts[1])
        copyto!(M, q, pts[1])
        q2 = exp(M, pts[1], X)
        exp!(M, q, q, X)
        @test norm(q - q2) ≈ 0

        X14_polar = inverse_retract(M, pts[1], pts[4], Manifolds.PolarInverseRetraction())
        p4_polar = retract(M, pts[1], X14_polar, Manifolds.PolarRetraction())
        @test isapprox(M, pts[4], p4_polar)

        X14_qr = inverse_retract(M, pts[1], pts[4], Manifolds.QRInverseRetraction())
        p4_qr = retract(M, pts[1], X14_qr, Manifolds.QRRetraction())
        @test isapprox(M, pts[4], p4_qr)
    end

    @testset "Distribution tests" begin
        usd_mmatrix =
            Manifolds.normal_rotation_distribution(M, (@MMatrix [1.0 0.0; 0.0 1.0]), 1.0)
        @test isa(rand(usd_mmatrix), MMatrix)

        usd1_mmatrix =
            Manifolds.normal_rotation_distribution(Rotations(1), (@MMatrix [1.0]), 1.0)
        @test isa(rand(usd1_mmatrix), MMatrix)
        @test rand(usd1_mmatrix) == @MMatrix [1.0]

        gtsd_mvector =
            Manifolds.normal_tvector_distribution(M, (@MMatrix [1.0 0.0; 0.0 1.0]), 1.0)
        @test isa(rand(gtsd_mvector), MMatrix)
    end

    Random.seed!(42)
    for n in (3, 4, 5)
        @testset "Rotations: SO($n)" begin
            SOn = Manifolds.Rotations(n)
            ptd = Manifolds.normal_rotation_distribution(SOn, Matrix(1.0I, n, n), 1.0)
            tvd = Manifolds.normal_tvector_distribution(SOn, Matrix(1.0I, n, n), 1.0)
            pts = [rand(ptd) for _ in 1:3]
            test_manifold(
                SOn,
                pts;
                test_injectivity_radius=false,
                test_musical_isomorphisms=true,
                test_mutating_rand=true,
                test_default_vector_transport=true,
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                point_distributions=[ptd],
                tvector_distributions=[tvd],
                basis_types_to_from=basis_types,
                exp_log_atol_multiplier=250,
                retraction_atol_multiplier=12,
                test_inplace=true,
                test_rand_point=true,
                test_rand_tvector=true,
            )

            @testset "vee/hat" begin
                p = Matrix(1.0I, n, n)
                Xf = randn(manifold_dimension(SOn))
                X = Manifolds.hat(SOn, p, Xf)
                @test isa(X, AbstractMatrix)
                @test norm(SOn, p, X) / sqrt(2) ≈ norm(Xf)
                @test Manifolds.vee(SOn, p, X) == Xf

                X = project(SOn, p, randn(n, n))
                Xf = Manifolds.vee(SOn, p, X)
                @test isa(Xf, AbstractVector)
                @test Manifolds.hat(SOn, p, Xf) ≈ X
            end

            if n == 4
                @testset "exp/log edge cases" begin
                    Xs = [
                        [0, 0, π, 0, 0, π],  # θ = (π, π)
                        [0, 0, π, 0, 0, 0],  # θ = (π, 0)
                        [0, 0, π / 2, 0, 0, π],  # θ = (π, π/2)
                        [0, 0, π, 0, 0, 0] ./ 2,  # θ = (π/2, 0)
                        [0, 0, π, 0, 0, π] ./ 2,  # θ = (π/2, π/2)
                        [0, 0, 0, 0, 0, 0],  # θ = (0, 0)
                        [0, 0, 1, 0, 0, 1] .* 1e-100, # α = β ≈ 0
                        [0, 0, 1, 0, 0, 1] .* 1e-6, # α = β ⩰ 0
                        [0, 0, 10, 0, 0, 1] .* 1e-6, # α ⪆ β ⩰ 0
                        [0, 0, π / 4, 0, 0, π / 4 - 1e-6], # α ⪆ β > 0
                    ]
                    for Xf in Xs
                        @testset "rotation vector $Xf" begin
                            X = Manifolds.hat(SOn, Matrix(1.0I, n, n), Xf)
                            p = exp(X)
                            @test p ≈ exp(SOn, one(p), X)
                            @test ForwardDiff.derivative(t -> exp(SOn, one(p), t * X), 0) ≈
                                  X
                            p2 = exp(log(SOn, one(p), p))
                            @test isapprox(p, p2; atol=1e-6)
                        end
                    end
                end
            end

            X = Matrix(
                Manifolds.hat(SOn, pts[1], π * normalize(randn(manifold_dimension(SOn)))),
            )
            p = exp(SOn, pts[1], X)
            X2 = log(SOn, pts[1], p)
            @test p ≈ exp(SOn, pts[1], X2)
        end
    end
    @testset "Test AbstractManifold Point and Tangent Vector checks" begin
        M = Manifolds.Rotations(2)
        for p in [1, [2.0 0.0; 0.0 1.0], [1.0 0.5; 0.0 1.0]]
            @test_throws DomainError is_point(M, p, true)
            @test !is_point(M, p)
        end
        p = one(zeros(2, 2))
        @test is_point(M, p)
        @test is_point(M, p, true)
        for X in [1, [0.0 1.0; 0.0 0.0]]
            @test_throws DomainError is_vector(M, p, X, true)
            @test !is_vector(M, p, X)
        end
        X = [0.0 1.0; -1.0 0.0]
        @test is_vector(M, p, X)
        @test is_vector(M, p, X, true)
    end
    @testset "Project point" begin
        M = Manifolds.Rotations(2)
        p = Matrix{Float64}(I, 2, 2)
        p1 = project(M, p)
        @test is_point(M, p1, true)

        M = Manifolds.Rotations(3)
        p = collect(reshape(1.0:9.0, (3, 3)))
        p2 = project(M, p)
        @test is_point(M, p2, true)

        rng = MersenneTwister(44)
        x3 = project(M, randn(rng, 3, 3))
        @test is_point(M, x3, true)
    end
    @testset "Convert from Lie algebra representation of tangents to Riemannian submanifold representation" begin
        M = Manifolds.Rotations(3)
        p = project(M, collect(reshape(1.0:9.0, (3, 3))))
        x = [[0, -1, 3] [1, 0, 2] [-3, -2, 0]]
        @test is_vector(M, p, x, true)
        @test embed(M, p, x) == p * x
        Y = zeros((3, 3))
        embed!(M, Y, p, x)
        @test Y == p * x
        @test Y ≈ p * (p'Y - Y'p) / 2
    end
    @testset "Edge cases of Rotations" begin
        @test_throws OutOfInjectivityRadiusError inverse_retract(
            Rotations(2),
            [1.0 0.0; 0.0 1.0],
            [1.0 0.0; 0.0 -1.0],
            PolarInverseRetraction(),
        )
    end
end
