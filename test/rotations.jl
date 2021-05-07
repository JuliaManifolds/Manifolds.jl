include("utils.jl")

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
    types = [Matrix{Float64}]
    TEST_FLOAT32 && push!(types, Matrix{Float32})
    TEST_STATIC_SIZED && push!(types, MMatrix{2,2,Float64,4})
    retraction_methods = [Manifolds.PolarRetraction(), Manifolds.QRRetraction()]

    inverse_retraction_methods =
        [Manifolds.PolarInverseRetraction(), Manifolds.QRInverseRetraction()]

    basis_types = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))

    @testset "vee/hat" begin
        M = Manifolds.Rotations(2)
        v = [1.23]
        x = Matrix{Float64}(I, 2, 2)
        V = Manifolds.hat(M, x, v)
        @test isa(V, AbstractMatrix)
        @test norm(M, x, V) / sqrt(2) ≈ norm(v)
        @test Manifolds.vee(M, x, V) == v

        V = project(M, x, randn(2, 2))
        v = Manifolds.vee(M, x, V)
        @test isa(v, AbstractVector)
        @test Manifolds.hat(M, x, v) == V
    end

    for T in types
        angles = (0.0, π / 2, 2π / 3, π / 4)
        pts = [convert(T, [cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)]) for ϕ in angles]
        test_manifold(
            M,
            pts;
            test_reverse_diff=false,
            test_injectivity_radius=false,
            test_project_tangent=true,
            test_musical_isomorphisms=true,
            retraction_methods=retraction_methods,
            inverse_retraction_methods=inverse_retraction_methods,
            point_distributions=[Manifolds.normal_rotation_distribution(M, pts[1], 1.0)],
            tvector_distributions=[Manifolds.normal_tvector_distribution(M, pts[1], 1.0)],
            basis_types_to_from=basis_types,
        )

        @testset "log edge cases" begin
            v = Manifolds.hat(M, pts[1], [Float64(π)])
            x = exp(M, pts[1], v)
            @test isapprox(x, exp(M, pts[1], log(M, pts[1], x)))
        end

        v = log(M, pts[1], pts[2])
        @test norm(M, pts[1], v) ≈ (angles[2] - angles[1]) * sqrt(2)

        v14_polar = inverse_retract(M, pts[1], pts[4], Manifolds.PolarInverseRetraction())
        p4_polar = retract(M, pts[1], v14_polar, Manifolds.PolarRetraction())
        @test isapprox(M, pts[4], p4_polar)

        v14_qr = inverse_retract(M, pts[1], pts[4], Manifolds.QRInverseRetraction())
        p4_qr = retract(M, pts[1], v14_qr, Manifolds.QRRetraction())
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
                test_forward_diff=n == 3,
                test_reverse_diff=false,
                test_injectivity_radius=false,
                test_musical_isomorphisms=true,
                test_mutating_rand=true,
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                point_distributions=[ptd],
                tvector_distributions=[tvd],
                basis_types_to_from=basis_types,
                exp_log_atol_multiplier=20,
                retraction_atol_multiplier=12,
            )

            @testset "vee/hat" begin
                x = Matrix(1.0I, n, n)
                v = randn(manifold_dimension(SOn))
                V = Manifolds.hat(SOn, x, v)
                @test isa(V, AbstractMatrix)
                @test norm(SOn, x, V) / sqrt(2) ≈ norm(v)
                @test Manifolds.vee(SOn, x, V) == v

                V = project(SOn, x, randn(n, n))
                v = Manifolds.vee(SOn, x, V)
                @test isa(v, AbstractVector)
                @test Manifolds.hat(SOn, x, v) ≈ V
            end

            if n == 4
                @testset "exp/log edge cases" begin
                    vs = [
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
                    for v in vs
                        @testset "rotation vector $v" begin
                            V = Manifolds.hat(SOn, Matrix(1.0I, n, n), v)
                            x = exp(V)
                            @test x ≈ exp(SOn, one(x), V)
                            @test ForwardDiff.derivative(t -> exp(SOn, one(x), t * V), 0) ≈
                                  V
                            x2 = exp(log(SOn, one(x), x))
                            @test isapprox(x, x2; atol=1e-6)
                        end
                    end
                end
            end

            v = Matrix(
                Manifolds.hat(SOn, pts[1], π * normalize(randn(manifold_dimension(SOn)))),
            )
            x = exp(SOn, pts[1], v)
            v2 = log(SOn, pts[1], x)
            @test x ≈ exp(SOn, pts[1], v2)
        end
    end
    @testset "Test AbstractManifold Point and Tangent Vector checks" begin
        M = Manifolds.Rotations(2)
        for x in [1, [2.0 0.0; 0.0 1.0], [1.0 0.5; 0.0 1.0]]
            @test_throws DomainError is_point(M, x, true)
            @test !is_point(M, x)
        end
        x = one(zeros(2, 2))
        @test is_point(M, x)
        @test is_point(M, x, true)
        for v in [1, [0.0 1.0; 0.0 0.0]]
            @test_throws DomainError is_tangent_vector(M, x, v, true)
            @test !is_tangent_vector(M, x, v)
        end
        v = [0.0 1.0; -1.0 0.0]
        @test is_tangent_vector(M, x, v)
        @test is_tangent_vector(M, x, v, true)
    end
    @testset "Project point" begin
        M = Manifolds.Rotations(2)
        x = Matrix{Float64}(I, 2, 2)
        x1 = project(M, x)
        @test is_point(M, x1, true)

        M = Manifolds.Rotations(3)
        x = collect(reshape(1.0:9.0, (3, 3)))
        x2 = project(M, x)
        @test is_point(M, x2, true)

        rng = MersenneTwister(44)
        x3 = project(M, randn(rng, 3, 3))
        @test is_point(M, x3, true)
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
