include("utils.jl")

@testset "Rotations" begin
    M = Manifolds.Rotations(2)
    @test representation_size(M) == (2,2)
    types = [Matrix{Float64},
             SizedMatrix{2, 2, Float64},
             MMatrix{2, 2, Float64},
             Matrix{Float32},
             SizedMatrix{2, 2, Float32},
             MMatrix{2, 2, Float32}]

    retraction_methods = [Manifolds.PolarRetraction(),
                          Manifolds.QRRetraction()]

    inverse_retraction_methods = [Manifolds.PolarInverseRetraction(),
                                  Manifolds.QRInverseRetraction()]

    @testset "vee/hat" begin
        M = Manifolds.Rotations(2)
        v = randn(1)
        V = Manifolds.hat(M, I, v)
        @test isa(V, MMatrix)
        @test norm(M, I, V) / sqrt(2) ≈ norm(v)
        @test Manifolds.vee(M, I, V) == v

        V = project_tangent(M, I, randn(2, 2))
        v = Manifolds.vee(M, I, V)
        @test isa(v, MVector)
        @test Manifolds.hat(M, I, v) == V
    end

    for T in types
        angles = (0.0, π/2, 2π/3, π/4)
        pts = [convert(T, [cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)]) for ϕ in angles]
        test_manifold(M, pts;
            test_reverse_diff = false,
            test_project_tangent = true,
            test_musical_isomorphisms = true,
            retraction_methods = retraction_methods,
            inverse_retraction_methods = inverse_retraction_methods,
            point_distributions = [Manifolds.normal_rotation_distribution(M, pts[1], 1.0)],
            tvector_distributions = [Manifolds.normal_tvector_distribution(M, pts[1], 1.0)])

        @testset "log edge cases" begin
            v = Manifolds.hat(M, pts[1], [Float64(π)])
            x = exp(M, pts[1], v)
            @test isapprox(x, exp(M, pts[1], log(M, pts[1], x)))
        end

        v = log(M, pts[1], pts[2])
        @test norm(M, pts[1], v) ≈ (angles[2] - angles[1])*sqrt(2)

        v14_polar = inverse_retract(M, pts[1], pts[4], Manifolds.PolarInverseRetraction())
        p4_polar = retract(M, pts[1], v14_polar, Manifolds.PolarRetraction())
        @test isapprox(M, pts[4], p4_polar)

        v14_qr = inverse_retract(M, pts[1], pts[4], Manifolds.QRInverseRetraction())
        p4_qr = retract(M, pts[1], v14_qr, Manifolds.QRRetraction())
        @test isapprox(M, pts[4], p4_qr)
    end

    @testset "Distribution tests" begin
        usd_mmatrix = Manifolds.normal_rotation_distribution(M, (@MMatrix [1.0 0.0; 0.0 1.0]), 1.0)
        @test isa(rand(usd_mmatrix), MMatrix)

        gtsd_mvector = Manifolds.normal_tvector_distribution(M, (@MMatrix [1.0 0.0; 0.0 1.0]), 1.0)
        @test isa(rand(gtsd_mvector), MMatrix)
    end

    Random.seed!(42)
    for n ∈ (3, 4, 5)
        @testset "Rotations: SO($n)" begin
            SOn = Manifolds.Rotations(n)
            ptd = Manifolds.normal_rotation_distribution(SOn, Matrix(1.0I, n, n), 1.0)
            tvd = Manifolds.normal_tvector_distribution(SOn, Matrix(1.0I, n, n), 1.0)
            pts = [rand(ptd) for _ in 1:3]
            test_manifold(SOn, pts;
                test_forward_diff = n==3,
                test_reverse_diff = false,
                test_musical_isomorphisms = true,
                test_mutating_rand = true,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                point_distributions = [ptd],
                tvector_distributions = [tvd],
                exp_log_atol_multiplier = 6)

            @testset "vee/hat" begin
                v = randn(manifold_dimension(SOn))
                V = Manifolds.hat(SOn, I, v)
                @test isa(V, MMatrix)
                @test norm(SOn, I, V) / sqrt(2) ≈ norm(v)
                @test Manifolds.vee(SOn, I, V) == v

                V = project_tangent(SOn, I, randn(n, n))
                v = Manifolds.vee(SOn, I, V)
                @test isa(v, MVector)
                @test Manifolds.hat(SOn, I, v) == V
            end

            if n == 4
                @testset "exp/log edge cases" begin
                    vs = [
                              [0, 0, π, 0, 0, π],  # θ = (π, π)
                              [0, 0, π, 0, 0, 0],  # θ = (π, 0)
                              [0, 0, π/2, 0, 0, π],  # θ = (π, π/2)
                              [0, 0, π, 0, 0, 0] ./ 2,  # θ = (π/2, 0)
                              [0, 0, π, 0, 0, π] ./ 2,  # θ = (π/2, π/2)
                              [0, 0, 0, 0, 0, 0],  # θ = (0, 0)
                              [0, 0, 1, 0, 0, 1] .* 1e-100, # α = β ≈ 0
                              [0, 0, 1, 0, 0, 1] .* 1e-6, # α = β ⩰ 0
                              [0, 0, 10, 0, 0, 1] .* 1e-6, # α ⪆ β ⩰ 0
                              [0, 0, π/4, 0, 0, π/4 - 1e-6], # α ⪆ β > 0
                         ]
                    for v in vs
                        @testset "rotation vector $v" begin
                            V = Manifolds.hat(SOn, I, v)
                            x = exp(V)
                            @test x ≈ exp(SOn, one(x), V)
                            @test ForwardDiff.derivative(t -> exp(SOn, one(x), t*V), 0) ≈ V
                            x2 = exp(log(SOn, one(x), x))
                            @test isapprox(x, x2; atol = 1e-6)
                        end
                    end
                end
            end

            v = Matrix(Manifolds.hat(SOn, pts[1], π * normalize(randn(manifold_dimension(SOn)))))
            x = exp(SOn, pts[1], v)
            v2 = log(SOn, pts[1], x)
            @test x ≈ exp(SOn, pts[1], v2)
        end
    end
    @testset "Test Manifold Point and Tangent Vector checks" begin
        for x in [1, [2. 0.;0. 1.], [1. 0.5; 0. 1.]]
            @test_throws DomainError is_manifold_point(M,x,true)
            @test !is_manifold_point(M,x)
        end
        x = one(zeros(2,2))
        @test is_manifold_point(M,x)
        @test is_manifold_point(M,x,true)
        for v in [1, [0. 1.;0. 0.]]
            @test_throws DomainError is_tangent_vector(M,x,v,true)
            @test !is_tangent_vector(M,x,v)
        end
        v = [0. 1.;-1. 0.]
        @test is_tangent_vector(M,x,v)
        @test is_tangent_vector(M,x,v,true)
    end
end
