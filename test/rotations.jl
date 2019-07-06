include("utils.jl")

@testset "Rotations" begin
    M = Manifolds.Rotations(2)

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

    for T in types
        angles = (0.0, π/2, 2π/3, π/4)
        pts = [convert(T, [cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]) for ϕ in angles]
        test_manifold(M, pts;
            test_reverse_diff = false,
            test_project_tangent = true,
            retraction_methods = retraction_methods,
            inverse_retraction_methods = inverse_retraction_methods,
            point_distributions = [Manifolds.normal_rotation_distribution(M, pts[1], 1.0)],
            tvector_distributions = [Manifolds.normal_tvector_distribution(M, pts[1], 1.0)])

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
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                point_distributions = [ptd],
                tvector_distributions = [tvd],
                exp_log_atol_multiplier = 6)
        end
    end

end
