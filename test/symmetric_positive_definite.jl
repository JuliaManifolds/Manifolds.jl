include("utils.jl")
@testset "Symmetric Positive Definite Matrices" begin
    M1 = Manifolds.SymmetricPositiveDefinite(3)
    M2 = MetricManifold(Manifolds.SymmetricPositiveDefinite(3), Manifolds.LinearAffineMetric())
    M3 = MetricManifold(Manifolds.SymmetricPositiveDefinite(3), Manifolds.LogCholeskyMetric())
    M4 = MetricManifold(Manifolds.SymmetricPositiveDefinite(3), Manifolds.LogEuclideanMetric())

    @test is_default_metric(M2) == Val{true}()
    @test is_default_metric(M1, Manifolds.LinearAffineMetric()) == Val{true}()
    @test is_default_metric(M1, Manifolds.LogCholeskyMetric()) == Val{false}()
    @test is_default_metric(M3) == Val{false}()

    metrics = [M1, M2, M3]
    types = [ Matrix{Float32},
            Matrix{Float64},
            MMatrix{3,3,Float32},
            MMatrix{3,3,Float64}
        ]
    for M in metrics
        @testset "$(typeof(M))" begin
            @test representation_size(M) == (3,3)
            for T in types
                exp_log_atol_multiplier = 8.0
                if T <: MMatrix{3,3,Float64}
                    # eigendecomposition of 3x3 SPD matrices from StaticArrays is not very accurate
                    exp_log_atol_multiplier = 5.0e7
                end
                if M == M3 && T <: MMatrix
                    # Cholesky or something does not work in vector_transport yet for MMatrix
                    continue
                end
                A(α) = [1. 0. 0.; 0. cos(α) sin(α); 0. -sin(α) cos(α)]
                ptsF = [#
                    [1. 0. 0.; 0. 1. 0.; 0. 0. 1],
                    [2. 0. 0.; 0. 2. 0.; 0. 0. 1],
                    A(π/6) * [1. 0. 0.; 0. 2. 0.; 0. 0. 1] * transpose(A(π/6)),
                ]
                pts = [convert(T, a) for a in ptsF]
                test_manifold(M, pts;
                    test_vector_transport = true,
                    test_forward_diff = false,
                    test_reverse_diff = false,
                    exp_log_atol_multiplier = exp_log_atol_multiplier
                )
            end
            @testset "Test Error cases in is_manifold_point and is_tangent_vector" begin
                pt1f = zeros(2,3); # wrong size
                pt2f = [1. 0. 0.; 0. 0. 0.; 0. 0. 1.]; # not positive Definite
                pt3f = [2. 0. 1.; 0. 1. 0.; 0. 0. 4.]; # not symmetric
                pt4 = [2. 1. 0.; 1. 2. 0.; 0. 0. 4.]
                @test !is_manifold_point(M,pt1f)
                @test !is_manifold_point(M,pt2f)
                @test !is_manifold_point(M,pt3f)
                @test is_manifold_point(M, pt4)
                @test !is_tangent_vector(M,pt4, pt1f)
                @test is_tangent_vector(M,pt4, pt2f)
                @test !is_tangent_vector(M,pt4, pt3f)
            end
        end
    end
    x = [1. 0. 0.; 0. 1. 0.; 0. 0. 1]
    y = [2. 0. 0.; 0. 2. 0.; 0. 0. 1]
    @testset "Convert SPD to Cholesky" begin
        v = log(M1,x,y)
        (l,w) = Manifolds.spd_to_cholesky(x,v)
        (xs,vs) = Manifolds.cholesky_to_spd(l,w)
        @test isapprox(xs,x)
        @test isapprox(vs,v)
    end
    @testset "Preliminary tests for LogEuclidean" begin
        @test representation_size(M4) == (3,3)
        @test isapprox( distance(M4,x,y), sqrt(2)log(2))
        @test manifold_dimension(M4) == manifold_dimension(M1)
    end
    @testset "Test for tangent ONB on LinearAffineMetric" begin
        v = log(M2,x,y)
        (X,k) = tangent_orthonormal_basis(base_manifold(M2),x,v)
        @test isapprox(0.0,first(k))
        for i = 1:length(X)
            @test isapprox(1.0, norm(M2,x,X[1]))
            for j=i+1:length(X)
                @test isapprox(0.0, inner(M2,x,X[1],X[2]))
            end
        end
    end
end
