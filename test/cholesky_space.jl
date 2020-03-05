include("utils.jl")

@testset "Cholesky Space" begin
    M = Manifolds.CholeskySpace(3)
    @test repr(M) == "CholeskySpace(3)"

    types = [
        Matrix{Float64},
        MMatrix{3, 3, Float64},
        Matrix{Float32},
    ]
    for T in types
        A(α) = [1. 0. 0.; 0. cos(α) sin(α); 0. -sin(α) cos(α)]
        ptsF = [#
            cholesky([1. 0. 0.; 0. 1. 0.; 0. 0. 1]).L,
            cholesky([2. 0. 0.; 0. 2. 0.; 0. 0. 1]).L,
            cholesky(A(π/6) * [1. 0. 0.; 0. 2. 0.; 0. 0. 1] * transpose(A(π/6))).L,
            ]
        pts = [convert(T, a) for a in ptsF]
        test_manifold(M, pts;
            test_injectivity_radius = false,
            test_vector_transport = true,
            test_forward_diff = false,
            test_reverse_diff = false,
            test_vee_hat = false,
            exp_log_atol_multiplier = 8.0,
        )
    end
    @testset "Test Error cases in is_manifold_point and is_tangent_vector" begin
        pt1f = zeros(2,3); # wrong size
        pt2f = [1. 0. 0.; 0. -1. 0.; 0. 0. 1.]; # nonpos diag
        pt3f = [2. 0. 1.; 0. 1. 0.; 0. 0. 4.]; # no lower and nonsym
        pt4 = [2. 0. 0.; 1. 2. 0.; 0. 0. 4.]
        @test !is_manifold_point(M, pt1f)
        @test_throws DomainError is_manifold_point(M, pt1f, true)
        @test !is_manifold_point(M, pt2f)
        @test_throws DomainError is_manifold_point(M, pt2f, true)
        @test !is_manifold_point(M,pt3f)
        @test_throws DomainError is_manifold_point(M,pt3f, true)
        @test is_manifold_point(M, pt4)
        @test !is_tangent_vector(M,pt3f, pt1f)
        @test_throws DomainError is_tangent_vector(M,pt3f, pt1f, true)
        @test !is_tangent_vector(M,pt4, pt1f)
        @test_throws DomainError is_tangent_vector(M,pt4, pt1f, true)
        @test !is_tangent_vector(M,pt4, pt3f)
        @test_throws DomainError is_tangent_vector(M,pt4, pt3f, true)
        @test is_tangent_vector(M,pt4, pt2f)
    end
end
