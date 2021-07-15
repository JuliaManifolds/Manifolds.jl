include("../utils.jl")

@testset "Cholesky Space" begin
    M = Manifolds.CholeskySpace(3)
    @test repr(M) == "CholeskySpace(3)"

    types = [Matrix{Float64}]
    TEST_FLOAT32 && push!(types, Matrix{Float32})
    TEST_STATIC_SIZED && push!(types, MMatrix{3,3,Float64,9})

    for T in types
        A(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
        ptsF = [#
            cholesky([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1]).L,
            cholesky([2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1]).L,
            cholesky(
                A(π / 6) * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1] * transpose(A(π / 6)),
            ).L,
        ]
        pts = [convert(T, a) for a in ptsF]
        test_manifold(
            M,
            pts;
            test_injectivity_radius=false,
            test_default_vector_transport=true,
            vector_transport_methods=[
                ParallelTransport(),
                SchildsLadderTransport(),
                PoleLadderTransport(),
            ],
            test_forward_diff=false,
            test_reverse_diff=false,
            test_vee_hat=false,
            exp_log_atol_multiplier=8.0,
            test_inplace=true,
        )
    end
    @testset "Test Error cases in is_point and is_vector" begin
        pt1f = zeros(2, 3) # wrong size
        pt2f = [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 1.0] # nonpos diag
        pt3f = [2.0 0.0 1.0; 0.0 1.0 0.0; 0.0 0.0 4.0] # no lower and nonsym
        pt4 = [2.0 0.0 0.0; 1.0 2.0 0.0; 0.0 0.0 4.0]
        @test !is_point(M, pt1f)
        @test_throws DomainError is_point(M, pt1f, true)
        @test !is_point(M, pt2f)
        @test_throws DomainError is_point(M, pt2f, true)
        @test !is_point(M, pt3f)
        @test_throws DomainError is_point(M, pt3f, true)
        @test is_point(M, pt4)
        @test !is_vector(M, pt3f, pt1f)
        @test_throws DomainError is_vector(M, pt3f, pt1f, true)
        @test !is_vector(M, pt4, pt1f)
        @test_throws DomainError is_vector(M, pt4, pt1f, true)
        @test !is_vector(M, pt4, pt3f)
        @test_throws DomainError is_vector(M, pt4, pt3f, true)
        @test is_vector(M, pt4, pt2f)
    end
end
