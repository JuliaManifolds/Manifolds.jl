include("utils.jl")

@testset "Symmetric Positive Semidefinite Matrices of Fixed Rank" begin
    @testset "Real Matrices" begin
        M = SymmetricPositiveSemidefiniteFixedRank(4, 2)
        @test repr(M) == "SymmetricPositiveSemidefiniteFixedRank(4, 2, ℝ)"
        @test manifold_dimension(M) == 7
        q = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]
        @test is_manifold_point(M, q)
        Y = [1.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]
        @test_throws DomainError is_manifold_point(M, Y, true)
        @test is_tangent_vector(M, q, Y)
        q2 = [2.0 1.0; 0.0 0.0; 0.0 1.0; 0.0 0.0]
        q3 = [0.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 0.0]

        types = [Matrix{Float64}]
        TEST_FLOAT32 && push!(types, Matrix{Float32})
        TEST_STATIC_SIZED && push!(types, MMatrix{4,2,Float64,8})
        for T in types
            pts = [convert(T, q), convert(T, q2), convert(T, q3)]
            @testset "Type $T" begin
                test_manifold(
                    M,
                    pts,
                    exp_log_atol_multiplier = 5,
                    is_tangent_atol_multiplier = 5,
                    test_forward_diff = false,
                    test_reverse_diff = false,
                    test_project_tangent = true,
                    vector_transport_methods = [ProjectionTransport()],
                )
            end
        end
    end
    @testset "Complex Matrices" begin
        M = SymmetricPositiveSemidefiniteFixedRank(4, 2, ℂ)
        @test repr(M) == "SymmetricPositiveSemidefiniteFixedRank(4, 2, ℂ)"
        @test manifold_dimension(M) == 12

    end
end
