include("utils.jl")

@testset "Euclidean" begin
    M = Manifolds.Euclidean(3)
    types = [Vector{Float64},
             SizedVector{3, Float64},
             MVector{3, Float64},
             Vector{Float32},
             SizedVector{3, Float32},
             MVector{3, Float32},
             Vector{Double64},
             MVector{3, Double64},
             SizedVector{3, Double64}]
    for T in types
        @testset "Type $T" begin
            pts = [convert(T, [1.0, 0.0, 0.0]),
                   convert(T, [0.0, 1.0, 0.0]),
                   convert(T, [0.0, 0.0, 1.0])]
            test_manifold(M,
                          pts,
                          test_reverse_diff = isa(T, Vector),
                          test_project_tangent = true,
                          point_distributions = [Manifolds.projected_distribution(M, Distributions.MvNormal(zero(pts[1]), 1.0))],
                          tvector_distributions = [Manifolds.normal_tvector_distribution(M, pts[1], 1.0)])
        end
    end

    M = Euclidean(2, 3)
    @testset "vee/hat" begin
        exp_dim = manifold_dimension(M)
        exp_size = (2, 3)
        v = randn(exp_dim)
        V = Manifolds.hat(M, I, v)
        @test length(V) == length(v)
        @test size(V) == exp_size
        @test Manifolds.vee(M, I, V) == v

        v = Manifolds.vee(M, I, V)
        @test length(v) == length(V)
        @test size(v) == (exp_dim,)
        @test Manifolds.hat(M, I, v) == V
    end
end
