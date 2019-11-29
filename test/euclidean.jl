include("utils.jl")

@testset "Euclidean" begin
    manifolds = [
        Manifolds.Euclidean(3),
        Manifolds.MetricManifold(Manifolds.Euclidean(3),Manifolds.EuclideanMetric())
    ]
    types = [Vector{Float64},
             SizedVector{3, Float64},
             MVector{3, Float64},
             Vector{Float32},
             SizedVector{3, Float32},
             MVector{3, Float32},
             Vector{Double64},
             MVector{3, Double64},
             SizedVector{3, Double64}]
    for M in manifolds
        for T in types
            @testset "$M Type $T" begin
                pts = [convert(T, [1.0, 0.0, 0.0]),
                       convert(T, [0.0, 1.0, 0.0]),
                       convert(T, [0.0, 0.0, 1.0])]
                test_manifold(M,
                              pts,
                              test_reverse_diff = isa(T, Vector),
                              test_project_tangent = true,
                              test_musical_isomorphisms = true,
                              test_vector_transport = true,
                              test_mutating_rand = isa(T, Vector),
                              point_distributions = [Manifolds.projected_distribution(M, Distributions.MvNormal(zero(pts[1]), 1.0))],
                              tvector_distributions = [Manifolds.normal_tvector_distribution(M, pts[1], 1.0)])
            end
        end
    end
end
