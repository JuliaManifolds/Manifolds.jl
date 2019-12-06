include("utils.jl")

@testset "Euclidean" begin
    E = Manifolds.Euclidean(3)
    EM = Manifolds.MetricManifold(E,Manifolds.EuclideanMetric())
    @test is_default_metric(EM) == Val{true}()
    @test is_default_metric(E,Manifolds.EuclideanMetric()) == Val{true}()
    x = zeros(3)
    @test det_local_metric(EM,x) == one(eltype(x))
    @test log_local_metric_density(EM,x) == zero(eltype(x))
    @test project_point!(E,x) == x

    manifolds = [ E, EM ]
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
