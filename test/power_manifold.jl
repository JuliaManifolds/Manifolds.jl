include("utils.jl")

using HybridArrays

@testset "Power manifold" begin
    Ms = Sphere(2)
    Ms1 = PowerManifold(Ms, (5,))
    Ms2 = PowerManifold(Ms, (5,7))

    types1 = [Array{Float64,2},
              HybridArray{Tuple{3,StaticArrays.Dynamic()}, Float64, 2}]

    #retraction_methods = [Manifolds.ProductRetraction(Manifolds.ExponentialRetraction(), Manifolds.ExponentialRetraction())]
    #inverse_retraction_methods = [Manifolds.InverseProductRetraction(Manifolds.LogarithmicInverseRetraction(), Manifolds.LogarithmicInverseRetraction())]

    sphere_dist = Manifolds.uniform_distribution(Ms, @SVector [1.0, 0.0, 0.0])
    power_dist = Manifolds.PowerPointDistribution(Ms1, sphere_dist, randn(Float64, 3, 5))
    for T in types1
        @testset "Type $T" begin
            pts = [convert(T, rand(power_dist)) for _ in 1:3]
            test_manifold(Ms1,
                          pts;
                          test_reverse_diff = true,
                          test_musical_isomorphisms = false,
                          retraction_methods = [],
                          inverse_retraction_methods = [],
                          point_distributions = [power_dist],
                          tvector_distributions = [])
        end
    end

end
