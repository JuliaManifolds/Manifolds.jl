include("utils.jl")

using HybridArrays

@testset "Power manifold" begin
    Ms = Sphere(2)
    Ms1 = PowerManifold(Ms, (5,))
    Ms2 = PowerManifold(Ms, (5,7))

    types1 = [Array{Float64,2},
              HybridArray{Tuple{3,StaticArrays.Dynamic()}, Float64, 2}]

    retraction_methods = [Manifolds.PowerRetraction(Manifolds.ExponentialRetraction())]
    inverse_retraction_methods = [Manifolds.InversePowerRetraction(Manifolds.LogarithmicInverseRetraction())]

    sphere_dist = Manifolds.uniform_distribution(Ms, @SVector [1.0, 0.0, 0.0])
    power_pt_dist = Manifolds.PowerPointDistribution(Ms1, sphere_dist, randn(Float64, 3, 5))
    sphere_tv_dist = Manifolds.normal_tvector_distribution(Ms, (@MVector [1.0, 0.0, 0.0]), 1.0)
    power_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Ms1), rand(power_pt_dist), sphere_tv_dist)

    for T in types1
        @testset "Type $T" begin
            pts = [convert(T, rand(power_pt_dist)) for _ in 1:3]
            test_manifold(Ms1,
                          pts;
                          test_reverse_diff = true,
                          test_musical_isomorphisms = false,
                          retraction_methods = retraction_methods,
                          inverse_retraction_methods = inverse_retraction_methods,
                          point_distributions = [power_pt_dist],
                          tvector_distributions = [power_tv_dist],
                          rand_tvector_atol_multiplier = 2.0)
        end
    end

end
