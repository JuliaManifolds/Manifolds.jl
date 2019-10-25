include("utils.jl")

using HybridArrays

@testset "Power manifold" begin
    Ms = Sphere(2)
    Ms1 = PowerManifold(Ms, (5,))
    Ms2 = PowerManifold(Ms, (5,7))

    types1 = [Array{Float64,2},
              HybridArray{Tuple{3,StaticArrays.Dynamic()}, Float64, 2}]
    types2 = [Array{Float64,3},
              HybridArray{Tuple{3,StaticArrays.Dynamic(),StaticArrays.Dynamic()}, Float64, 3}]
    retraction_methods = [Manifolds.PowerRetraction(Manifolds.ExponentialRetraction())]
    inverse_retraction_methods = [Manifolds.InversePowerRetraction(Manifolds.LogarithmicInverseRetraction())]

    sphere_dist = Manifolds.uniform_distribution(Ms, @SVector [1.0, 0.0, 0.0])
    power1_pt_dist = Manifolds.PowerPointDistribution(Ms1, sphere_dist, randn(Float64, 3, 5))
    power2_pt_dist = Manifolds.PowerPointDistribution(Ms2, sphere_dist, randn(Float64, 3, 5, 7))
    sphere_tv_dist = Manifolds.normal_tvector_distribution(Ms, (@MVector [1.0, 0.0, 0.0]), 1.0)
    power1_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Ms1), rand(power1_pt_dist), sphere_tv_dist)
    power2_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Ms2), rand(power2_pt_dist), sphere_tv_dist)

    for T in types1
        @testset "Type $(string(T)[1:10])..." begin
            pts1 = [convert(T, rand(power1_pt_dist)) for _ in 1:3]
            test_manifold(Ms1,
                          pts1;
                          test_reverse_diff = true,
                          test_musical_isomorphisms = false,
                          retraction_methods = retraction_methods,
                          inverse_retraction_methods = inverse_retraction_methods,
                          point_distributions = [power1_pt_dist],
                          tvector_distributions = [power1_tv_dist],
                          rand_tvector_atol_multiplier = 5.0)
        end
    end
    for T in types2
        @testset "Type $(string(T)[1:10])..." begin
            pts2 = [convert(T, rand(power2_pt_dist)) for _ in 1:3]
            test_manifold(Ms2,
                          pts2;
                          test_reverse_diff = true,
                          test_musical_isomorphisms = false,
                          retraction_methods = retraction_methods,
                          inverse_retraction_methods = inverse_retraction_methods,
                          point_distributions = [power2_pt_dist],
                          tvector_distributions = [power2_tv_dist],
                          rand_tvector_atol_multiplier = 5.0)
        end
    end

end
