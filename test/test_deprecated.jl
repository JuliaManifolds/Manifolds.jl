using Manifolds, ManifoldsBase, Random, Test

using StatsBase: AbstractWeights, pweights
using Random: GLOBAL_RNG, seed!

@testset "Deprecation tests" begin
    @testset "Depreacte extrinsic_method= keyword" begin
        rng = MersenneTwister(47)
        S = Sphere(2)
        x = [normalize(randn(rng, 3)) for _ in 1:10]
        w = pweights([rand(rng) for _ in 1:length(x)])
        mg1 = mean(S, x, w, ExtrinsicEstimation(EfficientEstimator()))
        # Statistics 414-418, depcreatce former extrinsic_method keyword
        mg2 = mean(
            S,
            x,
            w,
            ExtrinsicEstimation(EfficientEstimator());
            extrinsic_method=EfficientEstimator(),
        )
        @test isapprox(S, mg1, mg2)
        mg3 = median(S, x, w, ExtrinsicEstimation(CyclicProximalPointEstimation()))
        # Statistics 692-696, depcreatce former extrinsic_method keyword
        mg4 = median(
            S,
            x,
            w,
            ExtrinsicEstimation(CyclicProximalPointEstimation());
            extrinsic_method=CyclicProximalPointEstimation(),
        )
    end
end
