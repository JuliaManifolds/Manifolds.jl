include("utils.jl")
using StatsBase: pweights
using Random: GLOBAL_RNG, seed!
import ManifoldsBase: manifold_dimension, exp!, log!, distance, zero_tangent_vector!

struct TestStatsSphere{N} <: Manifold end
TestStatsSphere(N) = TestStatsSphere{N}()
manifold_dimension(M::TestStatsSphere{N}) where {N} = manifold_dimension(Sphere(N))
exp!(M::TestStatsSphere{N}, args...; kwargs...) where {N} = exp!(Sphere(N), args...; kwargs...)
log!(M::TestStatsSphere{N}, args...; kwargs...) where {N} = log!(Sphere(N), args...; kwargs...)
distance(M::TestStatsSphere{N}, args...; kwargs...) where {N} = distance(Sphere(N), args...; kwargs...)
zero_tangent_vector!(M::TestStatsSphere{N}, args...; kwargs...) where {N} = zero_tangent_vector!(Sphere(N), args...; kwargs...)

function test_mean(M, x, yexp = nothing; kwargs...)
    @testset "mean unweighted" begin
        y = mean(M, x; kwargs...)
        @test is_manifold_point(M, y; atol=10^-9)
        if yexp !== nothing
            @test isapprox(M, y, yexp; atol=10^-7)
        end
        y2, _ = mean_and_var(M, x; kwargs...)
        @test isapprox(M, y2, y; atol=10^-7)
        y3, _ = mean_and_std(M, x; kwargs...)
        @test isapprox(M, y3, y; atol=10^-7)
    end

    @testset "mean weighted" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))
        y = mean(M, x; kwargs...)
        for w in (w1, w2, w3)
            @test is_manifold_point(M, mean(M, x, w; kwargs...); atol=10^-9)
            @test isapprox(M, mean(M, x, w; kwargs...), y)

            @test isapprox(M, mean_and_var(M, x, w; kwargs...)[1], y; atol=10^-7)
            @test isapprox(M, mean_and_std(M, x, w; kwargs...)[1], y; atol=10^-7)
        end
        @test_throws DimensionMismatch mean(M, x, pweights(ones(n + 1)); kwargs...)
    end
end

function test_median(M, x, yexp = nothing; rng=GLOBAL_RNG, kwargs...)
    @testset "median unweighted" begin
        y = median(M, x; kwargs...)
        @test is_manifold_point(M, y; atol=10^-9)
        if yexp !== nothing
            @test isapprox(M, y, yexp; atol=10^-5)
        end

        y2 = median(M, x; shuffle_rng=rng, kwargs...)
        @test is_manifold_point(M, y2; atol = 10^-9)
        @test isapprox(M, y, y2; atol = 10^-5)
    end

    @testset "median weighted" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))
        y = median(M, x; kwargs...)
        for w in (w1, w2, w3)
            @test is_manifold_point(M, median(M, x, w; kwargs...); atol = 10^-9)
            @test isapprox(M, median(M, x, w; kwargs...), y; atol = 10^-5)
            @test is_manifold_point(M, median(M, x, w; shuffle_rng=rng, kwargs...); atol = 10^-9)
            @test isapprox(M, median(M, x, w; shuffle_rng = rng, kwargs...), y; atol = 10^-5)
        end
        @test_throws DimensionMismatch median(M, x, pweights(ones(n + 1)); kwargs...)
    end
end

function test_var(M, x, vexp = nothing; kwargs...)
    n = length(x)
    @testset "var unweighted" begin
        v = var(M, x; kwargs...)
        if vexp !== nothing
            @test v ≈ vexp
        end
        @test v == var(M, x; corrected = true, kwargs...)
        _, v2 = mean_and_var(M, x; kwargs...)
        @test v2 == v
        m = mean(M, x; kwargs...)
        @test var(M, x, m; kwargs...) == var(M, x; kwargs...)
        @test var(M, x; corrected = false, kwargs...) == var(M, x, m; corrected = false, kwargs...)
        @test var(M, x; corrected = false, kwargs...) ≈ var(M, x; kwargs...) * (n - 1) / n
        @test var(M, x, m; corrected = false, kwargs...) ≈ var(M, x, m; kwargs...) * (n - 1) / n
    end

    @testset "var weighted" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))

        v = var(M, x; corrected = false, kwargs...)
        for w in (w1, w2, w3)
            @test var(M, x, w; kwargs...) ≈ v
            @test var(M, x, w; corrected = false, kwargs...) ≈ v
            @test mean_and_var(M, x, w; kwargs...)[2] == var(M, x, w; kwargs...)
            m = mean(M, x, w; kwargs...)
            @test var(M, x, w, m; kwargs...) == var(M, x, w; kwargs...)
            @test var(M, x, w; corrected = true, kwargs...) == var(M, x, w, m; corrected = true, kwargs...)
            @test var(M, x, w; corrected = true, kwargs...) ≈ var(M, x, w; kwargs...) * n / (n - 1)
            @test var(M, x, w, m; corrected = true, kwargs...) ≈ var(M, x, w, m; kwargs...) * n / (n - 1)
        end
        @test_throws DimensionMismatch var(M, x, pweights(ones(n + 1)); kwargs...)
    end
end

function test_std(M, x, sexp = nothing; kwargs...)
    n = length(x)
    @testset "std unweighted" begin
        s = std(M, x; kwargs...)
        if sexp !== nothing
            @test s ≈ sexp
        end
        @test s == std(M, x; corrected = true, kwargs...)
        @test s ≈ √var(M, x; corrected = true, kwargs...)
        _, s2 = mean_and_std(M, x; kwargs...)
        @test s2 == s
        m = mean(M, x; kwargs...)
        @test std(M, x, m; kwargs...) == std(M, x; kwargs...)
        @test std(M, x; corrected = false, kwargs...) == std(M, x, m; corrected = false, kwargs...)
        @test std(M, x; corrected = false, kwargs...) ≈ std(M, x; kwargs...) * sqrt((n - 1) / n)
        @test std(M, x, m; corrected = false, kwargs...) ≈ std(M, x, m; kwargs...) * sqrt((n - 1) / n)
    end

    @testset "std weighted" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))

        s = std(M, x; corrected = false, kwargs...)
        for w in (w1, w2, w3)
            @test std(M, x, w; kwargs...) ≈ s
            @test std(M, x, w; corrected = false, kwargs...) ≈ s
            @test mean_and_std(M, x, w; kwargs...)[2] == std(M, x, w; kwargs...)
            m = mean(M, x, w; kwargs...)
            @test std(M, x, w, m; kwargs...) == std(M, x, w; kwargs...)
            @test std(M, x, w; corrected = true, kwargs...) == std(M, x, w, m; corrected = true, kwargs...)
            @test std(M, x, w; corrected = true, kwargs...) ≈ std(M, x, w; kwargs...) * sqrt(n / (n - 1))
            @test std(M, x, w, m; corrected = true, kwargs...) ≈ std(M, x, w, m; kwargs...) * sqrt(n / (n - 1))
        end
        @test_throws DimensionMismatch std(M, x, pweights(ones(n + 1)); kwargs...)
    end
end

@testset "Statistics" begin
    M = TestStatsSphere(2)

    @testset "consistency" begin
        p = [0.,0.,1.]
        n=3
        x = [ exp(M,p,2/n*[cos(α), sin(α), 0.]) for α = range(0,2*π - 2*π/n, length=n) ]
        test_mean(M, x)
        test_median(M, x; rng = MersenneTwister(1212), atol = 10^-12)
        test_var(M, x)
        test_std(M, x)
    end

    @testset "zero variance" begin
        x = fill([0.,0.,1.], 5)
        test_mean(M, x, [0.,0.,1.])
        test_median(M, x, [0.,0.,1.]; rng = MersenneTwister(1212), atol = 10^-12)
        test_var(M, x, 0.0)
        test_std(M, x, 0.0)
    end

    @testset "two points" begin
        x = [ [1., 0., 0.], [0., 1., 0.] ]
        θ = π / 4
        @test isapprox(M, mean(M, x), geodesic(M, x[1], x[2], θ))
        test_mean(M, x, [1.0, 1.0, 0.0] / √2)
        test_median(M, x, [1.0, 1.0, 0.0] / √2; rng = MersenneTwister(1212), atol = 10^-12)
        test_var(M, x, θ^2 * 2)
        test_std(M, x, θ * √2)
    end

    @testset "Euclidean fallbacks" begin
        x = [1., 2., 3., 4.,]
        w = pweights(  ones(length(x)) / length(x)  )
        @test mean(Euclidean(1),x) == mean(x)
        @test mean(Euclidean(1),x,w) == mean(x,w)
        @test median(Euclidean(1),x) == median(x)
        @test median(Euclidean(1),x,w) == median(x,w)
    end
end
