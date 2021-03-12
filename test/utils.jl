const TEST_FLOAT32 = false
const TEST_DOUBLE64 = false
const TEST_STATIC_SIZED = false

using Manifolds
using ManifoldsBase
using ManifoldsBase: number_of_coordinates

using LinearAlgebra
using Distributions
using DoubleFloats
using ForwardDiff
using Quaternions
using Random
using ReverseDiff
using StaticArrays
using Statistics
using StatsBase
using Test
using LightGraphs
using SimpleWeightedGraphs

using Manifolds.ManifoldTests

function include_test(path)
    @info "Testing $path"
    @time include(path)  # show basic timing, (this will print a newline at end)
end

function our_base_ambiguities()
    ambigs = Test.detect_ambiguities(Base)
    modules_we_care_about =
        [Base, LinearAlgebra, Manifolds, ManifoldsBase, StaticArrays, Statistics, StatsBase]
    our_ambigs = filter(ambigs) do (m1, m2)
        we_care = m1.module in modules_we_care_about && m2.module in modules_we_care_about
        return we_care && (m1.module === Manifolds || m2.module === Manifolds)
    end
    return our_ambigs
end

@testset "utils" begin
    @testset "log_safe!" begin
        n = 8
        Q = qr(randn(n, n)).Q
        A1 = Q * Diagonal(rand(n)) * Q'
        @test exp(Manifolds.log_safe!(similar(A1), A1)) ≈ A1 atol=1e-8
        A1_fail = Q * Diagonal([-1; rand(n - 1)]) * Q'
        @test_throws DomainError Manifolds.log_safe!(similar(A1_fail), A1_fail)

        T = triu!(randn(n, n))
        T[diagind(T)] .= rand.()
        @test exp(Manifolds.log_safe!(similar(T), T)) ≈ T atol=1e-8
        T_fail = copy(T)
        T_fail[1] = -1
        @test_throws DomainError Manifolds.log_safe!(similar(T_fail), T_fail)

        A2 = Q * T * Q'
        @test exp(Manifolds.log_safe!(similar(A2), A2)) ≈ A2 atol=1e-8
        A2_fail = Q * T_fail * Q'
        @test_throws DomainError Manifolds.log_safe!(similar(A2_fail), A2_fail)

        A3 = exp(randn(n, n))
        @test exp(Manifolds.log_safe!(similar(A3), A3)) ≈ A3 atol=1e-8

        A3_fail = Float64[1 2; 3 1]
        @test_throws DomainError Manifolds.log_safe!(similar(A3_fail), A3_fail)

        A4 = randn(ComplexF64, n, n)
        @test exp(Manifolds.log_safe!(similar(A4), A4)) ≈ A4 atol=1e-8
    end
end
