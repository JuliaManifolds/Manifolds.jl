s = joinpath(@__DIR__, "..", "ManifoldsTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))
using ManifoldsTestSuite

using LinearAlgebra, Manifolds, Test


@testset "The Spheres" begin

    M = Sphere(2)
    p = [1.0, 0.0, 0.0]
    q = [1 / sqrt(2), 1 / sqrt(2), 0.0]
    X = [0.0, π / 4, 0.0]
    Y = [0.0, 0.0, π / 4]

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => Manifolds.Test.all_functions(),
            :InverseRetractionMethods => [LogarithmicInverseRetraction(), ProjectionInverseRetraction()],
            :Points => [p, q], :Vectors => [X, Y],
            :InvalidPoints => [2 * p],
            :InvalidVectors => [p],
            :RetractionMethods => [ExponentialRetraction(), ProjectionRetraction()],
            :VectorTransportMethods => [ParallelTransport(), SchildsLadderTransport(), PoleLadderTransport()]
        ),
        # Expectations
        Dict(
            :atol => 1.0e-12,
            default_inverse_retraction_method => LogarithmicInverseRetraction(),
            default_retraction_method => StabilizedRetraction(),
            default_vector_transport_method => ParallelTransport(),
            distance => π / 4,
            exp => q,
            injectivity_radius => π,
            (injectivity_radius, ProjectionRetraction()) => π / 2,
            log => X, norm => π / 4,
            parallel_transport_to => parallel_transport_to(M, p, X, q),
            parallel_transport_direction => parallel_transport_to(M, p, X, q),
            manifold_dimension => 2,
            :IsPointErrors => [DomainError]
        ),
    )

    @testset "Specific Tests and boundary cases" begin
        @testset "small and large distance tests" begin
            p = [-0.18337624444127734, 0.8345313166281056, 0.5195484910396462]
            q = [-0.18337624444127681, 0.8345313166281058, 0.5195484910396464]
            @test isapprox(distance(M, p, q), 5.828670879282073e-16)
            @test isapprox(distance(M, p, -q), 3.1415926535897927; atol = eps())
        end

        @testset "DefaultOrthonormalBasis edge cases" begin
            B = DefaultOrthonormalBasis(ℝ)
            n = manifold_dimension(M)
            p = [1; zeros(n)]
            for i in 1:n  # p'x ± 1
                vcoord = [j == i for j in 1:n]
                v = [0; vcoord]
                @test get_coordinates(M, p, v, B) ≈ vcoord
                @test get_vector(M, p, vcoord, B) ≈ v
                @test get_coordinates(M, -p, v, B) ≈ vcoord
                @test get_vector(M, -p, vcoord, B) ≈ v
            end
            p = [0; 1; zeros(n - 1)] # p'x = 0
            for i in 1:n
                vcoord = [j == i for j in 1:n]
                v = get_vector(M, p, vcoord, B)
                @test is_vector(M, p, v)
                @test get_coordinates(M, p, v, B) ≈ vcoord
            end
        end

        @testset "log edge case" begin
            n = manifold_dimension(M)
            x = normalize(randn(n + 1))
            v = log(M, x, -x)
            @test norm(v) ≈ π
            @test isapprox(dot(x, v), 0; atol = 1.0e-12)
            vexp = normalize(project(M, x, [1, zeros(n)...]))
            @test v ≈ π * vexp

            x = [1, zeros(n)...]
            v = log(M, x, -x)
            @test norm(v) ≈ π
            @test isapprox(dot(x, v), 0; atol = 1.0e-12)
            vexp = normalize(project(M, x, [0, 1, zeros(n - 1)...]))
            @test v ≈ π * vexp
        end

        #= Rework the following into the test suite cases
          Once all corresponding functions are in the test suite.

            @testset "Complex Sphere" begin
                M = Sphere(2, ℂ)
                @test repr(M) == "Sphere(2, ℂ)"
                @test typeof(get_embedding(M)) === Euclidean{ℂ, TypeParameter{Tuple{3}}}
                @test representation_size(M) == (3,)
                p = [1.0, 1.0im, 1.0]
                q = project(M, p)
                @test is_point(M, q)
                Y = [2.0, 1.0im, 20.0]
                X = project(M, q, Y)
                @test is_vector(M, q, X, true; atol = 10^(-14))
                Random.seed!(42)
                r = rand(M)
                @test is_point(M, r)
                @test norm(imag.(r)) != 0
            end

            @testset "Quaternion Sphere" begin
                M = Sphere(2, ℍ)
                @test repr(M) == "Sphere(2, ℍ)"
                @test typeof(get_embedding(M)) === Euclidean{ℍ, TypeParameter{Tuple{3}}}
                @test representation_size(M) == (3,)
                p = [Quaternion(1.0), Quaternion(0, 1.0, 0, 0), Quaternion(0.0, 0.0, -1.0, 0.0)]
                q = project(M, p)
                @test is_point(M, q)
                Y = [Quaternion(2.0), Quaternion(0, 1.0, 0, 0), Quaternion(0.0, 0.0, 20.0, 0.0)]
                X = project(M, q, Y)
                @test is_vector(M, q, X, true; atol = 10^(-14))
            end

            @testset "Array Sphere" begin
                M = ArraySphere(2, 2; field = ℝ)
                @test repr(M) == "ArraySphere(2, 2; field=ℝ)"
                @test typeof(get_embedding(M)) === Euclidean{ℝ, TypeParameter{Tuple{2, 2}}}
                @test representation_size(M) == (2, 2)
                p = ones(2, 2)
                q = project(M, p)
                @test is_point(M, q)
                Y = [1.0 0.0; 0.0 1.1]
                X = project(M, q, Y)
                @test is_vector(M, q, X)
                M = ArraySphere(2, 2; field = ℂ)

                @test repr(M) == "ArraySphere(2, 2; field=ℂ)"
                @test typeof(get_embedding(M)) === Euclidean{ℂ, TypeParameter{Tuple{2, 2}}}
                @test representation_size(M) == (2, 2)
            end
        =#

        #= Add the following to the test suite as functions
            @testset "Weingarten" begin
                M = Sphere(2)
                p = [1.0, 0.0, 0.0]
                X = [0.0, 2.0, 0.0]
                V = [0.3, 0.0, 0.0]
                Y = -X * (p'V)
                @test Weingarten(M, p, X, V) == Y
            end

            @testset "Volume density" begin
                M = Sphere(2)
                p = [1.0, 0.0, 0.0]
                @test manifold_volume(Sphere(0)) ≈ 2
                @test manifold_volume(Sphere(1)) ≈ 2 * π
                @test manifold_volume(Sphere(2)) ≈ 4 * π
                @test manifold_volume(Sphere(3)) ≈ 2 * π * π
                @test volume_density(M, p, [0.0, 0.5, 0.5]) ≈ 0.9187253698655684
            end

            @testset "sectional curvature" begin
                M = Sphere(2; parameter = :field)
                K = Manifolds.sectional_curvature_matrix(
                    M,
                    [1.0, 0.0, 0.0],
                    DefaultOrthonormalBasis(),
                )
                @test isapprox(K, [0 1; 1 0])
                @test isapprox(
                    Manifolds.estimated_sectional_curvature_matrix(
                        M,
                        [1.0, 0.0, 0.0],
                        DefaultOrthonormalBasis(),
                    ),
                    [0.0 1.0; 1.0 0.0],
                    atol = 0.15,
                )
            end
        =#
    end
end
