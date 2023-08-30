include("../utils.jl")

@testset "Positive Numbers" begin
    M = PositiveNumbers()
    @testset "Positive Numbers Basics" begin
        @test repr(M) == "PositiveNumbers()"
        @test repr(PositiveVectors(2)) == "PositiveVectors(2)"
        @test repr(PositiveMatrices(2, 3)) == "PositiveMatrices(2, 3)"
        @test repr(PositiveArrays(2, 3, 4)) == "PositiveArrays(2, 3, 4)"
        @test representation_size(M) == ()
        @test manifold_dimension(M) == 1
        @test !is_flat(M)
        @test !is_point(M, -1.0)
        @test_throws DomainError is_point(M, -1.0, true)
        @test is_vector(M, 1.0, 0.0)
        @test vector_transport_to(M, 1.0, 3.0, 2.0, ParallelTransport()) == 6.0
        @test retract(M, 1.0, 1.0) == exp(M, 1.0, 1.0)
        @test isinf(injectivity_radius(M))
        @test isinf(injectivity_radius(M, -2.0))
        @test isinf(injectivity_radius(M, -2.0, ExponentialRetraction()))
        @test isinf(injectivity_radius(M, ExponentialRetraction()))
        @test project(M, 1.5, 1.0) == 1.0
        @test embed(M, 1.0) == 1.0
        @test zero_vector(M, 1.0) == 0.0
        X = similar([1.0])
        zero_vector!(M, X, 1.0)
        @test X == [0.0]
        @test get_coordinates(M, 2.0, 1.0, DefaultOrthonormalBasis()) == 0.5
        @test get_vector(M, 2.0, 0.5, DefaultOrthonormalBasis()) == 1.0

        @test change_metric(M, EuclideanMetric(), 2, 3) == 3 * 2
        @test change_representer(M, EuclideanMetric(), 2, 3) == 3 * 2^2
        N = PositiveVectors(2)
        @test change_metric(M, EuclideanMetric(), [1, 2], [3, 4]) == [3, 4 * 2]
        @test change_representer(M, EuclideanMetric(), [1, 2], [3, 4]) == [3, 4 * 2^2]
        @test get_coordinates(N, [2.0, 4.0], [1.0, 10.0], DefaultOrthonormalBasis()) ==
              [0.5, 2.5]
        @test get_vector(N, [2.0, 4.0], [0.5, 2.5], DefaultOrthonormalBasis()) ==
              [1.0, 10.0]
        tmp = zeros(2)
        get_coordinates!(N, tmp, [2.0, 4.0], [1.0, 10.0], DefaultOrthonormalBasis())
        @test tmp == [0.5, 2.5]
        get_vector!(N, tmp, [2.0, 4.0], [0.5, 2.5], DefaultOrthonormalBasis())
        @test tmp == [1.0, 10.0]
    end
    types = [Float64]
    TEST_FLOAT32 && push!(types, Float32)
    for T in types
        @testset "Type $T" begin
            pts = convert.(Ref(T), [1.0, 4.0, 2.0])
            test_manifold(
                M,
                pts,
                test_vector_spaces=false,
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                test_vee_hat=false,
                is_mutating=false,
                test_rand_point=true,
                test_rand_tvector=true,
            )
        end
    end
    @testset "Power of Positive Numbers" begin
        M2 = PositiveVectors(2)
        for T in types
            pts2 = [convert.(Ref(T), v) for v in [[1.0, 1.1], [3.0, 3.3], [2.0, 2.2]]]
            test_manifold(
                M2,
                pts2,
                test_vector_spaces=false,
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                test_vee_hat=false,
            )
        end
    end
    @testset "Weingarten & Hessian" begin
        M = PositiveNumbers()
        p = 1.0
        G = 2.0
        H = 3.0
        X = 4.0
        rH = riemannian_Hessian(M, p, G, H, X)
        @test rH == p * H * p + X * G * p
    end

    @testset "Manifold volume" begin
        M5 = PositiveVectors(3)
        @test isinf(manifold_volume(M))
        @test isinf(manifold_volume(M5))
        @test volume_density(M, 0.5, 2.0) ≈ exp(4.0)
        @test volume_density(M5, [0.5, 1.0, 2.0], [1.0, -1.0, 2.0]) ≈ exp(2.0)
    end
end
