using LinearAlgebra, Manifolds, Random, Test

@testset "Positive Numbers" begin
    M = PositiveNumbers()

    @testset "Constructors and basics" begin
        @test repr(M) == "PositiveNumbers()"
        @test repr(PositiveVectors(2)) == "PositiveVectors(2)"
        @test repr(PositiveMatrices(2, 3)) == "PositiveMatrices(2, 3)"
        @test repr(PositiveArrays(2, 3, 4)) == "PositiveArrays(2, 3, 4)"
        @test representation_size(M) == ()
        @test manifold_dimension(M) == 1
        @test !is_flat(M)
        @test !is_point(M, -1.0)
        @test_throws DomainError is_point(M, -1.0; error = :error)
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

        xi = flat(M, 2.0, 1.0)
        @test xi(1.0) ≈ norm(M, 2.0, 1.0)^2
        @test sharp(M, 2.0, xi) == 1.0
    end

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                default_inverse_retraction_method,
                default_retraction_method,
                default_vector_transport_method,
                distance,
                exp,
                geodesic,
                get_coordinates,
                get_vector,
                injectivity_radius,
                inner,
                inverse_retract,
                is_flat,
                is_point,
                is_vector,
                log,
                manifold_dimension,
                manifold_volume,
                mid_point,
                norm,
                parallel_transport_direction,
                parallel_transport_to,
                rand,
                repr,
                representation_size,
                retract,
                shortest_geodesic,
                vector_transport_direction,
                vector_transport_to,
                volume_density,
                zero_vector,
            ],
            :Bases => [DefaultOrthonormalBasis()],
            :Coordinates => [1.0],
            :InvalidPoints => [-1.0],
            :InverseRetractionMethods => [LogarithmicInverseRetraction()],
            :Mutating => false,
            :Name => "Manifolds.Test suite for PositiveNumbers()",
            :Points => [1.0, 4.0, 2.0],
            :RetractionMethods => [ExponentialRetraction()],
            :SecondVector => 2.0,
            :VectorTransportMethods => [ParallelTransport()],
            :Vectors => [1.0, 2.0],
        ),
        Dict(
            default_inverse_retraction_method => LogarithmicInverseRetraction(),
            default_retraction_method => ExponentialRetraction(),
            default_vector_transport_method => ParallelTransport(),
            distance => log(4.0),
            injectivity_radius => Inf,
            (injectivity_radius, ExponentialRetraction()) => Inf,
            is_flat => false,
            manifold_dimension => 1,
            manifold_volume => Inf,
            repr => "PositiveNumbers()",
            representation_size => (),
            volume_density => exp(1.0),
            :IsPointErrors => [DomainError],
        ),
    )

    @testset "Positive vectors" begin
        M2 = PositiveVectors(2)
        p1 = [1.0, 1.1]
        p2 = [3.0, 3.3]
        p3 = [2.0, 2.2]
        X1 = [1.0, 1.1]
        X2 = [0.5, 2.2]

        Manifolds.Test.test_manifold(
            M2,
            Dict(
                :Functions => [
                    default_inverse_retraction_method,
                    default_retraction_method,
                    default_vector_transport_method,
                    distance,
                    exp,
                    flat,
                    geodesic,
                    get_coordinates,
                    get_vector,
                    injectivity_radius,
                    inner,
                    inverse_retract,
                    is_flat,
                    is_point,
                    is_vector,
                    log,
                    manifold_dimension,
                    manifold_volume,
                    mid_point,
                    norm,
                    parallel_transport_direction,
                    parallel_transport_to,
                    rand,
                    repr,
                    representation_size,
                    retract,
                    sharp,
                    shortest_geodesic,
                    vector_transport_direction,
                    vector_transport_to,
                    volume_density,
                    zero_vector,
                ],
                :Bases => [DefaultOrthonormalBasis()],
                :Coordinates => [[1.0, 1.0]],
                :Covectors => [flat(M2, p1, X1)],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :Name => "Manifolds.Test suite for PositiveVectors(2)",
                :Points => [p1, p2, p3],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => X2,
                :VectorTransportMethods => [ParallelTransport()],
                :Vectors => [X1, X2],
            ),
            Dict(
                default_inverse_retraction_method => LogarithmicInverseRetraction(),
                default_retraction_method => ExponentialRetraction(),
                default_vector_transport_method => ParallelTransport(),
                injectivity_radius => Inf,
                (injectivity_radius, ExponentialRetraction()) => Inf,
                is_flat => false,
                manifold_dimension => 2,
                manifold_volume => Inf,
                repr => "PositiveVectors(2)",
                representation_size => (2,),
                volume_density => exp(2.0),
                :atols => Dict(shortest_geodesic => 1.0e-12),
            ),
        )
    end

    @testset "Weingarten and Hessian" begin
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

    @testset "In-place random values" begin
        p = fill(NaN)
        X = fill(NaN)
        rand!(M, p)
        @test is_point(M, p)
        rand!(M, X; vector_at = p)
        @test is_vector(M, p, X)

        rng = MersenneTwister(123)
        rand!(rng, M, p)
        @test is_point(M, p)
        rand!(rng, M, X; vector_at = p)
        @test is_vector(M, p, X)
    end

    @testset "Field parameter" begin
        M1 = PositiveVectors(3; parameter = :field)
        @test repr(M1) == "PositiveVectors(3; parameter=:field)"
        M2 = PositiveMatrices(3, 4; parameter = :field)
        @test repr(M2) == "PositiveMatrices(3, 4; parameter=:field)"
        M3 = PositiveArrays(3, 4, 5; parameter = :field)
        @test repr(M3) == "PositiveArrays(3, 4, 5; parameter=:field)"
    end
end
