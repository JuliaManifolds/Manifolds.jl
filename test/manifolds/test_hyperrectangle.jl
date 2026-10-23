using LinearAlgebra, Manifolds, ManifoldsBase, Random, StatsBase, Test

@testset "The Hyperrectangle" begin
    @test_throws ArgumentError Hyperrectangle([1.0, 2.0], [-1.0, 5.0])

    M = Hyperrectangle([-1.0, 2.0, -3.0], [1.0, 4.0, 9.0])
    @testset "Basics" begin
        @test M == Hyperrectangle([-1.0, 2.0, -3.0], [1.0, 4.0, 9.0])
        @test hash(M) == hash(Hyperrectangle([-1.0, 2.0, -3.0], [1.0, 4.0, 9.0]))
        @test M != Hyperrectangle([-1.0, 2.0, -3.0], [1.0, 4.0, 10.0])
        @test !isequal(Hyperrectangle([-0.0], [1.0]), Hyperrectangle([0.0], [1.0]))
        @test metric(M) == EuclideanMetric()
        @test_throws DomainError is_point(M, [10.0, 3.0, 0.0]; atol = 1.0e-5, error = :error)
        @test_throws DomainError is_vector(M, [1], [1.0, 1.0, 0.0]; error = :error)
    end

    functions = [
        copy, copyto!, default_inverse_retraction_method, default_retraction_method,
        default_vector_transport_method, distance, embed, exp, geodesic, get_basis,
        get_coordinates, get_embedding, get_vector, get_vectors, injectivity_radius, inner, inverse_retract,
        is_flat, is_point, is_vector, log, manifold_dimension, manifold_volume, mid_point, norm,
        parallel_transport_direction, parallel_transport_to, project, rand, repr,
        representation_size, retract, sectional_curvature, shortest_geodesic,
        vector_transport_direction, vector_transport_to, volume_density, zero_vector,
    ]
    common_expectations = Dict(
        :atol => 1.0e-12,
        default_inverse_retraction_method => LogarithmicInverseRetraction(),
        default_retraction_method => ProjectionRetraction(),
        default_vector_transport_method => ParallelTransport(),
        (get_vectors, DefaultOrthonormalBasis()) => :Orthonormal,
        injectivity_radius => 0.0,
        (injectivity_radius, ExponentialRetraction()) => 0.0,
        (injectivity_radius, ProjectionRetraction()) => 0.0,
        is_flat => true,
        sectional_curvature => 0.0,
        sectional_curvature_max => 0.0,
        sectional_curvature_min => 0.0,
        volume_density => 1.0,
    )

    @testset "Vector Hyperrectangle" begin
        # p1 is in the interior with distance 1 to the boundary, p3 is on the boundary
        p1, p2, p3 = [0.0, 3.0, 0.0], [0.0, 3.5, 1.0], [1.0, 2.0, 0.0]
        X = [0.2, -0.3, 0.5]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [DefaultOrthonormalBasis()],
                :Coordinates => [[1.0, 2.0, 3.0]],
                :EmbeddedPoints => [[4.0, -2.0, 3.0]],
                :EmbeddedVectors => [[2.0, -0.5, -10.0]],
                :Functions => functions,
                :InvalidPoints => Vector[[1.0im, 0.0, 0.0], [NaN, 3.0, 0.0], [10.0, 3.0, 0.0]], # Vector avoids promotion to complex
                :InvalidVectors => Vector[[1.0im, 0.0, 0.0], [1.0], [1.0, 0.0, 1.0im]],
                :InverseRetractionMethods => [LogarithmicInverseRetraction(), missing],
                :Points => [p1, p2, p3],
                :RetractionMethods => [ExponentialRetraction(), ProjectionRetraction()],
                :SecondVector => [0.0, 1.0, 0.0],
                :Vectors => [X, [1.0, 0.0, 0.0], [-1.0, 1.0, 1.0]],
                :VectorTransportMethods => [ParallelTransport()],
                :Seed => 42,
            ),
            merge(
                common_expectations,
                Dict(
                    distance => sqrt(1.25),
                    exp => [0.2, 2.7, 0.5],
                    get_embedding => Euclidean(3; parameter = :field),
                    get_coordinates => X,
                    (get_vector, [1.0, 2.0, 3.0], DefaultOrthonormalBasis()) => [1.0, 2.0, 3.0],
                    (injectivity_radius, p1) => 1.0,
                    (injectivity_radius, p1, ExponentialRetraction()) => 1.0,
                    (injectivity_radius, p1, ProjectionRetraction()) => 1.0,
                    inner => -0.3,
                    log => [0.0, 0.5, 1.0],
                    manifold_dimension => 3,
                    manifold_volume => 48.0,
                    mid_point => [0.0, 3.25, 0.5],
                    norm => sqrt(0.38),
                    parallel_transport_direction => X,
                    parallel_transport_to => X,
                    (project, :Point) => [1.0, 2.0, 3.0],
                    (project, :Vector) => [0.0, 0.0, -10.0],
                    repr => "Hyperrectangle([-1.0, 2.0, -3.0], [1.0, 4.0, 9.0])",
                    representation_size => (3,),
                    (retract, ProjectionRetraction()) => [0.2, 2.7, 0.5],
                    (vector_transport_to, ParallelTransport()) => X,
                    :IsPointErrors => [DomainError, DomainError, DomainError],
                    :IsVectorErrors => [DomainError, DomainError, DomainError],
                    :IsVectorBasepointError => DomainError,
                ),
            ),
        )
    end

    @testset "Array Hyperrectangle" begin
        MA = Hyperrectangle([-1.0 2.0 3.0; 2.0 5.0 10.0], [10.0 12.0 13.0; 12.0 15.0 20.0])
        # p1 is in the interior with distance 3 to the boundary, p2 and p3 are on the boundary
        p1 = [2.0 5.0 6.0; 6.0 8.0 14.0]
        p2 = [-1.0 5.0 3.0; 4.0 5.0 10.0]
        p3 = [-1.0 2.0 5.0; 6.0 6.0 15.0]
        X = [0.1 -0.2 0.3; -0.4 0.5 -0.6]
        Manifolds.Test.test_manifold(
            MA,
            Dict(
                :Bases => [DefaultOrthonormalBasis()],
                :Coordinates => [collect(1.0:6.0)],
                :EmbeddedPoints => [[-2.0 5.0 6.0; 6.0 8.0 25.0]],
                :EmbeddedVectors => [[1.0 1.0 1.0; 1.0 1.0 1.0]],
                :Functions => functions,
                :InvalidPoints => Any[[-2.0 5.0 6.0; 6.0 8.0 14.0], [2.0, 5.0, 6.0]],
                :InvalidVectors => [ones(2, 2)],
                :InverseRetractionMethods => [LogarithmicInverseRetraction(), missing],
                :Points => [p1, p2, p3],
                :RetractionMethods => [ExponentialRetraction(), ProjectionRetraction()],
                :SecondVector => [1.0 0.0 0.0; 0.0 0.0 0.0],
                :Vectors => [X, zeros(2, 3), ones(2, 3)],
                :VectorTransportMethods => [ParallelTransport()],
                :Seed => 42,
            ),
            merge(
                common_expectations,
                Dict(
                    distance => norm(p2 - p1),
                    exp => p1 + X,
                    get_embedding => Euclidean(2, 3; parameter = :field),
                    get_coordinates => vec(X),
                    (get_vector, collect(1.0:6.0), DefaultOrthonormalBasis()) => [1.0 3.0 5.0; 2.0 4.0 6.0],
                    (injectivity_radius, p1) => 3.0,
                    inner => 0.1,
                    log => p2 - p1,
                    manifold_dimension => 6,
                    manifold_volume => 1.1e6,
                    mid_point => (p1 + p2) / 2,
                    norm => norm(X),
                    parallel_transport_to => X,
                    (project, :Point) => [-1.0 5.0 6.0; 6.0 8.0 20.0],
                    (project, :Vector) => [1.0 1.0 1.0; 1.0 1.0 0.0],
                    repr => "Hyperrectangle([-1.0 2.0 3.0; 2.0 5.0 10.0], [10.0 12.0 13.0; 12.0 15.0 20.0])",
                    representation_size => (2, 3),
                    (retract, ProjectionRetraction()) => p1 + X,
                    :IsPointErrors => [DomainError, DomainError],
                    :IsVectorErrors => [DomainError],
                    :IsVectorBasepointError => DomainError,
                ),
            ),
        )
    end

    @testset "projections" begin
        @test project(M, [4.0, -2.0, 3.0]) ≈ [1.0, 2.0, 3.0]
        @test project(M, [1.0, 2.0, 3.0], [2.0, -0.5, -10.0]) ≈ [0.0, 0.0, -10.0]
        @test retract(M, [0.0, 3.0, 0.0], [10.0, 10.0, 10.0]) == [1.0, 4.0, 9.0]
    end

    @testset "injectivity_radius" begin
        @test injectivity_radius(M) == 0.0
        @test injectivity_radius(M, ExponentialRetraction()) == 0.0
        @test injectivity_radius(M, [0.0, 2.5, 1.0]) == 0.5
        @test injectivity_radius(M, [0.0, 2.5, 1.0], ExponentialRetraction()) == 0.5
        @test injectivity_radius(M, [0.0, 2.5, 1.0], ProjectionRetraction()) == 0.5
        @test injectivity_radius(M, [0.0, 2.0, 1.0]) == 1.0
        @test injectivity_radius(M, [0.5, 4.0, 1.0]) == 0.5

        M2 = Hyperrectangle([-1.0, 2.0, -3.0], [1.0, 2.0, 9.0])
        @test injectivity_radius(M2, [0.0, 2.0, 1.0]) == 1.0
    end

    @testset "Hyperrectangle(1)" begin
        M = Hyperrectangle([0.0], [5.0])
        @test distance(M, 2.0, 4.0) == 2.0
    end

    @testset "errors" begin
        M = Hyperrectangle([-1.0, 2.0, -3.0, 0.0], [1.0, 4.0, 9.0, 3.0])
        @test_throws DimensionMismatch distance(M, [1, 2, 3, 4], [1 2; 3 4])
    end

    @testset "Weingarten & Hessian" begin
        M = Hyperrectangle([-1.0, -1.0], [10.0, 10.0])
        p = [1.0, 2.0]
        G = [3.0, 4.0]
        H = [5.0, 6.0]
        X = [7.0, 8.0]
        rH = riemannian_Hessian(M, p, G, H, X)
        @test rH == H
    end
    pts = [[1.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.5, 1.0]]
    @testset "statistics" begin
        @test default_approximation_method(M, mean) === EfficientEstimator()
        @test mean(M, pts) ≈ [1 / 3, 17 / 6, 1 / 3]
        @test mean(M, pts, pweights(ones(3) / 3)) ≈ [1 / 3, 17 / 6, 1 / 3]
        @test_throws DimensionMismatch mean(M, pts, pweights(ones(4) / 4))
        @test var(M, pts) ≈ 1.25
    end

    @testset "Euclidean metric tests" begin
        @test riemann_tensor(
            M, pts[1], pts[2] - pts[1], pts[3] - pts[1], (pts[3] - pts[1]) / 2
        ) == [0.0, 0.0, 0.0]
        @test sectional_curvature(M, pts[2], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == 0.0
        @test sectional_curvature_max(M) == 0.0
        @test sectional_curvature_min(M) == 0.0
    end

    @testset "rand special cases" begin
        M = Hyperrectangle([-1.0, -1.0, -Inf, -Inf], [1.0, Inf, 1.0, Inf])
        p = rand(M)
        @test is_point(M, p)
    end

    @testset "has_components and r-norms" begin
        M = Hyperrectangle([-1.0, -1.0], [10.0, 10.0])
        @test has_components(M)
        p = [1.0, 2.0]
        q = [1.0, 2.0]
        X = [4.0, 5.0]
        for r in [1, 2, Inf]
            @test norm(M, p, X, r) ≈ norm(X, r)
            @test distance(M, p, q, r) ≈ norm(p - q, r)
        end
    end
end
