include("../header.jl")

@testset "Hyperrectangle" begin
    M = Hyperrectangle([-1.0, 2.0, -3.0], [1.0, 4.0, 9.0])
    @test repr(M) == "Hyperrectangle([-1.0, 2.0, -3.0], [1.0, 4.0, 9.0])"

    @test_throws ArgumentError Hyperrectangle([1.0, 2.0], [-1.0, 5.0])

    @test is_flat(M)
    p = zeros(3)
    @test project!(M, p, p) == p
    @test embed!(M, p, p) == p
    X = zeros(3)
    X[1] = 1.0
    Y = similar(X)
    project!(M, Y, p, X)
    @test Y == X
    @test embed(M, p, X) == X

    @test_throws DomainError is_point(M, [1.0im, 0.0, 0.0]; error = :error)
    @test_throws DomainError is_point(M, [10.0, 3.0, 0.0]; error = :error)
    @test_throws DomainError is_vector(M, [1.0, 2.0, 0.0], [1.0im, 0.0, 0.0]; error = :error)
    @test_throws DomainError is_vector(M, [1], [1.0, 1.0, 0.0]; error = :error)
    @test_throws DomainError is_vector(M, [0.0, 0.0, 0.0], [1.0]; error = :error)
    @test_throws DomainError is_vector(M, [0.0, 0.0, 0.0], [1.0, 0.0, 1.0im]; error = :error)

    @testset "projections" begin
        @test project(M, [4.0, -2.0, 3.0]) ≈ [1.0, 2.0, 3.0]
        @test project(M, [1.0, 2.0, 3.0], [2.0, 0.5, -10.0]) ≈ [0.0, 0.5, -6.0]
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

    basis_types = (DefaultOrthonormalBasis(),)
    pts = [[1.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.5, 1.0]]

    test_manifold(
        M,
        pts,
        test_injectivity_radius = false,
        parallel_transport = true,
        test_project_point = true,
        test_project_tangent = true,
        test_default_vector_transport = true,
        vector_transport_methods = [ParallelTransport()],
        retraction_methods = [ProjectionRetraction()],
        test_mutating_rand = true,
        basis_types_vecs = basis_types,
        basis_types_to_from = basis_types,
        test_inplace = true,
        test_rand_point = true,
        test_rand_tvector = true,
    )
    @testset "Array Hyperrectangle" begin
        MA = Hyperrectangle([-1.0 2.0 3.0; 2.0 5.0 10.0], [10.0 12.0 13.0; 12.0 15.0 20.0])
        pts_a = [
            [2.0 2.0 3.0; 2.0 6.0 10.0],
            [-1.0 5.0 3.0; 4.0 5.0 10.0],
            [-1.0 2.0 5.0; 6.0 6.0 15.0],
        ]
        test_manifold(
            MA,
            pts_a,
            test_injectivity_radius = false,
            test_project_point = true,
            test_project_tangent = true,
            test_default_vector_transport = true,
            vector_transport_methods = [ParallelTransport()],
            test_mutating_rand = true,
            basis_types_vecs = basis_types,
            basis_types_to_from = basis_types,
            test_inplace = true,
            test_rand_point = true,
            test_rand_tvector = true,
        )
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
    @testset "Volume" begin
        M = Hyperrectangle([-1.0, 2.0, -3.0], [1.0, 4.0, 9.0])
        @test manifold_volume(M) == 48.0
        p = zeros(3)
        X = zeros(3)
        @test volume_density(M, p, X) == 1.0
    end

    @testset "statistics" begin
        @test default_approximation_method(M, mean) === EfficientEstimator()
        @test mean(M, pts) ≈ [1 / 3, 17 / 6, 1 / 3]
        @test mean(M, pts, pweights(ones(3) / 3)) ≈ [1 / 3, 17 / 6, 1 / 3]
        @test_throws DimensionMismatch mean(M, pts, pweights(ones(4) / 4))
        @test var(M, pts) ≈ 1.25
    end

    @testset "Euclidean metric tests" begin
        @test riemann_tensor(
            M,
            pts[1],
            pts[2] - pts[1],
            pts[3] - pts[1],
            (pts[3] - pts[1]) / 2,
        ) == [0.0, 0.0, 0.0]
        @test sectional_curvature(M, p, [1.0, 0.0], [0.0, 1.0]) == 0.0
        @test sectional_curvature_max(M) == 0.0
        @test sectional_curvature_min(M) == 0.0
    end
end
