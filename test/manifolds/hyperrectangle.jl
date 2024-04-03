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

    @test_throws DomainError is_vector(M, [1], [1.0, 1.0, 0.0]; error=:error)
    @test_throws DomainError is_vector(M, [0.0, 0.0, 0.0], [1.0]; error=:error)
    @test_throws DomainError is_vector(M, [0.0, 0.0, 0.0], [1.0, 0.0, 1.0im]; error=:error)

    @testset "projections" begin
        @test project(M, [4.0, -2.0, 3.0]) ≈ [1.0, 2.0, 3.0]
        @test project(M, [1.0, 2.0, 3.0], [2.0, 0.5, -10.0]) ≈ [0.0, 0.5, -6.0]
    end

    basis_types = (DefaultOrthonormalBasis(),)
    pts = [[1.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.5, 1.0]]
    test_manifold(
        M,
        pts,
        test_project_point=true,
        test_project_tangent=true,
        test_default_vector_transport=true,
        vector_transport_methods=[ParallelTransport()],
        test_mutating_rand=true,
        basis_types_vecs=basis_types,
        basis_types_to_from=basis_types,
        test_inplace=true,
        test_rand_point=true,
        test_rand_tvector=true,
    )

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
end
