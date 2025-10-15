using LinearAlgebra, Manifolds, ManifoldsBase, Test

@testset "Heisenberg matrices" begin
    M = HeisenbergMatrices(1)
    @test repr(M) == "HeisenbergMatrices(1)"
    @test is_flat(M)

    pts = [
        [1.0 2.0 3.0; 0.0 1.0 -1.0; 0.0 0.0 1.0],
        [1.0 4.0 -3.0; 0.0 1.0 3.0; 0.0 0.0 1.0],
        [1.0 -2.0 1.0; 0.0 1.0 1.1; 0.0 0.0 1.0],
    ]
    Xpts = [
        [0.0 2.0 3.0; 0.0 0.0 -1.0; 0.0 0.0 0.0],
        [0.0 4.0 -3.0; 0.0 0.0 3.0; 0.0 0.0 0.0],
        [0.0 -2.0 1.0; 0.0 0.0 1.1; 0.0 0.0 0.0],
    ]

    @test check_point(M, [0.0 2.0 3.0; 0.0 1.0 -1.0; 0.0 0.0 1.0]) isa DomainError
    @test check_point(M, [1.0 2.0 3.0; 0.0 1.0 -1.0; 1.0 0.0 1.0]) isa DomainError
    @test check_point(M, [1.0 2.0 3.0; 0.0 2.0 -1.0; 0.0 0.0 1.0]) isa DomainError
    @test check_point(M, [1.0 2.0 3.0; 0.0 1.0 -1.0; 0.0 0.0 2.0]) isa DomainError
    @test check_point(M, [1.0 2.0 3.0; 0.0 1.0 -1.0; 0.0 1.0 1.0]) isa DomainError
    @test check_point(M, [1.0 2.0 3.0; 1.0 1.0 -1.0; 0.0 0.0 1.0]) isa DomainError

    @test check_vector(M, pts[1], [1.0 2.0 3.0; 0.0 0.0 -1.0; 0.0 0.0 0.0]) isa DomainError
    @test check_vector(M, pts[1], [0.0 2.0 3.0; 1.0 0.0 -1.0; 0.0 0.0 0.0]) isa DomainError
    @test check_vector(M, pts[1], [0.0 2.0 3.0; 0.0 0.0 -1.0; 0.0 0.0 2.0]) isa DomainError

    test_manifold(
        M,
        pts;
        basis_types_to_from = (DefaultOrthonormalBasis(),),
        parallel_transport = true,
        test_injectivity_radius = true,
        test_musical_isomorphisms = false,
        test_rand_point = true,
        test_rand_tvector = true,
    )

    @test all(
        iszero,
        Weingarten(M, pts[1], Xpts[1], [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]),
    )
    @testset "field parameter" begin
        G = HeisenbergMatrices(1; parameter = :field)
        @test typeof(get_embedding(G)) === Euclidean{ℝ, Tuple{Int, Int}}
        @test repr(G) == "HeisenbergMatrices(1; parameter=:field)"
    end
end
