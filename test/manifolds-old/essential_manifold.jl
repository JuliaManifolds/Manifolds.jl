include("../header.jl")

@testset "the essential distance where a rotation angle reaches zero" begin
    M = EssentialManifold(true)
    Id = Matrix(1.0I, 3, 3)
    Rx = [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 -1.0]
    Rz = [-1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 1.0]
    p = [Id, Id]
    # a half turn about z is shared by both cameras, one about x is not
    @test distance(M, p, [Id, Rz]) ≈ π
    @test distance(M, p, [Rx, Id]) ≈ sqrt(2) * π
    @test distance(M, p, [Id, Rx]) ≈ sqrt(2) * π
end

@testset "the essential manifold has as many coordinates as dimensions" begin
    M = EssentialManifold()
    p = rand(MersenneTwister(42), M)
    X = project(M, p, rand(MersenneTwister(44), M; vector_at = p))
    B = DefaultOrthonormalBasis()
    c = get_coordinates(M, p, X, B)
    @test length(c) == manifold_dimension(M)
    @test isapprox(M, p, get_vector(M, p, c, B), X)
    @test norm(c) ≈ norm(M, p, X)
    Bc = get_basis(M, p, B)
    @test length(get_vectors(M, p, Bc)) == manifold_dimension(M)
    @test get_coordinates(M, p, X, Bc) ≈ c
    @test get_coordinates!(M, similar(c), p, X, Bc) ≈ c
    @test isapprox(M, p, get_vector(M, p, c, Bc), X)
end

@testset "the unsigned essential distance is the same for every representative" begin
    M = EssentialManifold(false)
    Id = Matrix(1.0I, 3, 3)
    Rx = [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 -1.0]
    Ry = [-1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 -1.0]
    Rz = [-1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 1.0]
    p = [Id, Id]
    # turns by 1 about the x axis and by 2 about the y axis, at distance sqrt(2 * (1^2 + 2^2))
    q = [
        [1.0 0.0 0.0; 0.0 cos(1) -sin(1); 0.0 sin(1) cos(1)],
        [cos(2) 0.0 sin(2); 0.0 1.0 0.0; -sin(2) 0.0 cos(2)],
    ]
    for h in [[Id, Id], [Rx, Rx], [Id, Rz], [Rx, Ry]]
        @test distance(M, p, [h[1] * q[1], h[2] * q[2]]) ≈ sqrt(10)
    end
end

@testset "Essential manifold" begin
    M = EssentialManifold()
    a = π / 6
    b = π / 6
    c = π / 6
    r1 = [1.0 0.0 0.0; 0.0 cos(a) -sin(a); 0.0 sin(a) cos(a)]
    r2 = [cos(b) 0.0 sin(b); 0.0 1.0 0.0; -sin(b) 0.0 cos(b)]
    r3 = [cos(c) -sin(c) 0.0; sin(c) cos(c) 0.0; 0.0 0.0 1.0]
    nr = [1.0 0.0 0.0; 0.0 0.0 -1.0; 0.0 -1.0 0.0]
    p1 = [r1, r2]
    p2 = [r1, r3]
    p3 = [r2, r2]
    @testset "Essential manifold Basics" begin
        @test M.manifold == Rotations(3)
        @test repr(M) == "EssentialManifold(true)"
        @test manifold_dimension(M) == 5
        @test !is_flat(M)
        np1 = [r1, nr]
        np2 = [nr, nr]
        np3 = [r1, r2, r3]
        @test !is_point(M, r1)
        # first two components of r1 are not rotations
        @test_throws DomainError is_point(M, r1; error = :error)
        @test_throws DomainError is_point(M, np3; error = :error)
        @test is_point(M, p1)
        @test_throws ComponentManifoldError is_point(M, np1; error = :error)
        @test_throws CompositeManifoldError is_point(M, np2; error = :error)
        @test !is_vector(M, p1, 0.0)
        @test_throws DomainError is_vector(
            M,
            p1,
            [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0];
            error = :error,
        )
        @test !is_vector(M, np1, [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
        @test !is_vector(M, p1, p2)
        # projection test
        @test is_vector(M, p1, project(M, p1, log(M, p1, p2)))
        @test is_vector(M, p1, project(M, p1, log(M, p2, p1)))
    end
    @testset "Signed Essential" begin
        Manifolds.test_manifold(
            M,
            [p1, p2, p3],
            test_vector_spaces = true,
            test_project_point = true,
            projection_atol_multiplier = 10,
            test_project_tangent = false, # since it includes vert_proj.
            test_musical_isomorphisms = false,
            test_default_vector_transport = true,
            test_representation_size = false,
            test_exp_log = true,
            mid_point12 = nothing,
            exp_log_atol_multiplier = 4,
            test_inplace = true,
            parallel_transport = true,
        )
    end
    @testset "Unsigned Essential" begin
        Manifolds.test_manifold(
            EssentialManifold(false),
            [p1, p2, p3],
            test_vector_spaces = true,
            test_project_point = true,
            projection_atol_multiplier = 10,
            test_project_tangent = false, # since it includes vert_proj.
            test_musical_isomorphisms = false,
            test_default_vector_transport = true,
            test_representation_size = false,
            test_exp_log = true,
            mid_point12 = nothing,
            exp_log_atol_multiplier = 4,
            parallel_transport = true,
        )
    end

    @testset "random tangent vectors on the essential manifold are horizontal" begin
        M = EssentialManifold(true)
        N = PowerManifold(Rotations(3), NestedPowerRepresentation(), 2)
        B = DefaultOrthonormalBasis()
        p = rand(MersenneTwister(5), M)
        X = rand(MersenneTwister(5), M; vector_at = p)
        @test distance(M, p, exp(M, p, X)) ≈ norm(M, p, X)
        # the same draw on the pair of rotations differs from X only by a vertical vector
        X0 = rand(MersenneTwister(5), N; vector_at = p)
        @test get_coordinates(M, p, X, B) ≈ get_coordinates(M, p, X0, B)
        Random.seed!(5)
        Y = rand(M; vector_at = p)
        @test norm(get_coordinates(M, p, Y, B)) ≈ norm(M, p, Y)
    end
end
