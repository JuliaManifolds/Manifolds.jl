include("../utils.jl")
include("group_utils.jl")

@testset "Heisenberg group" begin
    G = HeisenbergGroup(1)
    @test repr(G) == "HeisenbergGroup(1)"

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

    ge = allocate(pts[1])
    identity_element!(G, ge)
    @test isapprox(ge, I; atol=1e-10)

    @test check_point(G, [0.0 2.0 3.0; 0.0 1.0 -1.0; 0.0 0.0 1.0]) isa DomainError
    @test check_point(G, [1.0 2.0 3.0; 0.0 1.0 -1.0; 1.0 0.0 1.0]) isa DomainError
    @test check_point(G, [1.0 2.0 3.0; 0.0 2.0 -1.0; 0.0 0.0 1.0]) isa DomainError
    @test check_point(G, [1.0 2.0 3.0; 0.0 1.0 -1.0; 0.0 0.0 2.0]) isa DomainError
    @test check_point(G, [1.0 2.0 3.0; 0.0 1.0 -1.0; 0.0 1.0 1.0]) isa DomainError
    @test check_point(G, [1.0 2.0 3.0; 1.0 1.0 -1.0; 0.0 0.0 1.0]) isa DomainError

    @test check_vector(G, pts[1], [1.0 2.0 3.0; 0.0 0.0 -1.0; 0.0 0.0 0.0]) isa DomainError
    @test check_vector(G, pts[1], [0.0 2.0 3.0; 1.0 0.0 -1.0; 0.0 0.0 0.0]) isa DomainError
    @test check_vector(G, pts[1], [0.0 2.0 3.0; 0.0 0.0 -1.0; 0.0 0.0 2.0]) isa DomainError

    @test compose(G, pts[1], pts[2]) ≈ pts[1] * pts[2]
    test_group(
        G,
        pts,
        Xpts,
        Xpts;
        test_diff=true,
        test_invariance=true,
        test_lie_bracket=true,
        test_adjoint_action=true,
    )

    test_manifold(
        G,
        pts;
        basis_types_to_from=(DefaultOrthonormalBasis(),),
        parallel_transport=true,
        test_injectivity_radius=true,
        test_project_point=true,
        test_project_tangent=true,
        test_musical_isomorphisms=false,
        test_rand_point=true,
        test_rand_tvector=true,
    )
end
