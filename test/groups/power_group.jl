include("../utils.jl")
include("group_utils.jl")

@testset "Power group" begin
    Mr = SpecialOrthogonal(3)
    Mr1 = PowerManifold(Mr, 5)
    Mrn1 = PowerManifold(Mr, Manifolds.NestedPowerRepresentation(), 5)
    Mrnr1 = PowerManifold(Mr, Manifolds.NestedReplacingPowerRepresentation(), 5)
    Gr1 = PowerGroup(Mr1)
    Grn1 = PowerGroup(Mrn1)
    Grnr1 = PowerGroup(Mrnr1)

    @test_throws ErrorException PowerGroup(PowerManifold(Stiefel(3, 2), 3))

    @test base_manifold(Gr1) === Mr1
    @test base_manifold(Grn1) === Mrn1
    @test base_manifold(Grnr1) === Mrnr1

    @testset "Group $G" for G in [Gr1, Grn1, Grnr1]
        M = base_manifold(G)
        @test G isa PowerGroup
        pts = [rand(G) for _ in 1:3]
        X_pts = [rand(G; vector_at=pts[1]) for _ in 1:3]

        @test compose(G, pts[1], Identity(G)) == pts[1]
        @test compose(G, Identity(G), pts[1]) == pts[1]
        test_group(G, pts, X_pts, X_pts; test_diff=true)

        X = log_lie(G, pts[1])
        Z = zero_vector(G, pts[1])
        log_lie!(G, Z, pts[1])
        @test isapprox(G, pts[1], X, Z)
        p = exp_lie(G, X)
        q = identity_element(G)
        @test is_identity(G, q)
        @test isapprox(G, q, Identity(G))
        @test isapprox(G, Identity(G), q)
        exp_lie!(G, q, X)
        @test isapprox(G, p, q)
        log_lie!(G, Z, Identity(G))
        @test isapprox(G, Identity(G), Z, zero_vector(G, identity_element(G)))
        @test isapprox(
            G,
            Identity(G),
            log_lie(G, Identity(G)),
            zero_vector(G, identity_element(G)),
        )

        @test compose(G, pts[1], Identity(G)) == pts[1]
        @test compose(G, Identity(G), pts[1]) == pts[1]
        test_manifold(G, pts; is_mutating=true)
    end
end
