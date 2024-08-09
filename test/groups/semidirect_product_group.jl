include("../header.jl")
include("group_utils.jl")

@testset "Semidirect product group" begin
    M1 = TranslationGroup(2)
    A = TranslationAction(M1, M1)
    G = SemidirectProductGroup(M1, M1, A, Manifolds.LeftInvariantRepresentation())
    @test G === GroupManifold(
        TranslationGroup(2) × TranslationGroup(2),
        Manifolds.SemidirectProductOperation(A),
        Manifolds.LeftInvariantRepresentation(),
    )
    @test repr(G) == "SemidirectProductGroup($(M1), $(M1), $(A))"
    @test repr(G.op) == "SemidirectProductOperation($(A))"
    M = base_manifold(G)
    @test M === TranslationGroup(2) × TranslationGroup(2)

    ts1 = Vector{Float64}.([1:2, 2:3, 3:4])
    ts2 = Vector{Float64}.([1:2, 2:3, 3:4]) .* 10
    tuple_pts = [zip(ts1, ts2)...]

    pts = [ArrayPartition(tp...) for tp in tuple_pts]

    @testset "setindex! and getindex" begin
        p1 = pts[1]
        p2 = allocate(p1)
        @test p1[G, 1] === p1[M, 1]
        p2[G, 1] = p1[M, 1]
        @test p2[G, 1] == p1[M, 1]
    end

    X = log(base_manifold(G), pts[1], pts[1])
    Y = zero_vector(G, pts[1])
    Z = Manifolds.allocate_result(G, zero_vector, pts[1])
    Z = zero_vector!(M, Z, pts[1])
    @test norm(G, pts[1], X) ≈ 0
    @test norm(G, pts[1], Y) ≈ 0
    @test norm(G, pts[1], Z) ≈ 0

    e = Identity(G)
    @test inv(G, e) === e

    @test compose(G, e, pts[1]) == pts[1]
    @test compose(G, pts[1], e) == pts[1]
    @test compose(G, e, e) === e

    # test in-place composition
    o1 = copy(pts[1])
    compose!(G, o1, o1, pts[2])
    @test isapprox(G, o1, compose(G, pts[1], pts[2]))

    eA = identity_element(G)
    @test isapprox(G, eA, e)
    @test isapprox(G, e, eA)
    W = log(base_manifold(G), eA, pts[1])
    Z = log(base_manifold(G), eA, pts[1])
    @test isapprox(G, e, W, Z)
end
