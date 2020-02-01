include("../utils.jl")
include("group_utils.jl")

@testset "Semidirect product group" begin
    M1 = TranslationGroup(2)
    A = TranslationAction(M1, M1)
    G = SemidirectProductGroup(M1, M1, A)
    @test G === GroupManifold(
        TranslationGroup(2) × TranslationGroup(2),
        Manifolds.SemidirectProductOperation(A),
    )
    @test repr(G) == "SemidirectProductGroup($(M1), $(M1), $(A))"
    @test repr(G.op) == "SemidirectProductOperation($(A))"
    M = base_manifold(G)
    @test M === TranslationGroup(2) × TranslationGroup(2)

    ts1 = Vector{Float64}.([1:2, 2:3, 3:4])
    ts2 = Vector{Float64}.([1:2, 2:3, 3:4]) .* 10
    tuple_pts = [zip(ts1, ts2)...]
    tuple_v = ([-1.0, 1.0], [2.0, 3.0])

    shape_se = Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M.manifolds...)
    pts = [Manifolds.prod_point(shape_se, tp...) for tp in tuple_pts]
    v_pts = [Manifolds.prod_point(shape_se, tuple_v...)]

    e = Identity(G)
    @test inv(G, e) === e
    @test identity(G, e) === e
    @test identity!(G, e, e) === e

    @test compose(G, e, pts[1]) == pts[1]
    @test compose(G, pts[1], e) == pts[1]
    @test compose(G, e, e) === e
end
