include("../utils.jl")
include("group_utils.jl")

@testset "Special Euclidean group" begin
    G = SpecialEuclidean(3)
    @test repr(G) == "SpecialEuclidean(3)"
    M = base_manifold(G)
    @test M === TranslationGroup(3) × SpecialOrthogonal(3)
    Rn = Rotations(3)
    x = Matrix(I, 3, 3)

    t = Vector{Float64}.([1:3, 2:4, 4:6])
    ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
    tuple_pts = [(ti, exp(Rn, x, hat(Rn, x, ωi))) for (ti, ωi) in zip(t, ω)]

    reshapers = (Manifolds.ArrayReshaper(), Manifolds.StaticReshaper())
    for reshaper in reshapers
        shape_se = Manifolds.ShapeSpecification(reshaper, M.manifolds...)
        pts = [Manifolds.prod_point(shape_se, tp...) for tp in tuple_pts]

        g1, g2 = pts[1:2]
        t1, R1 = g1.parts
        t2, R2 = g2.parts
        g1g2 = Manifolds.prod_point(shape_se, R1 * t2 + t1, R1 * R2)
        @test compose(G, g1, g2) ≈ g1g2
        test_group(G, pts)
    end
end
