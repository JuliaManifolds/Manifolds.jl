include("../utils.jl")
include("group_utils.jl")

@testset "Product group" begin
    SOn = SpecialOrthogonal(3)
    Tn = TranslationGroup(2)
    Rn = Rotations(3)
    M = ProductManifold(SOn, Tn)
    G = ProductGroup(M)
    @test G isa ProductGroup
    @test base_manifold(G) === M
    @test repr(G) == "ProductGroup($(SOn), $(Tn))"
    x = Matrix{Float64}(I, 3, 3)

    t = Vector{Float64}.([1:2, 2:3, 3:4])
    ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
    tuple_pts = [(exp(Rn, x, hat(Rn, x, ωi)), ti) for (ωi, ti) in zip(ω, t)]
    tuple_v = (hat(Rn, x, [1.0, 0.5, -0.5]), [-1.0, 2.0])

    @testset "product point" begin
        reshapers = (Manifolds.ArrayReshaper(), Manifolds.StaticReshaper())
        for reshaper in reshapers
            shape_se = Manifolds.ShapeSpecification(reshaper, M.manifolds...)
            pts = [Manifolds.prod_point(shape_se, tp...) for tp in tuple_pts]
            v_pts = [Manifolds.prod_point(shape_se, tuple_v...)]
            test_group(G, pts, v_pts; test_diff = true)
        end
    end

    @testset "product repr" begin
        pts = [ProductRepr(tp...) for tp in tuple_pts]
        v_pts = [ProductRepr(tuple_v...)]
        test_group(G, pts, v_pts; test_diff = true, test_mutating = false)
    end
end
