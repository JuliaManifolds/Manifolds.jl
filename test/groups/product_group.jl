include("../utils.jl")
include("group_utils.jl")

@testset "Product group" begin
    SOn = SpecialOrthogonal(3)
    Tn = TranslationGroup(2)
    Rn = Rotations(3)
    M = ProductManifold(SOn, Tn)
    G = ProductGroup(M)
    @test_throws ErrorException ProductGroup(ProductManifold(Rotations(3), Stiefel(3, 2)))
    @test G isa ProductGroup
    @test submanifold(G, 1) === SOn
    @test submanifold(G, 2) === Tn
    @test base_manifold(G) === M
    @test sprint(show, G) == "ProductGroup($(SOn), $(Tn))"
    @test sprint(show, "text/plain", G) == "ProductGroup with 2 subgroups:\n $(SOn)\n $(Tn)"
    x = Matrix{Float64}(I, 3, 3)
    for f in [exp_lie!, log_lie!]
        @test Manifolds.decorator_transparent_dispatch(f, G, x, x) === Val{:transparent}()
    end
    t = Vector{Float64}.([1:2, 2:3, 3:4])
    ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
    tuple_pts = [(exp(Rn, x, hat(Rn, x, ωi)), ti) for (ωi, ti) in zip(ω, t)]
    tuple_v = (hat(Rn, x, [1.0, 0.5, -0.5]), [-1.0, 2.0])
    eA = [x, zeros(2)]
    shape_se = Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M.manifolds...)
    e = Manifolds.prod_point(shape_se, eA...)

    @testset "product point" begin
        reshapers = (Manifolds.ArrayReshaper(), Manifolds.StaticReshaper())
        for reshaper in reshapers
            shape_se = Manifolds.ShapeSpecification(reshaper, M.manifolds...)
            pts = [Manifolds.prod_point(shape_se, tp...) for tp in tuple_pts]
            v_pts = [Manifolds.prod_point(shape_se, tuple_v...)]
            @test compose(G, pts[1], Identity(G)) == pts[1]
            @test compose(G, Identity(G), pts[1]) == pts[1]
            test_group(G, pts, v_pts, v_pts; test_diff=true)
            @test isapprox(
                G,
                Identity(G),
                exp_lie(G, v_pts[1]),
                Manifolds.prod_point(
                    shape_se,
                    exp_lie(SOn, v_pts[1].parts[1]),
                    exp_lie(Tn, v_pts[1].parts[2]),
                ),
            )
            @test isapprox(
                G,
                Identity(G),
                log_lie(G, pts[1]),
                Manifolds.prod_point(
                    shape_se,
                    log_lie(SOn, pts[1].parts[1]),
                    log_lie(Tn, pts[1].parts[2]),
                ),
            )
        end
    end

    @testset "product repr" begin
        pts = [ProductRepr(tp...) for tp in tuple_pts]
        v_pts = [ProductRepr(tuple_v...)]
        @test compose(G, pts[1], Identity(G)) == pts[1]
        @test compose(G, Identity(G), pts[1]) == pts[1]
        test_group(G, pts, v_pts, v_pts; test_diff=true, test_mutating=false)
        @test isapprox(
            G,
            exp_lie(G, v_pts[1]),
            ProductRepr(exp_lie(SOn, v_pts[1].parts[1]), exp_lie(Tn, v_pts[1].parts[2])),
        )
        @test isapprox(
            G,
            log_lie(G, pts[1]),
            ProductRepr(log_lie(SOn, pts[1].parts[1]), log_lie(Tn, pts[1].parts[2])),
        )
    end
    @test sprint(show, "text/plain", G) === """
    ProductGroup with 2 subgroups:
     SpecialOrthogonal(3)
     TranslationGroup(2; field = ℝ)"""
end
