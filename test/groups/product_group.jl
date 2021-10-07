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

    @testset "Product Identity" begin
        i = Identity(G)
        @test submanifold_components(G, i) == (Identity(SOn), Identity(Tn))
        @test submanifold_component(G, i, 1) == Identity(SOn)
        @test submanifold_component(G, i, 2) == Identity(Tn)
    end

    @testset "product repr" begin
        pts = [ProductRepr(tp...) for tp in tuple_pts]
        X_pts = [ProductRepr(tuple_v...)]

        @test compose(G, pts[1], Identity(G)) == pts[1]
        @test compose(G, Identity(G), pts[1]) == pts[1]
        test_group(G, pts, X_pts, X_pts; test_diff=true)
        @test isapprox(
            G,
            Identity(G),
            exp_lie(G, X_pts[1]),
            ProductRepr(exp_lie(SOn, X_pts[1].parts[1]), exp_lie(Tn, X_pts[1].parts[2])),
        )
        @test isapprox(
            G,
            Identity(G),
            log_lie(G, pts[1]),
            ProductRepr(log_lie(SOn, pts[1].parts[1]), log_lie(Tn, pts[1].parts[2])),
        )
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
        test_group(G, pts, X_pts, X_pts; test_diff=true, test_mutating=false)
        @test isapprox(
            G,
            exp_lie(G, X_pts[1]),
            ProductRepr(exp_lie(SOn, X_pts[1].parts[1]), exp_lie(Tn, X_pts[1].parts[2])),
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
