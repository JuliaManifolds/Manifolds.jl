include("../utils.jl")
include("group_utils.jl")

@testset "Translation group" begin
    @testset "real" begin
        G = TranslationGroup(2, 3)
        @test repr(G) == "TranslationGroup(2, 3; field = ℝ)"
        @test repr(TranslationGroup(2, 3; field=ℂ)) == "TranslationGroup(2, 3; field = ℂ)"

        @test has_invariant_metric(G, LeftAction())
        @test has_invariant_metric(G, RightAction())
        @test has_biinvariant_metric(G)
        @test is_default_metric(MetricManifold(G, EuclideanMetric())) === true
        @test is_group_manifold(G)
        @test !is_group_manifold(G.manifold)
        types = [Matrix{Float64}]
        @test base_manifold(G) === Euclidean(2, 3)
        @test log_lie(G, Identity(G)) == zeros(2, 3) # log_lie with Identity on Addition group.
        pts = [reshape(i:(i + 5), (2, 3)) for i in 1:3]
        Xpts = [reshape(-2:3, (2, 3)), reshape(-1:4, (2, 3))]
        # test fallback exp/exp! that the fallback works correctly, cf.
        # https://github.com/JuliaManifolds/Manifolds.jl/issues/483
        q = exp(G, pts[1], Xpts[1])
        q2 = copy(G, pts[1])
        exp!(G, q2, pts[1], Xpts[1])
        @test q == q2
        for T in types
            gpts = convert.(T, pts)
            Xgpts = convert.(T, Xpts)
            @test compose(G, gpts[1], gpts[2]) ≈ gpts[1] + gpts[2]
            test_group(
                G,
                gpts,
                Xgpts,
                Xgpts;
                test_diff=true,
                test_invariance=true,
                test_lie_bracket=true,
                test_adjoint_action=true,
                test_log_from_identity=true,
            )
        end
    end

    @testset "complex" begin
        G = TranslationGroup(2, 3; field=ℂ)
        @test repr(G) == "TranslationGroup(2, 3; field = ℂ)"

        types = [Matrix{ComplexF64}]
        @test base_manifold(G) === Euclidean(2, 3; field=ℂ)

        pts = [reshape(complex.(i:(i + 5), (i + 1):(i + 6)), (2, 3)) for i in 1:3]
        Xpts = [reshape(complex.(-2:3, -1:4), (2, 3)), reshape(complex.(-1:4, 0:5), (2, 3))]
        for T in types
            gpts = convert.(T, pts)
            Xgpts = convert.(T, Xpts)
            @test compose(G, gpts[1], gpts[2]) ≈ gpts[1] + gpts[2]
            @test translate_diff(G, gpts[2], gpts[1], Xgpts[1]) ≈ Xgpts[1]
            test_group(
                G,
                gpts,
                Xgpts,
                Xgpts;
                test_diff=true,
                test_invariance=true,
                test_lie_bracket=true,
                test_adjoint_action=true,
            )
        end
    end
end
