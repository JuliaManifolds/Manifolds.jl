struct NotImplementedOperation <: AbstractGroupOperation end

struct NotImplementedManifold <: Manifold end

"""
    test_group(G::AbstractGroupManifold, g_pts::AbstractVector)

Tests general properties of the group `G`, given at least three different points
elements of it (contained in `g_pts`).
Optionally, specify `test_diff` to test differentials of translation, using `v_pts`, which
must contain at least one tangent vector at `g_pts[1]`.
"""
function test_group(
    G::AbstractGroupManifold,
    g_pts::AbstractVector,
    v_pts::AbstractVector = [];
    test_mutating = true,
    test_diff = false,
)
    e = Identity(G)

    @testset "Basic group properties" begin
        @testset "Closed" begin
            for g1 in g_pts, g2 in g_pts
                g3 = compose(G, g1, g2)
                @test is_manifold_point(G, g3)
            end
        end

        @testset "Associative" begin
            g12_3 = compose(G, compose(G, g_pts[1], g_pts[2]), g_pts[3])
            g1_23 = compose(G, g_pts[1], compose(G, g_pts[2], g_pts[3]))
            @test isapprox(G, g12_3, g1_23)

            test_mutating && @testset "mutating" begin
                g12, g23, g12_3, g1_23 = similar.(repeat([g_pts[1]], 4))
                @test compose!(G, g12, g_pts[1], g_pts[2]) === g12
                @test compose!(G, g23, g_pts[2], g_pts[3]) === g23
                @test compose!(G, g12_3, g12, g_pts[3]) === g12_3
                @test compose!(G, g1_23, g_pts[1], g23) === g1_23
                @test isapprox(G, g12_3, g1_23)
            end
        end

        @testset "Identity" begin
            @test isapprox(G, e, e)
            @test identity(G, e) === e
            @test compose(G, e, e) === e
            @test copyto!(e, e) === e

            for g in g_pts
                @test isapprox(G, compose(G, g, e), g)
                @test isapprox(G, compose(G, e, g), g)

                ge = identity(G, g)
                @test isapprox(G, compose(G, g, ge), g)
                @test isapprox(G, compose(G, ge, g), g)
            end

            test_mutating && @testset "mutating" begin
                for g in g_pts
                    h = similar(g)
                    @test compose!(G, h, g, e) === h
                    @test isapprox(G, h, g)
                    h = similar(g)
                    @test compose!(G, h, e, g) === h
                    @test isapprox(G, h, g)

                    ge = similar(g)
                    @test identity!(G, ge, e) === ge
                    @test isapprox(G, compose(G, g, ge), g)
                    @test isapprox(G, compose(G, ge, g), g)

                    ge = similar(g)
                    @test compose!(G, ge, e, e) === ge
                    @test isapprox(G, ge, e)
                end
            end
        end

        @testset "Inverse" begin
            for g in g_pts
                ginv = inv(G, g)
                @test isapprox(G, compose(G, g, ginv), e)
                @test isapprox(G, compose(G, ginv, g), e)
                @test isapprox(G, e, compose(G, g, ginv))
                @test isapprox(G, e, compose(G, ginv, g))
                @test inv(G, e) === e

                test_mutating && @testset "mutating" begin
                    ginv = similar(g)
                    @test inv!(G, ginv, g) === ginv
                    @test isapprox(G, compose(G, g, ginv), e)
                    @test isapprox(G, compose(G, ginv, g), e)

                    @test inv(G, e) === e
                    geinv = similar(g)
                    @test inv!(G, geinv, e) === geinv
                    @test isapprox(G, geinv, e)
                end
            end
        end
    end

    @testset "translation" begin
        convs = ((), (LeftAction(),), (RightAction(),))

        @test translate(G, g_pts[1], g_pts[2]) ≈ compose(G, g_pts[1], g_pts[2])
        @test translate(G, g_pts[1], g_pts[2], LeftAction()) ≈ compose(G, g_pts[1], g_pts[2])
        @test translate(G, g_pts[1], g_pts[2], RightAction()) ≈ compose(G, g_pts[2], g_pts[1])

        for conv in convs
            @test inverse_translate(G, g_pts[1], translate(G, g_pts[1], g_pts[2], conv...), conv...) ≈ g_pts[2]
            @test translate(G, g_pts[1], inverse_translate(G, g_pts[1], g_pts[2], conv...), conv...) ≈ g_pts[2]
        end

        test_mutating && @testset "mutating" begin
            for conv in convs
                g = similar(g_pts[1])
                @test translate!(G, g, g_pts[1], g_pts[2], conv...) === g
                @test g ≈ translate(G, g_pts[1], g_pts[2], conv...)

                g = translate(G, g_pts[1], g_pts[2], conv...)
                g2 = similar(g)
                @test inverse_translate!(G, g2, g_pts[1], g, conv...) === g2
                @test g2 ≈ g_pts[2]

                g = inverse_translate(G, g_pts[1], g_pts[2], conv...)
                g2 = similar(g)
                @test translate!(G, g2, g_pts[1], g, conv...) === g2
                @test g2 ≈ g_pts[2]
            end
        end
    end

    test_diff && @testset "translation differential" begin
        v = v_pts[1]
        convs = ((), (LeftAction(),), (RightAction(),))
        g21 = compose(G, g_pts[2], g_pts[1])
        g12 = compose(G, g_pts[1], g_pts[2])
        @test translate_diff(G, g_pts[2], g_pts[1], v) ≈ translate_diff(G, g_pts[2], g_pts[1], v, LeftAction())
        @test is_tangent_vector(G, g12, translate_diff(G, g_pts[2], g_pts[1], v, LeftAction()))
        @test is_tangent_vector(G, g21, translate_diff(G, g_pts[2], g_pts[1], v, RightAction()))

        for conv in convs
            @test inverse_translate_diff(G, g_pts[2], g_pts[1], translate_diff(G, g_pts[2], g_pts[1], v, conv...), conv...) ≈ v
            @test translate_diff(G, g_pts[2], g_pts[1], inverse_translate_diff(G, g_pts[2], g_pts[1], v, conv...), conv...) ≈ v
        end

        test_mutating && @testset "mutating" begin
            for conv in convs
                vout = similar(v)
                @test translate_diff!(G, vout, g_pts[2], g_pts[1], v, conv...) === vout
                @test vout ≈ translate_diff(G, g_pts[2], g_pts[1], v, conv...)

                vout = translate_diff(G, g_pts[2], g_pts[1], v, conv...)
                vout2 = similar(vout)
                @test inverse_translate_diff!(G, vout2, g_pts[2], g_pts[1], vout, conv...) === vout2
                @test vout2 ≈ v

                vout = inverse_translate_diff(G, g_pts[2], g_pts[1], v, conv...)
                vout2 = similar(vout)
                @test translate_diff!(G, vout2, g_pts[2], g_pts[1], vout, conv...) === vout2
                @test vout2 ≈ v
            end
        end
    end
end


"""
    test_action(A::AbstractGroupAction,
        a_pts::AbstractVector,
        m_pts::AbstractVector)

Tests general properties of the action `A`, given at least three different points
that lie on it (contained in `a_pts`) and three different point that lie
on the manifold it acts upon (contained in `m_pts`).
"""
function test_action(
        A::AbstractGroupAction,
        a_pts::AbstractVector,
        m_pts::AbstractVector;
        atol_inv = 1e-10,
        test_optimal_alignment = false
    )

    G = base_group(A)
    GM = base_manifold(G)
    M = g_manifold(A)
    e = Identity(G)

    @testset "Action direction" begin
        if isa(A, AbstractGroupAction{LeftAction})
            @test isa(switch_direction(A), AbstractGroupAction{RightAction})
        else
            @test isa(switch_direction(A), AbstractGroupAction{LeftAction})
        end
    end

    @testset "Type calculation and adding identity element" begin
        for ap in a_pts
            e_ap = e(ap)
            @test isapprox(G, e_ap, identity(G, ap))
            @test isapprox(G, compose(G, e, ap), ap)
            @test isapprox(G, compose(G, e_ap, ap), ap)
            @test isapprox(G, compose(G, ap, e), ap)
            @test isapprox(G, compose(G, ap, e_ap), ap)
            inv_ap = inv(G, ap)
            @test isapprox(G, compose(G, ap, inv_ap), e_ap; atol = atol_inv)
            @test isapprox(G, compose(G, inv_ap, ap), e_ap; atol = atol_inv)
            ap2 = similar(e_ap)
            identity!(G, ap2, e_ap)
            @test isapprox(G, ap2, e_ap)
        end

        @test inv(G, e) == e
        b = similar(a_pts[1])
    end

    @testset "Associativity" begin
        N = length(a_pts)
        for i in 1:N
            assoc_l = compose(G, compose(G, a_pts[i], a_pts[mod1(i+1, N)]), a_pts[mod1(i+2, N)])
            assoc_r = compose(G, a_pts[i], compose(G, a_pts[mod1(i+1, N)], a_pts[mod1(i+2, N)]))
            @test isapprox(G, assoc_l, assoc_r)
        end
    end

    @testset "Compatibility" begin
        N = length(a_pts)
        for p in m_pts
            for i in 1:N
                aip = apply(A, a_pts[mod1(i+1, N)], p)
                acg = compose(G, a_pts[i], a_pts[mod1(i+1, N)])
                @test isapprox(M, apply(A, acg, p), apply(A, a_pts[i], aip))
            end
        end
    end

    @testset "Mutable apply!" begin
        N = length(a_pts)
        for p in m_pts
            for i in 1:N
                aip = similar(p)
                apply!(A, aip, a_pts[i], p)
                @test isapprox(M, apply(A, a_pts[i], p), aip)
            end
        end
    end

    @testset "Mutable inv!" begin
        for ap in a_pts
            ai = similar(ap)
            inv!(G, ai, ap)
            @test isapprox(G, compose(G, ai, ap), e(ap))
            @test isapprox(G, compose(G, ap, ai), e(ap))
        end
    end

    @testset "Translate vs compose" begin
        N = length(a_pts)
        for i in 1:N
            ai = a_pts[i]
            aip = a_pts[mod1(i+1, N)]
            @test isapprox(G, translate(G, ai, aip), compose(G, ai, aip))
            @test isapprox(G, translate(G, ai, aip, LeftAction()), compose(G, ai, aip))
            @test isapprox(G, translate(G, aip, ai, RightAction()), compose(G, ai, aip))
        end
    end

    @testset "Translate mutation" begin
        N = length(a_pts)
        for i in 1:N
            ai = a_pts[i]
            aip = a_pts[mod1(i+1, N)]
            as = similar(ai)
            translate!(G, as, ai, aip)
            @test isapprox(G, as, translate(G, ai, aip))
            translate!(G, as, ai, aip, LeftAction())
            @test isapprox(G, as, translate(G, ai, aip, LeftAction()))
            translate!(G, as, ai, aip, RightAction())
            @test isapprox(G, as, translate(G, ai, aip, RightAction()))
        end
    end

    @testset "translate_diff" begin
        tv = log(GM, a_pts[1], a_pts[2])
        tvd = translate_diff(G, a_pts[2], a_pts[1], tv)
        @test is_tangent_vector(G, compose(G, a_pts[1], a_pts[2]), tvd)
        tvdi = inverse_translate_diff(G, inv(G, a_pts[2]), a_pts[1], tv)
        @test isapprox(G, a_pts[1], tvd, tvdi)
    end

    @testset "mutating translate_diff!" begin
        tv = log(GM, a_pts[1], a_pts[2])
        tvd = similar(tv)
        translate_diff!(G, tvd, a_pts[2], a_pts[1], tv)
        @test isapprox(GM, a_pts[1], tvd, translate_diff(G, a_pts[2], a_pts[1], tv))

        tvdi = similar(tv)
        inverse_translate_diff!(G, tvdi, inv(G, a_pts[2]), a_pts[1], tv)
        @test isapprox(GM, a_pts[1], tvd, tvdi)
    end

    @testset "Action of group identity" begin
        e_ap = e(a_pts[1])
        for p in m_pts
            @test isapprox(M, p, apply(A, e, p))
            @test isapprox(M, p, apply(A, e_ap, p))
        end
    end

    @testset "Action composition" begin
        a12 = compose(A, a_pts[1], a_pts[2])
        b = similar(a_pts[1])
        compose!(A, b, a_pts[1], a_pts[2])
        @test isapprox(G, a12, b)

        if isa(A, AbstractGroupAction{LeftAction})
            x_a = apply(A, a_pts[1], apply(A, a_pts[2], m_pts[1]))
            x_b = apply(A, a12, m_pts[1])
            @test isapprox(M, x_a, x_b)
        elseif isa(A, AbstractGroupAction{RightAction})
            x_a = apply(A, a_pts[2], apply(A, a_pts[1], m_pts[1]))
            x_b = apply(A, a12, m_pts[1])
            @test isapprox(M, x_a, x_b)
        else
            # most likely a bug in testset
            @test false
        end
    end

    test_optimal_alignment && @testset "Center of orbit" begin
        act = center_of_orbit(A, [m_pts[1]], m_pts[2])
        act_opt = optimal_alignment(A, m_pts[2], m_pts[1])
        @test isapprox(G, act, act_opt)
    end
end
