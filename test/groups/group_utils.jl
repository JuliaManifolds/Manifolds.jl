

"""
    test_action(A::AbstractActionOnManifold,
        a_pts::AbstractVector,
        m_pts::AbstractVector)

Tests general properties of the action `A`, given at least three different points
that lie on it (contained in `a_pts`) and three different point that lie
on the manifold it acts upon (contained in `m_pts`).
"""
function test_action(
        A::AbstractActionOnManifold,
        a_pts::AbstractVector,
        m_pts::AbstractVector;
        atol_inv = 1e-10,
        test_optimal_alignment = false
    )

    G = base_group(A)
    M = action_on(A)
    e = Identity(G)
    @testset "Type calculation and adding identity element" begin
        for ap in a_pts
            e_ap = e(ap)
            @test isapprox(G, e_ap, identity(G, ap))
            @test isapprox(G, compose_left(G, e, ap), ap)
            @test isapprox(G, compose_left(G, e_ap, ap), ap)
            @test isapprox(G, compose_left(G, ap, e), ap)
            @test isapprox(G, compose_left(G, ap, e_ap), ap)
            inv_ap = inv(G, ap)
            @test isapprox(G, compose_left(G, ap, inv_ap), e_ap; atol = atol_inv)
            @test isapprox(G, compose_left(G, inv_ap, ap), e_ap; atol = atol_inv)
            ap2 = similar(e_ap)
            identity!(G, ap2, e_ap)
            @test isapprox(G, ap2, e_ap)
        end
    end

    @testset "Associativity" begin
        N = length(a_pts)
        for i in 1:N
            assoc_l = compose_left(G, a_pts[i], a_pts[mod1(i+1, N)], a_pts[mod1(i+2, N)])
            assoc_l2 = compose_left(G, compose_left(G, a_pts[i], a_pts[mod1(i+1, N)]), a_pts[mod1(i+2, N)])
            assoc_r = compose_left(G, a_pts[i], compose_left(G, a_pts[mod1(i+1, N)]), a_pts[mod1(i+2, N)])
            @test isapprox(G, assoc_l, assoc_r)
            @test isapprox(G, assoc_l, assoc_l2)
        end
    end

    @testset "Compatibility" begin
        N = length(a_pts)
        for p in m_pts
            for i in 1:N
                aip = apply(A, p, a_pts[mod1(i+1, N)])
                acg = compose_left(G, a_pts[i], a_pts[mod1(i+1, N)])
                @test isapprox(M, apply(A, p, acg), apply(A, aip, a_pts[i]))
            end
        end
    end

    @testset "Mutable apply!" begin
        N = length(a_pts)
        for p in m_pts
            for i in 1:N
                aip = similar(p)
                apply!(A, aip, p, a_pts[i])
                @test isapprox(M, apply(A, p, a_pts[i]), aip)
            end
        end
    end

    @testset "Mutable inv!" begin
        for ap in a_pts
            ai = similar(ap)
            inv!(G, ai, ap)
            @test isapprox(G, compose_left(G, ai, ap), e(ap))
            @test isapprox(G, compose_left(G, ap, ai), e(ap))
        end
    end

    @testset "Translate vs compose_left" begin
        N = length(a_pts)
        for i in 1:N
            ai = a_pts[i]
            aip = a_pts[mod1(i+1, N)]
            @test isapprox(G, translate(G, ai, aip), compose_left(G, ai, aip))
            @test isapprox(G, translate(G, ai, aip, LeftAction()), compose_left(G, ai, aip))
            @test isapprox(G, translate(G, aip, ai, RightAction()), compose_left(G, ai, aip))
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

    @testset "Action of group identity" begin
        e_ap = e(a_pts[1])
        for p in m_pts
            @test isapprox(M, p, apply(A, p, e))
            @test isapprox(M, p, apply(A, p, e_ap))
        end
    end

    test_optimal_alignment && @testset "Center of orbit" begin
        act = center_of_orbit(A, [m_pts[1]], m_pts[2])
        act_opt = optimal_alignment(A, m_pts[2], m_pts[1])
        @test isapprox(G, act, act_opt)
    end
end
