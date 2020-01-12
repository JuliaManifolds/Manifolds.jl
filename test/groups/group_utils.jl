struct NotImplementedOperation <: AbstractGroupOperation end

struct NotImplementedManifold <: Manifold end

"""
    test_group(
        G::AbstractGroupManifold,
        g_pts::AbstractVector,
        v_pts::AbstractVector = [];
        atol = 1e-10,
        test_mutating = true,
        test_diff = false,
        diff_convs = [(), (LeftAction(),), (RightAction(),)],
    )

Tests general properties of the group `G`, given at least three different points
elements of it (contained in `g_pts`).
Optionally, specify `test_diff` to test differentials of translation, using `v_pts`, which
must contain at least one tangent vector at `g_pts[1]`, and the direction conventions
specified in `diff_convs`.
"""
function test_group(
    G::AbstractGroupManifold,
    g_pts::AbstractVector,
    v_pts::AbstractVector = [];
    atol = 1e-10,
    test_mutating = true,
    test_diff = false,
    diff_convs = [(), (LeftAction(),), (RightAction(),)],
)
    e = Identity(G)

    @testset "Basic group properties" begin
        @testset "Closed" begin
            for g1 in g_pts, g2 in g_pts
                g3 = compose(G, g1, g2)
                @test is_manifold_point(G, g3, true; atol = atol)
            end
        end

        @testset "Associative" begin
            g12_3 = compose(G, compose(G, g_pts[1], g_pts[2]), g_pts[3])
            g1_23 = compose(G, g_pts[1], compose(G, g_pts[2], g_pts[3]))
            @test isapprox(G, g12_3, g1_23; atol = atol)

            test_mutating && @testset "mutating" begin
                g12, g23, g12_3, g1_23 = similar.(repeat([g_pts[1]], 4))
                @test compose!(G, g12, g_pts[1], g_pts[2]) === g12
                @test compose!(G, g23, g_pts[2], g_pts[3]) === g23
                @test compose!(G, g12_3, g12, g_pts[3]) === g12_3
                @test compose!(G, g1_23, g_pts[1], g23) === g1_23
                @test isapprox(G, g12_3, g1_23; atol = atol)
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
                @test isapprox(G, compose(G, g, ginv), e; atol = atol)
                @test isapprox(G, compose(G, ginv, g), e; atol = atol)
                @test isapprox(G, e, compose(G, g, ginv); atol = atol)
                @test isapprox(G, e, compose(G, ginv, g); atol = atol)
                @test inv(G, e) === e

                test_mutating && @testset "mutating" begin
                    ginv = similar(g)
                    @test inv!(G, ginv, g) === ginv
                    @test isapprox(G, compose(G, g, ginv), e; atol = atol)
                    @test isapprox(G, compose(G, ginv, g), e; atol = atol)

                    @test inv(G, e) === e
                    geinv = similar(g)
                    @test inv!(G, geinv, e) === geinv
                    @test isapprox(G, geinv, e; atol = atol)
                end
            end
        end
    end

    @testset "translation" begin
        convs = ((), (LeftAction(),), (RightAction(),))

        @test translate(G, g_pts[1], g_pts[2]) ≈ compose(G, g_pts[1], g_pts[2]) atol = atol
        @test translate(G, g_pts[1], g_pts[2], LeftAction()) ≈ compose(G, g_pts[1], g_pts[2]) atol = atol
        @test translate(G, g_pts[1], g_pts[2], RightAction()) ≈ compose(G, g_pts[2], g_pts[1]) atol = atol

        for conv in convs
            @test inverse_translate(G, g_pts[1], translate(G, g_pts[1], g_pts[2], conv...), conv...) ≈ g_pts[2] atol = atol
            @test translate(G, g_pts[1], inverse_translate(G, g_pts[1], g_pts[2], conv...), conv...) ≈ g_pts[2] atol = atol
        end

        test_mutating && @testset "mutating" begin
            for conv in convs
                g = similar(g_pts[1])
                @test translate!(G, g, g_pts[1], g_pts[2], conv...) === g
                @test g ≈ translate(G, g_pts[1], g_pts[2], conv...) atol = atol

                g = translate(G, g_pts[1], g_pts[2], conv...)
                g2 = similar(g)
                @test inverse_translate!(G, g2, g_pts[1], g, conv...) === g2
                @test g2 ≈ g_pts[2] atol = atol

                g = inverse_translate(G, g_pts[1], g_pts[2], conv...)
                g2 = similar(g)
                @test translate!(G, g2, g_pts[1], g, conv...) === g2
                @test g2 ≈ g_pts[2] atol = atol
            end
        end
    end

    test_diff && @testset "translation differential" begin
        v = v_pts[1]
        g21 = compose(G, g_pts[2], g_pts[1])
        g12 = compose(G, g_pts[1], g_pts[2])
        @test translate_diff(G, g_pts[2], g_pts[1], v) ≈ translate_diff(G, g_pts[2], g_pts[1], v, LeftAction()) atol = atol
        @test is_tangent_vector(G, g12, translate_diff(G, g_pts[2], g_pts[1], v, LeftAction()), true; atol = atol)
        RightAction() in diff_convs && @test is_tangent_vector(G, g21, translate_diff(G, g_pts[2], g_pts[1], v, RightAction()), true; atol = atol)

        for conv in diff_convs
            g2g1 = translate(G, g_pts[2], g_pts[1], conv...)
            g2invg1 = inverse_translate(G, g_pts[2], g_pts[1], conv...)
            @test inverse_translate_diff(G, g_pts[2], g2g1, translate_diff(G, g_pts[2], g_pts[1], v, conv...), conv...) ≈ v atol = atol
            @test translate_diff(G, g_pts[2], g2invg1, inverse_translate_diff(G, g_pts[2], g_pts[1], v, conv...), conv...) ≈ v atol = atol
        end

        test_mutating && @testset "mutating" begin
            for conv in diff_convs
                g2g1 = translate(G, g_pts[2], g_pts[1], conv...)
                g2invg1 = inverse_translate(G, g_pts[2], g_pts[1], conv...)
                vout = similar(v)
                @test translate_diff!(G, vout, g_pts[2], g_pts[1], v, conv...) === vout
                @test vout ≈ translate_diff(G, g_pts[2], g_pts[1], v, conv...) atol = atol

                vout = translate_diff(G, g_pts[2], g_pts[1], v, conv...)
                vout2 = similar(vout)
                @test inverse_translate_diff!(G, vout2, g_pts[2], g2g1, vout, conv...) === vout2
                @test vout2 ≈ v atol = atol

                vout = inverse_translate_diff(G, g_pts[2], g_pts[1], v, conv...)
                vout2 = similar(vout)
                @test translate_diff!(G, vout2, g_pts[2], g2invg1, vout, conv...) === vout2
                @test vout2 ≈ v atol = atol
            end
        end
    end
end

"""
    test_action(
        A::AbstractGroupAction,
        a_pts::AbstractVector,
        m_pts::AbstractVector;
        atol = 1e-10,
        test_optimal_alignment = false,
        test_mutating = true,
        test_diff = false,
    )

Tests general properties of the action `A`, given at least three different points
that lie on it (contained in `a_pts`) and three different point that lie
on the manifold it acts upon (contained in `m_pts`).
"""
function test_action(
        A::AbstractGroupAction,
        a_pts::AbstractVector,
        m_pts::AbstractVector,
        v_pts = [];
        atol = 1e-10,
        test_optimal_alignment = false,
        test_mutating = true,
        test_diff = false,
    )

    G = base_group(A)
    M = g_manifold(A)
    e = Identity(G)

    @testset "Basic action properties" begin
        @testset "Direction" begin
            if isa(A, AbstractGroupAction{LeftAction})
                @test direction(A) === LeftAction()
                @test isa(switch_direction(A), AbstractGroupAction{RightAction})
                @test direction(switch_direction(A)) === RightAction()
            else
                @test direction(A) === RightAction()
                @test isa(switch_direction(A), AbstractGroupAction{LeftAction})
                @test direction(switch_direction(A)) === LeftAction()
            end
        end

        @testset "Closed" begin
            @testset "over actions" begin
                for a1 in a_pts, a2 in a_pts
                    a3 = compose(A, a1, a2)
                    @test is_manifold_point(G, a3, true; atol = atol)
                end
            end
            @testset "over g-manifold" begin
                for a in a_pts, m in m_pts
                    @test is_manifold_point(M, apply(A, a, m), true; atol = atol)
                    @test is_manifold_point(M, inverse_apply(A, a, m), true; atol = atol)
                end
            end
        end

        @testset "Associative" begin
            a12 = compose(A, a_pts[1], a_pts[2])
            a23 = compose(A, a_pts[2], a_pts[3])

            @testset "over compose" begin
                a12_a3 = compose(A, a12, a_pts[3])
                a1_a23 = compose(A, a_pts[1], a23)
                @test isapprox(G, a12_a3, a1_a23)
            end

            @testset "over apply" begin
                for m in m_pts
                    a12_a3_m = apply(A, a12, apply(A, a_pts[3], m))
                    a1_a23_m = apply(A, a_pts[1], apply(A, a23, m))
                    @test isapprox(M, a12_a3_m, a1_a23_m)
                end
            end

            test_mutating && @testset "mutating" begin
                a12, a23, a12_3, a1_23 = similar.(repeat([a_pts[1]], 4))
                @test compose!(A, a12, a_pts[1], a_pts[2]) === a12
                @test compose!(A, a23, a_pts[2], a_pts[3]) === a23
                @test compose!(A, a12_3, a12, a_pts[3]) === a12_3
                @test compose!(A, a1_23, a_pts[1], a23) === a1_23
                @test isapprox(G, a12_3, a1_23)

                for m in m_pts
                    a12_a3_m, a1_a23_m = similar(m), similar(m)
                    @test apply!(A, a12_a3_m, a12, apply(A, a_pts[3], m)) === a12_a3_m
                    @test apply!(A, a1_a23_m, a_pts[1], apply(A, a23, m)) === a1_a23_m
                    @test isapprox(M, a12_a3_m, a1_a23_m)
                end
            end
        end

        @testset "Identity" begin
            @test compose(A, e, e) === e

            for a in a_pts
                @test isapprox(G, compose(A, a, e), a)
                @test isapprox(G, compose(A, e, a), a)

                ge = identity(G, a)
                @test isapprox(G, compose(A, a, ge), a)
                @test isapprox(G, compose(A, ge, a), a)

                for m in m_pts
                    @test isapprox(M, apply(A, e, m), m)
                    @test isapprox(M, apply(A, ge, m), m)
                    @test isapprox(M, inverse_apply(A, e, m), m)
                    @test isapprox(M, inverse_apply(A, ge, m), m)
                end
            end

            test_mutating && @testset "mutating" begin
                for a in a_pts
                    h = similar(a)
                    @test compose!(A, h, a, e) === h
                    @test isapprox(G, h, a)
                    h = similar(a)
                    @test compose!(A, h, e, a) === h
                    @test isapprox(G, h, a)

                    ge = identity(G, a)
                    @test isapprox(G, compose(A, a, ge), a)
                    @test isapprox(G, compose(A, ge, a), a)

                    ge = identity(G, a)
                    @test compose!(A, ge, e, e) === ge
                    @test isapprox(G, ge, e)

                    for m in m_pts
                        em = similar(m)
                        @test apply!(A, em, e, m) === em
                        @test isapprox(M, em, m)
                        em = similar(m)
                        @test apply!(A, em, ge, m) === em
                        @test isapprox(M, em, m)
                        em = similar(m)
                        @test inverse_apply!(A, em, e, m) === em
                        @test isapprox(M, em, m)
                        em = similar(m)
                        @test inverse_apply!(A, em, ge, m) === em
                        @test isapprox(M, em, m)
                    end
                end
            end
        end

        @testset "Inverse" begin
            for a in a_pts
                ainv = inv(G, a)
                @test isapprox(G, compose(A, a, ainv), e)
                @test isapprox(G, compose(A, ainv, a), e)
                @test isapprox(G, e, compose(A, a, ainv))
                @test isapprox(G, e, compose(A, ainv, a))

                for m in m_pts
                    @test isapprox(M, apply(A, a, m), inverse_apply(A, ainv, m))
                    @test isapprox(M, apply(A, ainv, m), inverse_apply(A, a, m))
                end
            end
        end
    end

    test_diff && @testset "apply differential" begin
        for (m, v) in zip(m_pts, v_pts)
            for a in a_pts
                am, av = apply(A, a, m), apply_diff(A, a, m, v)
                ainvm, ainvv = inverse_apply(A, a, m), inverse_apply_diff(A, a, m, v)
                @test is_tangent_vector(M, am, av, true; atol = atol)
                @test is_tangent_vector(M, ainvm, ainvv, true; atol = atol)
            end

            a12 = compose(A, a_pts[1], a_pts[2])
            a2m = apply(A, a_pts[2], m)
            a12v = apply_diff(A, a12, m, v)
            a2v = apply_diff(A, a_pts[2], m, v)
            @test apply_diff(A, a_pts[1], a2m, a2v) ≈ a12v

            @test apply_diff(A, e, m, v) ≈ v
            @test inverse_apply_diff(A, e, m, v) ≈ v
        end

        test_mutating && @testset "mutating" begin
            for (m, v) in zip(m_pts, v_pts)
                for a in a_pts
                    am = apply(A, a, m)
                    av = similar(v)
                    @test apply_diff!(A, av, a, m, v) === av
                    ainvm = inverse_apply(A, a, m)
                    ainvv = similar(v)
                    @test inverse_apply_diff!(A, ainvv, a, m, v) === ainvv
                    @test is_tangent_vector(M, am, av, true; atol = atol)
                    @test is_tangent_vector(M, ainvm, ainvv, true; atol = atol)
                end

                a12 = compose(A, a_pts[1], a_pts[2])
                a2m = apply(A, a_pts[2], m)
                a12v, a2v, a1_a2v = similar(v), similar(v), similar(v)
                @test apply_diff!(A, a12v, a12, m, v) === a12v
                @test apply_diff!(A, a2v, a_pts[2], m, v) === a2v
                @test apply_diff!(A, a1_a2v, a_pts[1], a2m, a2v) === a1_a2v
                @test a1_a2v ≈ a12v

                ev = similar(v)
                @test apply_diff!(A, ev, e, m, v) === ev
                @test ev ≈ v
                ev = similar(v)
                @test inverse_apply_diff!(A, ev, e, m, v) === ev
                @test ev ≈ v
            end
        end
    end

    test_optimal_alignment && @testset "Center of orbit" begin
        act = center_of_orbit(A, [m_pts[1]], m_pts[2])
        act_opt = optimal_alignment(A, m_pts[2], m_pts[1])
        @test isapprox(G, act, act_opt)
    end
end
