import Manifolds: is_decorator_manifold

struct NotImplementedOperation <: AbstractGroupOperation end

struct NotImplementedManifold <: Manifold end

struct NotImplementedGroupDecorator{M} <: Manifold
    manifold::M
end
is_decorator_manifold(::NotImplementedGroupDecorator) = Val(true)

"""
    test_group(
        G,
        g_pts::AbstractVector,
        v_pts::AbstractVector = [];
        atol = 1e-10,
        test_mutating = true,
        test_diff = false,
        test_invariance = false,
        diff_convs = [(), (LeftAction(),), (RightAction(),)],
    )

Tests general properties of the group `G`, given at least three different points
elements of it (contained in `g_pts`).
Optionally, specify `test_diff` to test differentials of translation, using `v_pts`, which
must contain at least one tangent vector at `g_pts[1]`, and the direction conventions
specified in `diff_convs`.
If the group is equipped with an invariant metric, `test_invariance` indicates that the
invariance should be checked for the provided points.
"""
function test_group(
    G,
    g_pts::AbstractVector,
    v_pts::AbstractVector = [],
    ve_pts::AbstractVector = [];
    atol = 1e-10,
    test_mutating = true,
    test_group_exp_log = true,
    test_diff = false,
    test_invariance = false,
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
                g12, g23, g12_3, g1_23 = allocate.(repeat([g_pts[1]], 4))
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
                    h = allocate(g)
                    @test compose!(G, h, g, e) === h
                    @test isapprox(G, h, g)
                    h = allocate(g)
                    @test compose!(G, h, e, g) === h
                    @test isapprox(G, h, g)

                    ge = allocate(g)
                    @test identity!(G, ge, e) === ge
                    @test isapprox(G, compose(G, g, ge), g)
                    @test isapprox(G, compose(G, ge, g), g)

                    ge = allocate(g)
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
                    ginv = allocate(g)
                    @test inv!(G, ginv, g) === ginv
                    @test isapprox(G, compose(G, g, ginv), e; atol = atol)
                    @test isapprox(G, compose(G, ginv, g), e; atol = atol)

                    @test inv(G, e) === e
                    geinv = allocate(g)
                    @test inv!(G, geinv, e) === geinv
                    @test isapprox(G, geinv, e; atol = atol)
                end
            end
        end
    end

    @testset "translation" begin
        convs = ((), (LeftAction(),), (RightAction(),))

        @test isapprox(G, translate(G, g_pts[1], g_pts[2]), compose(G, g_pts[1], g_pts[2]); atol = atol)
        @test isapprox(G, translate(G, g_pts[1], g_pts[2], LeftAction()), compose(G, g_pts[1], g_pts[2]); atol = atol)
        @test isapprox(G, translate(G, g_pts[1], g_pts[2], RightAction()), compose(G, g_pts[2], g_pts[1]); atol = atol)

        for conv in convs
            @test isapprox(G, inverse_translate(G, g_pts[1], translate(G, g_pts[1], g_pts[2], conv...), conv...), g_pts[2]; atol = atol)
            @test isapprox(G, translate(G, g_pts[1], inverse_translate(G, g_pts[1], g_pts[2], conv...), conv...), g_pts[2]; atol = atol)
        end

        test_mutating && @testset "mutating" begin
            for conv in convs
                g = allocate(g_pts[1])
                @test translate!(G, g, g_pts[1], g_pts[2], conv...) === g
                @test isapprox(G, g, translate(G, g_pts[1], g_pts[2], conv...); atol = atol)

                g = translate(G, g_pts[1], g_pts[2], conv...)
                g2 = allocate(g)
                @test inverse_translate!(G, g2, g_pts[1], g, conv...) === g2
                @test isapprox(G, g2, g_pts[2]; atol = atol)

                g = inverse_translate(G, g_pts[1], g_pts[2], conv...)
                g2 = allocate(g)
                @test translate!(G, g2, g_pts[1], g, conv...) === g2
                @test isapprox(G, g2, g_pts[2]; atol = atol)
            end
        end
    end

    test_diff && @testset "translation differential" begin
        X = v_pts[1]
        g21 = compose(G, g_pts[2], g_pts[1])
        g12 = compose(G, g_pts[1], g_pts[2])
        @test isapprox(G, g12, translate_diff(G, g_pts[2], g_pts[1], X), translate_diff(G, g_pts[2], g_pts[1], X, LeftAction()); atol = atol)
        @test is_tangent_vector(G, g12, translate_diff(G, g_pts[2], g_pts[1], X, LeftAction()), true; atol = atol)
        RightAction() in diff_convs && @test is_tangent_vector(G, g21, translate_diff(G, g_pts[2], g_pts[1], X, RightAction()), true; atol = atol)

        for conv in diff_convs
            g2g1 = translate(G, g_pts[2], g_pts[1], conv...)
            g2invg1 = inverse_translate(G, g_pts[2], g_pts[1], conv...)
            @test isapprox(G, g_pts[1], inverse_translate_diff(G, g_pts[2], g2g1, translate_diff(G, g_pts[2], g_pts[1], X, conv...), conv...), X; atol = atol)
            @test isapprox(G, g_pts[1], translate_diff(G, g_pts[2], g2invg1, inverse_translate_diff(G, g_pts[2], g_pts[1], X, conv...), conv...), X; atol = atol)
            Xe = inverse_translate_diff(G, g_pts[1], g_pts[1], X, conv...)
            @test isapprox(G, e, Xe, translate_diff(G, e, e, Xe, conv...); atol = atol)
            @test isapprox(G, e, Xe, inverse_translate_diff(G, e, e, Xe, conv...); atol = atol)
        end

        test_mutating && @testset "mutating" begin
            for conv in diff_convs
                g2g1 = translate(G, g_pts[2], g_pts[1], conv...)
                g2invg1 = inverse_translate(G, g_pts[2], g_pts[1], conv...)
                Y = allocate(X)
                @test translate_diff!(G, Y, g_pts[2], g_pts[1], X, conv...) === Y
                @test isapprox(G, g2g1, Y, translate_diff(G, g_pts[2], g_pts[1], X, conv...); atol = atol)

                Y = translate_diff(G, g_pts[2], g_pts[1], X, conv...)
                Z = allocate(Y)
                @test inverse_translate_diff!(G, Z, g_pts[2], g2g1, Y, conv...) === Z
                @test isapprox(G, g_pts[1], Z, X; atol = atol)

                Y = inverse_translate_diff(G, g_pts[2], g_pts[1], X, conv...)
                Z = allocate(Y)
                @test translate_diff!(G, Z, g_pts[2], g2invg1, Y, conv...) === Z
                @test isapprox(G, g_pts[1], Z, X; atol = atol)
            end
        end
    end

    test_group_exp_log && @testset "group exp/log properties" begin
        @testset "e = exp(0)" begin
            v = group_log(G, identity(G, g_pts[1]))
            g = group_exp(G, v)
            @test isapprox(G, Identity(G), g; atol = atol)

            test_mutating && @testset "mutating" begin
                v = allocate(ve_pts[1])
                @test group_log!(G, v, identity(G, g_pts[1])) === v
                g = allocate(g_pts[1])
                @test group_exp!(G, g, v) === g
                @test isapprox(G, Identity(G), g; atol = atol)
            end
        end

        @testset "v = log(exp(v))" begin
            for v in ve_pts
                g = group_exp(G, v)
                @test is_manifold_point(G, g; atol = atol)
                v2 = group_log(G, g)
                @test isapprox(G, Identity(G), v2, v; atol = atol)
            end

            test_mutating && @testset "mutating" begin
                for v in ve_pts
                    g = allocate(g_pts[1])
                    @test group_exp!(G, g, v) === g
                    @test is_manifold_point(G, g; atol = atol)
                    @test isapprox(G, g, group_exp(G, v); atol = atol)
                    v2 = allocate(v)
                    @test group_log!(G, v2, g) === v2
                    @test isapprox(G, Identity(G), v2, v; atol = atol)
                end
            end
        end

        @testset "inv(g) = exp(-log(g))" begin
            g = g_pts[1]
            v = group_log(G, g)
            ginv = group_exp(G, -v)
            @test isapprox(G, ginv, inv(G, g); atol = atol)
        end

        @testset "exp(sv)∘exp(tv) = exp((s+t)v)" begin
            g1 = group_exp(G, 0.2 * ve_pts[1])
            g2 = group_exp(G, 0.3 * ve_pts[1])
            g12 = group_exp(G, 0.5 * ve_pts[1])
            g1_g2 = compose(G, g1, g2)
            g2_g1 = compose(G, g2, g1)
            isapprox(G, g1_g2, g12; atol = atol)
            isapprox(G, g2_g1, g12; atol = atol)
        end
    end

    test_group_exp_log && test_diff && @testset "exp/log retract/inverse_retract" begin
        for conv in diff_convs
            y = retract(G, g_pts[1], v_pts[1], Manifolds.GroupExponentialRetraction(conv...))
            @test is_manifold_point(G, y; atol = atol)
            v2 = inverse_retract(G, g_pts[1], y, Manifolds.GroupLogarithmicInverseRetraction(conv...))
            @test isapprox(G, g_pts[1], v2, v_pts[1]; atol = atol)

            if has_biinvariant_metric(G) === Val(true)
                @test isapprox(G, exp(G, g_pts[1], v_pts[1]), y; atol = atol)
                @test isapprox(G, g_pts[1], log(G, g_pts[1], y), v2; atol = atol)
            end
        end

        test_mutating && @testset "mutating" begin
            for conv in diff_convs
                y = allocate(g_pts[1])
                @test retract!(G, y, g_pts[1], v_pts[1], Manifolds.GroupExponentialRetraction(conv...)) === y
                @test is_manifold_point(G, y; atol = atol)
                v2 = allocate(v_pts[1])
                @test inverse_retract!(G, v2, g_pts[1], y, Manifolds.GroupLogarithmicInverseRetraction(conv...)) === v2
                @test isapprox(G, g_pts[1], v2, v_pts[1]; atol = atol)
            end
        end
    end

    test_invariance && @testset "metric invariance" begin
        if has_invariant_metric(G, LeftAction()) === Val(true)
            @testset "left-invariant" begin
                @test check_has_invariant_metric(
                    G,
                    g_pts[1],
                    v_pts[1],
                    v_pts[end],
                    g_pts,
                    LeftAction(),
                )
            end
        end
        if has_invariant_metric(G, RightAction()) === Val(true)
            @testset "right-invariant" begin
                @test check_has_invariant_metric(
                    G,
                    g_pts[1],
                    v_pts[1],
                    v_pts[end],
                    g_pts,
                    RightAction(),
                )
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
            Aswitch = switch_direction(A)
            if isa(A, AbstractGroupAction{LeftAction})
                @test direction(A) === LeftAction()
                @test isa(Aswitch, AbstractGroupAction{RightAction})
                @test direction(Aswitch) === RightAction()
            else
                @test direction(A) === RightAction()
                @test isa(Aswitch, AbstractGroupAction{LeftAction})
                @test direction(Aswitch) === LeftAction()
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
                a12, a23, a12_3, a1_23 = allocate.(repeat([a_pts[1]], 4))
                @test compose!(A, a12, a_pts[1], a_pts[2]) === a12
                @test compose!(A, a23, a_pts[2], a_pts[3]) === a23
                @test compose!(A, a12_3, a12, a_pts[3]) === a12_3
                @test compose!(A, a1_23, a_pts[1], a23) === a1_23
                @test isapprox(G, a12_3, a1_23)

                for m in m_pts
                    a12_a3_m, a1_a23_m = allocate(m), allocate(m)
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
                    h = allocate(a)
                    @test compose!(A, h, a, e) === h
                    @test isapprox(G, h, a)
                    h = allocate(a)
                    @test compose!(A, h, e, a) === h
                    @test isapprox(G, h, a)

                    ge = identity(G, a)
                    @test isapprox(G, compose(A, a, ge), a)
                    @test isapprox(G, compose(A, ge, a), a)

                    ge = identity(G, a)
                    @test compose!(A, ge, e, e) === ge
                    @test isapprox(G, ge, e)

                    for m in m_pts
                        em = allocate(m)
                        @test apply!(A, em, e, m) === em
                        @test isapprox(M, em, m)
                        em = allocate(m)
                        @test apply!(A, em, ge, m) === em
                        @test isapprox(M, em, m)
                        em = allocate(m)
                        @test inverse_apply!(A, em, e, m) === em
                        @test isapprox(M, em, m)
                        em = allocate(m)
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
            @test isapprox(M, a2m, apply_diff(A, a_pts[1], a2m, a2v), a12v; atol = atol)

            @test isapprox(M, m, apply_diff(A, e, m, v), v; atol = atol)
            @test isapprox(M, m, inverse_apply_diff(A, e, m, v), v; atol = atol)
        end

        test_mutating && @testset "mutating" begin
            for (m, v) in zip(m_pts, v_pts)
                for a in a_pts
                    am = apply(A, a, m)
                    av = allocate(v)
                    @test apply_diff!(A, av, a, m, v) === av
                    ainvm = inverse_apply(A, a, m)
                    ainvv = allocate(v)
                    @test inverse_apply_diff!(A, ainvv, a, m, v) === ainvv
                    @test is_tangent_vector(M, am, av, true; atol = atol)
                    @test is_tangent_vector(M, ainvm, ainvv, true; atol = atol)
                end

                a12 = compose(A, a_pts[1], a_pts[2])
                a2m = apply(A, a_pts[2], m)
                a12m = apply(A, a12, m)
                a12v, a2v, a1_a2v = allocate(v), allocate(v), allocate(v)
                @test apply_diff!(A, a12v, a12, m, v) === a12v
                @test apply_diff!(A, a2v, a_pts[2], m, v) === a2v
                @test apply_diff!(A, a1_a2v, a_pts[1], a2m, a2v) === a1_a2v
                @test isapprox(M, a12m, a1_a2v, a12v; atol = atol)

                ev = allocate(v)
                @test apply_diff!(A, ev, e, m, v) === ev
                @test isapprox(G, m, ev, v; atol = atol)
                ev = allocate(v)
                @test inverse_apply_diff!(A, ev, e, m, v) === ev
                @test isapprox(G, m, ev, v; atol = atol)
            end
        end
    end

    test_optimal_alignment && @testset "Center of orbit" begin
        act = center_of_orbit(A, [m_pts[1]], m_pts[2])
        act2 = center_of_orbit(A, [m_pts[1]], m_pts[2], GradientDescentEstimation())
        act_opt = optimal_alignment(A, m_pts[2], m_pts[1])
        @test isapprox(G, act, act_opt; atol = atol)
        @test isapprox(G, act2, act_opt; atol = atol)

        test_mutating && @testset "mutating" begin
            act_opt2 = allocate(act_opt)
            optimal_alignment!(A, act_opt2, m_pts[2], m_pts[1])
            @test isapprox(G, act_opt, act_opt2; atol = atol)
        end
    end
end
