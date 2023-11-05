using Base: IdentityUnitRange

using Manifolds:
    LeftForwardAction, LeftBackwardAction, RightForwardAction, RightBackwardAction

translate_diff_id(G, q, X, conv) = translate_diff(G, q, identity_element(G), X, conv)

"""
    _move(G::AbstractManifold, q, X, ::LeftSide)

The left translation ``τ_L(g, X) = gX``.
"""
_move(G::AbstractManifold, q, X, ::LeftSide) =
    translate_diff_id(G, q, X, (LeftAction(), LeftSide()))
"""
    _move(G::AbstractManifold, q, X, ::RightSide)

The right translation ``τ_R(g, X) = Xg``.
"""
_move(G::AbstractManifold, q, X, ::RightSide) =
    translate_diff_id(G, q, X, (RightAction(), RightSide()))

"""
    test_inv_diff_fn(G::AbstractManifold, q, X, conv)

Test the differential of the inverse on a Lie group.
Denote this inverse by ``I(g) := g^{-1}``.
If the left and right transports are ``τ_L(g,X) := gX``
and ``τ_R(g,X) := Xg`` respectively, then
```math
⟨DI(g), τ_L(g,X)⟩ = -τ_R(g^{-1}, X)
```
and
``` math
⟨DI(g), τ_R(g,X)⟩ = -τ_L(g^{-1}, X)
```
"""
function test_inv_diff_fn(G::AbstractManifold, q, X, conv)
    q_ = inv(G, q)
    Test.@test isapprox(
        TangentSpace(G, q_),
        inv_diff(G, q, _move(G, q, X, conv)),
        -_move(G, q_, X, switch_side(conv)),
    )
end

_get_side(::LeftAction) = RightSide()
_get_side(::RightAction) = LeftSide()

_transporter(G::AbstractManifold, q, X, dir) = _move(G, q, X, _get_side(dir))

@doc raw"""
    test_apply_diff_group_fn(A::AbstractGroupAction{TAD}, q, X, p) where {TAD}

This should hold for *any* group action ``A`` on any manifold.
If you define ``π_p(g) := A(g, p)`` for ``g ∈ \mathcal G`` and ``p ∈ \mathcal M``,
and define, for ``X in Alg(G)``,
    ``τ_R(g, X) := Xg`` (the right translation),
and ``τ_L(g, X) := gX`` (the left translation), then we have the identity:
```math
⟨Dπ_{q}(g), τ(g, X)⟩ = ⟨Dπ_{A(g, q)}(1), X⟩
```
where, for a *left* action, ``τ`` is the *right* translation,
and for a *right* action, ``τ`` is the *left* translation.
"""
function test_apply_diff_group_fn(A::AbstractGroupAction{TAD}, q, X, p) where {TAD}
    G = base_group(A)
    p_ = apply(A, q, p)
    X1 = apply_diff_group(A, q, _transporter(G, q, X, TAD()), p)
    X2 = apply_diff_group(A, identity_element(G), X, p_)
    Test.@test isapprox(TangentSpace(G, p_), X1, X2)
end

"""
    test_group(
        G,
        g_pts::AbstractVector,
        X_pts::AbstractVector=[],
        Xe_pts::AbstractVector=[];
        atol::Real=1e-10,
        test_mutating::Bool=true,
        test_exp_lie_log::Bool=true,
        test_diff::Bool=false,
        test_invariance::Bool=false,
        test_lie_bracket::Bool=false,
        test_adjoint_action::Bool=false,
        test_inv_diff::Bool=false,
        test_apply_diff_group::Bool=false,
        diff_convs = [(), (LeftForwardAction(),), (RightBackwardAction(),)],
    )

Tests general properties of the group `G`, given at least three different points
elements of it (contained in `g_pts`).
Optionally, specify `test_diff` to test differentials of translation, using `X_pts`, which
must contain at least one tangent vector at `g_pts[1]`, and the direction conventions
specified in `diff_convs`.
`Xe_pts` should contain tangent vectors at identity for testing Lie algebra operations.
If the group is equipped with an invariant metric, `test_invariance` indicates that the
invariance should be checked for the provided points.
"""
function test_group(
    G,
    g_pts::AbstractVector,
    X_pts::AbstractVector=[],
    Xe_pts::AbstractVector=[];
    atol::Real=1e-10,
    test_mutating::Bool=true,
    test_exp_lie_log::Bool=true,
    test_one_arg_identity_element::Bool=true,
    test_diff::Bool=false,
    test_invariance::Bool=false,
    test_lie_bracket::Bool=false,
    test_adjoint_action::Bool=false,
    test_inv_diff::Bool=false,
    diff_convs=[(), (LeftForwardAction(),), (RightBackwardAction(),)],
    test_log_from_identity::Bool=false,
    test_exp_from_identity::Bool=false,
    test_vee_hat_from_identity::Bool=false,
    test_apply_diff_group::Bool=false,
)
    e = Identity(G)

    Test.@testset "Basic group properties" begin # COV_EXCL_LINE
        Test.@testset "Closed" begin
            for g1 in g_pts, g2 in g_pts
                g3 = compose(G, g1, g2)
                Test.@test is_point(G, g3, true; atol=atol)
            end
        end

        Test.@testset "Associative" begin # COV_EXCL_LINE
            g12_3 = compose(G, compose(G, g_pts[1], g_pts[2]), g_pts[3])
            g1_23 = compose(G, g_pts[1], compose(G, g_pts[2], g_pts[3]))
            Test.@test isapprox(G, g12_3, g1_23; atol=atol)

            test_mutating && Test.@testset "mutating" begin
                g12, g23, g12_3, g1_23 = allocate.(repeat([g_pts[1]], 4))
                Test.@test compose!(G, g12, g_pts[1], g_pts[2]) === g12
                Test.@test compose!(G, g23, g_pts[2], g_pts[3]) === g23
                Test.@test compose!(G, g12_3, g12, g_pts[3]) === g12_3
                Test.@test compose!(G, g1_23, g_pts[1], g23) === g1_23
                Test.@test isapprox(G, g12_3, g1_23; atol=atol)
            end
        end

        Test.@testset "Identity" begin # COV_EXCL_LINE
            Test.@test is_point(G, e)
            wrong_e = if e === Identity(MultiplicationOperation())
                Identity(AdditionOperation())
            else
                Identity(MultiplicationOperation())
            end
            Test.@test !is_point(G, wrong_e)
            Test.@test !is_identity(G, wrong_e)
            Test.@test isapprox(G, e, e)
            Test.@test compose(G, e, e) === e
            Test.@test copyto!(G, e, e) === e

            ge = identity_element(G, g_pts[1])
            test_one_arg_identity_element && Test.@test is_identity(G, identity_element(G))
            for g in g_pts
                Test.@test isapprox(G, compose(G, g, e), g)
                Test.@test isapprox(G, compose(G, e, g), g)

                Test.@test isapprox(G, compose(G, g, ge), g)
                Test.@test isapprox(G, compose(G, ge, g), g)
            end

            test_mutating && Test.@testset "mutating" begin
                for g in g_pts
                    h = allocate(g)
                    Test.@test compose!(G, h, g, e) === h
                    Test.@test isapprox(G, h, g)
                    h = allocate(g)
                    Test.@test compose!(G, h, e, g) === h
                    Test.@test isapprox(G, h, g)

                    ge = Identity(G)
                    Test.@test isapprox(G, compose(G, g, e), g)
                    Test.@test isapprox(G, compose(G, e, g), g)

                    Test.@test compose!(G, ge, e, e) === ge
                    Test.@test isapprox(G, ge, e)
                end
            end
        end

        Test.@testset "Inverse" begin # COV_EXCL_LINE
            Test.@test inv(G, e) === e
            for g in g_pts
                ginv = inv(G, g)
                Test.@test is_identity(G, compose(G, g, ginv); atol=atol)
                Test.@test is_identity(G, compose(G, ginv, g); atol=atol)

                test_mutating && Test.@testset "mutating" begin
                    Test.@test inv!(G, e, e) === e
                    ginv = allocate(g)
                    Test.@test inv!(G, ginv, g) === ginv
                    Test.@test is_identity(G, compose(G, g, ginv); atol=atol)
                    Test.@test is_identity(G, compose(G, ginv, g); atol=atol)

                    Test.@test inv(G, e) === e
                    geinv = allocate(g)
                    Test.@test inv!(G, geinv, e) === geinv
                    Test.@test is_identity(G, geinv; atol=atol)
                end
            end
        end
    end

    Test.@testset "translation" begin # COV_EXCL_LINE
        convs = ((), (LeftForwardAction(),), (RightBackwardAction(),))

        Test.@test isapprox(
            G,
            translate(G, g_pts[1], g_pts[2]),
            compose(G, g_pts[1], g_pts[2]);
            atol=atol,
        )
        Test.@test isapprox(
            G,
            translate(G, g_pts[1], g_pts[2], LeftForwardAction()),
            compose(G, g_pts[1], g_pts[2]);
            atol=atol,
        )
        Test.@test isapprox(
            G,
            translate(G, g_pts[1], g_pts[2], RightBackwardAction()),
            compose(G, g_pts[2], g_pts[1]);
            atol=atol,
        )

        for conv in convs
            Test.@test isapprox(
                G,
                inverse_translate(
                    G,
                    g_pts[1],
                    translate(G, g_pts[1], g_pts[2], conv...),
                    conv...,
                ),
                g_pts[2];
                atol=atol,
            )
            Test.@test isapprox(
                G,
                translate(
                    G,
                    g_pts[1],
                    inverse_translate(G, g_pts[1], g_pts[2], conv...),
                    conv...,
                ),
                g_pts[2];
                atol=atol,
            )
        end

        test_mutating && Test.@testset "mutating" begin
            for conv in convs
                g = allocate(g_pts[1])
                Test.@test translate!(G, g, g_pts[1], g_pts[2], conv...) === g
                Test.@test isapprox(
                    G,
                    g,
                    translate(G, g_pts[1], g_pts[2], conv...);
                    atol=atol,
                )

                g = translate(G, g_pts[1], g_pts[2], conv...)
                g2 = allocate(g)
                Test.@test inverse_translate!(G, g2, g_pts[1], g, conv...) === g2
                Test.@test isapprox(G, g2, g_pts[2]; atol=atol)

                g = inverse_translate(G, g_pts[1], g_pts[2], conv...)
                g2 = allocate(g)
                Test.@test translate!(G, g2, g_pts[1], g, conv...) === g2
                Test.@test isapprox(G, g2, g_pts[2]; atol=atol)
            end
        end
    end

    test_diff && Test.@testset "translation differential" begin
        X = X_pts[1]
        g21 = compose(G, g_pts[2], g_pts[1])
        g12 = compose(G, g_pts[1], g_pts[2])
        Test.@test isapprox(
            G,
            g12,
            translate_diff(G, g_pts[2], g_pts[1], X),
            translate_diff(G, g_pts[2], g_pts[1], X, LeftForwardAction());
            atol=atol,
        )
        Test.@test is_vector(
            G,
            g12,
            translate_diff(G, g_pts[2], g_pts[1], X, LeftForwardAction()),
            true;
            atol=atol,
        )
        RightBackwardAction() in diff_convs && Test.@test is_vector(
            G,
            g21,
            translate_diff(G, g_pts[2], g_pts[1], X, RightBackwardAction()),
            true;
            atol=atol,
        )

        for conv in diff_convs
            g2g1 = translate(G, g_pts[2], g_pts[1], conv...)
            g2invg1 = inverse_translate(G, g_pts[2], g_pts[1], conv...)
            Test.@test isapprox(
                G,
                g_pts[1],
                inverse_translate_diff(
                    G,
                    g_pts[2],
                    g2g1,
                    translate_diff(G, g_pts[2], g_pts[1], X, conv...),
                    conv...,
                ),
                X;
                atol=atol,
            )
            Test.@test isapprox(
                G,
                g_pts[1],
                translate_diff(
                    G,
                    g_pts[2],
                    g2invg1,
                    inverse_translate_diff(G, g_pts[2], g_pts[1], X, conv...),
                    conv...,
                ),
                X;
                atol=atol,
            )
            Xe = inverse_translate_diff(G, g_pts[1], g_pts[1], X, conv...)
            Test.@test isapprox(G, e, Xe, translate_diff(G, e, e, Xe, conv...); atol=atol)
            Test.@test isapprox(
                G,
                e,
                Xe,
                inverse_translate_diff(G, e, e, Xe, conv...);
                atol=atol,
            )
        end

        test_mutating && Test.@testset "mutating" begin
            for conv in diff_convs
                g2g1 = translate(G, g_pts[2], g_pts[1], conv...)
                g2invg1 = inverse_translate(G, g_pts[2], g_pts[1], conv...)
                Y = allocate(X)
                Test.@test translate_diff!(G, Y, g_pts[2], g_pts[1], X, conv...) === Y
                Test.@test isapprox(
                    G,
                    g2g1,
                    Y,
                    translate_diff(G, g_pts[2], g_pts[1], X, conv...);
                    atol=atol,
                )

                Y = translate_diff(G, g_pts[2], g_pts[1], X, conv...)
                Z = allocate(Y)
                Test.@test inverse_translate_diff!(G, Z, g_pts[2], g2g1, Y, conv...) === Z
                Test.@test isapprox(G, g_pts[1], Z, X; atol=atol)

                Y = inverse_translate_diff(G, g_pts[2], g_pts[1], X, conv...)
                Z = allocate(Y)
                Test.@test translate_diff!(G, Z, g_pts[2], g2invg1, Y, conv...) === Z
                Test.@test isapprox(G, g_pts[1], Z, X; atol=atol)
            end
        end
    end

    test_inv_diff && Test.@testset "Differential of inverse" begin # COV_EXCL_LINE
        Test.@test isapprox(inv_diff(G, e, Xe_pts[1]), -Xe_pts[1]; atol=atol)
        Test.@test isapprox(
            inv_diff(G, inv(G, g_pts[1]), inv_diff(G, g_pts[1], X_pts[1])),
            X_pts[1];
            atol=atol,
        )
        Test.@testset "test_inv_diff" for side in [LeftSide(), RightSide()]
            test_inv_diff_fn(G, g_pts[1], X_pts[1], side)
        end
    end

    test_exp_lie_log && Test.@testset "group exp/log properties" begin
        Test.@testset "e = exp(0)" begin
            X = log_lie(G, Identity(G))
            g = exp_lie(G, X)
            Test.@test isapprox(G, Identity(G), g; atol=atol)
            ep = identity_element(G, g)
            Test.@test isapprox(G, ep, zero_vector(G, ep), log_lie(G, ep); atol=atol)

            test_mutating && Test.@testset "mutating" begin
                X = allocate(Xe_pts[1])
                Test.@test log_lie!(G, X, Identity(G)) === X
                g = allocate(g_pts[1])
                Test.@test exp_lie!(G, g, X) === g
                Test.@test is_identity(G, g; atol=atol) || "is_identity($G, $g; atol=$atol)"
            end
        end

        Test.@testset "X = log(exp(X))" begin
            for X in Xe_pts
                g = exp_lie(G, X)
                Test.@test is_point(G, g; atol=atol)
                X2 = log_lie(G, g)
                Test.@test isapprox(G, Identity(G), X2, X; atol=atol)
            end

            test_mutating && Test.@testset "mutating" begin
                for X in Xe_pts
                    g = allocate(g_pts[1])
                    Test.@test exp_lie!(G, g, X) === g
                    Test.@test is_point(G, g, true; atol=atol)
                    Test.@test isapprox(G, g, exp_lie(G, X); atol=atol)
                    X2 = allocate(X)
                    Test.@test log_lie!(G, X2, g) === X2
                    Test.@test isapprox(G, Identity(G), X2, X; atol=atol)
                end
            end
        end

        Test.@testset "inv(g) = exp(-log(g))" begin # COV_EXCL_LINE
            g = g_pts[1]
            X = log_lie(G, g)
            ginv = exp_lie(G, -X)
            Test.@test isapprox(G, ginv, inv(G, g); atol=atol)
        end

        Test.@testset "exp(sX)∘exp(tX) = exp((s+t)X)" begin # COV_EXCL_LINE
            g1 = exp_lie(G, 0.2 * Xe_pts[1])
            g2 = exp_lie(G, 0.3 * Xe_pts[1])
            g12 = exp_lie(G, 0.5 * Xe_pts[1])
            g1_g2 = compose(G, g1, g2)
            g2_g1 = compose(G, g2, g1)
            isapprox(G, g1_g2, g12; atol=atol)
            isapprox(G, g2_g1, g12; atol=atol)
        end
    end

    test_exp_lie_log &&
        test_diff &&
        Test.@testset "exp/log retract/inverse_retract" begin # COV_EXCL_LINE
            for conv in diff_convs
                y = retract(
                    G,
                    g_pts[1],
                    X_pts[1],
                    Manifolds.GroupExponentialRetraction(conv...),
                )
                Test.@test is_point(G, y, true; atol=atol)
                X2 = inverse_retract(
                    G,
                    g_pts[1],
                    y,
                    Manifolds.GroupLogarithmicInverseRetraction(conv...),
                )
                Test.@test isapprox(G, g_pts[1], X2, X_pts[1]; atol=atol)
            end

            test_mutating && Test.@testset "mutating" begin
                for conv in diff_convs
                    y = allocate(g_pts[1])
                    Test.@test retract!(
                        G,
                        y,
                        g_pts[1],
                        X_pts[1],
                        Manifolds.GroupExponentialRetraction(conv...),
                    ) === y
                    Test.@test is_point(G, y, true; atol=atol)
                    X2 = allocate(X_pts[1])
                    Test.@test inverse_retract!(
                        G,
                        X2,
                        g_pts[1],
                        y,
                        Manifolds.GroupLogarithmicInverseRetraction(conv...),
                    ) === X2
                    Test.@test isapprox(G, g_pts[1], X2, X_pts[1]; atol=atol)
                end
            end
        end

    test_invariance && Test.@testset "metric invariance" begin
        if has_invariant_metric(G, LeftForwardAction())
            Test.@testset "left-invariant" begin # COV_EXCL_LINE
                Test.@test has_approx_invariant_metric(
                    G,
                    g_pts[1],
                    X_pts[1],
                    X_pts[end],
                    g_pts,
                    LeftForwardAction(),
                )
            end
        end
        if has_invariant_metric(G, RightBackwardAction())
            Test.@testset "right-invariant" begin # COV_EXCL_LINE
                Test.@test has_approx_invariant_metric(
                    G,
                    g_pts[1],
                    X_pts[1],
                    X_pts[end],
                    g_pts,
                    RightBackwardAction(),
                )
            end
        end
    end

    test_adjoint_action && Test.@testset "Adjoint action" begin
        # linearity
        X = Xe_pts[1]
        Y = Xe_pts[2]
        e = Identity(G)
        Test.@test isapprox(
            G,
            e,
            adjoint_action(G, g_pts[2], X + Y),
            adjoint_action(G, g_pts[2], X) + adjoint_action(G, g_pts[2], Y),
        )
        # inverse property
        Test.@test isapprox(
            G,
            e,
            adjoint_action(G, g_pts[2], adjoint_action(G, inv(G, g_pts[2]), X)),
            X,
        )
        if test_mutating
            Z = allocate(X)
            adjoint_action!(G, Z, g_pts[2], X)
            Test.@test isapprox(G, e, Z, adjoint_action(G, g_pts[2], X))
        end

        # interaction with Lie bracket
        if test_lie_bracket
            Test.@test isapprox(
                G,
                e,
                adjoint_action(G, g_pts[2], lie_bracket(G, X, Y)),
                lie_bracket(
                    G,
                    adjoint_action(G, g_pts[2], X),
                    adjoint_action(G, g_pts[2], Y),
                ),
            )
        end
    end

    test_lie_bracket && Test.@testset "Lie bracket" begin
        # anticommutativity
        X = Xe_pts[1]
        Y = Xe_pts[2]
        e = Identity(G)
        Test.@test isapprox(G, e, lie_bracket(G, X, Y), -lie_bracket(G, Y, X))

        if test_mutating
            Z = allocate(X)
            lie_bracket!(G, Z, X, Y)
            Test.@test isapprox(G, e, Z, lie_bracket(G, X, Y))
        end
    end

    Test.@testset "Metric operations with Identity" begin # COV_EXCL_LINE
        if test_log_from_identity
            pe = identity_element(G)
            Test.@test isapprox(G, pe, log(G, e, g_pts[1]), log(G, pe, g_pts[1]))
            if test_mutating
                X = zero_vector(G, pe)
                log!(G, X, e, g_pts[1])
                Test.@test isapprox(G, pe, X, log(G, pe, g_pts[1]))
            end
        end
        if test_exp_from_identity
            pe = identity_element(G)
            Test.@test isapprox(G, pe, exp(G, e, Xe_pts[1]), exp(G, pe, Xe_pts[1]))
            if test_mutating
                q = allocate(G, Xe_pts[1])
                exp!(G, q, e, Xe_pts[1])
                Test.@test isapprox(G, q, exp(G, pe, Xe_pts[1]))
            end
        end
        if test_vee_hat_from_identity
            pe = identity_element(G)
            coeffs = vee(G, pe, Xe_pts[1])
            Test.@test isapprox(coeffs, vee(G, e, Xe_pts[1]))
            Test.@test isapprox(G, pe, Xe_pts[1], hat(G, e, coeffs))

            if test_mutating
                coeffs2 = similar(coeffs)
                vee!(G, coeffs2, e, Xe_pts[1])
                Test.@test isapprox(coeffs, coeffs2)

                X = allocate(G, Xe_pts[1])
                hat!(G, X, e, coeffs2)
                Test.@test isapprox(G, pe, Xe_pts[1], X)
            end
        end
    end

    test_apply_diff_group && Test.@testset "apply_diff_group" begin
        Test.@testset "$dir, $side" for side in [LeftSide(), RightSide()],
            dir in [LeftAction(), RightAction()]

            test_apply_diff_group_fn(
                GroupOperationAction(G, (dir, side)),
                g_pts[1],
                X_pts[1],
                g_pts[2],
            )
        end
    end

    return nothing
end

_direction_from_type(::AbstractGroupAction{TD}) where {TD<:ActionDirection} = TD()

"""
    test_action(
        A::AbstractGroupAction,
        a_pts::AbstractVector,
        m_pts::AbstractVector,
        X_pts = [];
        atol = 1e-10,
        atol_ident_compose = 0,
        test_optimal_alignment = false,
        test_mutating_group=true,
        test_mutating_action=true,
        test_diff = false,
        test_switch_direction = true,
    )

Tests general properties of the action `A`, given at least three different points
that lie on it (contained in `a_pts`) and three different point that lie
on the manifold it acts upon (contained in `m_pts`).

# Arguments
- `atol_ident_compose = 0`: absolute tolerance for the test that composition with identity
  doesn't change the group element.
"""
function test_action(
    A::AbstractGroupAction,
    a_pts::AbstractVector,
    m_pts::AbstractVector,
    X_pts=[];
    atol=1e-10,
    atol_ident_compose=0,
    test_optimal_alignment=false,
    test_mutating_group=true,
    test_mutating_action=true,
    test_diff=false,
    test_switch_direction=true,
)
    G = base_group(A)
    M = group_manifold(A)
    e = Identity(G)

    Test.@testset "Basic action properties" begin # COV_EXCL_LINE
        test_switch_direction && Test.@testset "Direction" begin
            Aswitch = switch_direction(A)

            Test.@test direction(A) === _direction_from_type(A)
            sd = switch_direction(_direction_from_type(A))
            Test.@test isa(Aswitch, AbstractGroupAction{typeof(sd)})
            Test.@test direction(Aswitch) === sd
        end
        Test.@testset "Closed" begin # COV_EXCL_LINE
            Test.@testset "over actions" begin # COV_EXCL_LINE
                for a1 in a_pts, a2 in a_pts
                    a3 = compose(A, a1, a2)
                    Test.@test is_point(G, a3, true; atol=atol)
                end
            end
            Test.@testset "over g-manifold" begin # COV_EXCL_LINE
                for a in a_pts, m in m_pts
                    Test.@test is_point(M, apply(A, a, m), true; atol=atol)
                    Test.@test is_point(M, inverse_apply(A, a, m), true; atol=atol)
                end
            end
        end

        Test.@testset "Associative" begin # COV_EXCL_LINE
            a12 = compose(A, a_pts[1], a_pts[2])
            a23 = compose(A, a_pts[2], a_pts[3])

            Test.@testset "over compose" begin # COV_EXCL_LINE
                a12_a3 = compose(A, a12, a_pts[3])
                a1_a23 = compose(A, a_pts[1], a23)
                Test.@test isapprox(G, a12_a3, a1_a23; atol=atol)
            end

            Test.@testset "over apply" begin # COV_EXCL_LINE
                for m in m_pts
                    a12_a3_m = apply(A, a12, apply(A, a_pts[3], m))
                    a1_a23_m = apply(A, a_pts[1], apply(A, a23, m))
                    Test.@test isapprox(M, a12_a3_m, a1_a23_m; atol=atol)
                end
            end

            test_mutating_group && Test.@testset "mutating group composition" begin
                a12, a23, a12_3, a1_23 = allocate.(repeat([a_pts[1]], 4))
                Test.@test compose!(A, a12, a_pts[1], a_pts[2]) === a12
                Test.@test compose!(A, a23, a_pts[2], a_pts[3]) === a23
                Test.@test compose!(A, a12_3, a12, a_pts[3]) === a12_3
                Test.@test compose!(A, a1_23, a_pts[1], a23) === a1_23
                Test.@test isapprox(G, a12_3, a1_23; atol=atol)
            end
            test_mutating_action && Test.@testset "mutating apply!" begin
                for m in m_pts
                    a12_a3_m, a1_a23_m = allocate(m), allocate(m)
                    Test.@test apply!(A, a12_a3_m, a12, apply(A, a_pts[3], m)) === a12_a3_m
                    Test.@test apply!(A, a1_a23_m, a_pts[1], apply(A, a23, m)) === a1_a23_m
                    Test.@test isapprox(M, a12_a3_m, a1_a23_m; atol=atol)
                end
            end
        end

        Test.@testset "Identity" begin # COV_EXCL_LINE
            Test.@test compose(A, e, e) === e

            for a in a_pts
                Test.@test isapprox(G, compose(A, a, e), a; atol=atol_ident_compose)
                Test.@test isapprox(G, compose(A, e, a), a; atol=atol_ident_compose)

                ge = identity_element(G, a)
                Test.@test isapprox(G, compose(A, a, ge), a; atol=atol_ident_compose)
                Test.@test isapprox(G, compose(A, ge, a), a; atol=atol_ident_compose)

                for m in m_pts
                    Test.@test isapprox(M, apply(A, e, m), m)
                    Test.@test isapprox(M, apply(A, ge, m), m)
                    Test.@test isapprox(M, inverse_apply(A, e, m), m)
                    Test.@test isapprox(M, inverse_apply(A, ge, m), m)
                end
            end

            test_mutating_group && Test.@testset "mutating" begin
                for a in a_pts
                    h = allocate(a)
                    Test.@test compose!(A, h, a, e) === h
                    Test.@test isapprox(G, h, a)
                    h = allocate(a)
                    Test.@test compose!(A, h, e, a) === h
                    Test.@test isapprox(G, h, a)

                    ge = identity_element(G, a)
                    Test.@test isapprox(G, compose(A, a, ge), a)
                    Test.@test isapprox(G, compose(A, ge, a), a)

                    ge = allocate(a)
                    Test.@test compose!(A, ge, e, e) === ge
                    Test.@test isapprox(G, ge, e)

                    for m in m_pts
                        em = allocate(m)
                        Test.@test apply!(A, em, e, m) === em
                        Test.@test isapprox(M, em, m)
                        em = allocate(m)
                        Test.@test apply!(A, em, ge, m) === em
                        Test.@test isapprox(M, em, m)
                        em = allocate(m)
                        Test.@test inverse_apply!(A, em, e, m) === em
                        Test.@test isapprox(M, em, m)
                        em = allocate(m)
                        Test.@test inverse_apply!(A, em, ge, m) === em
                        Test.@test isapprox(M, em, m)
                    end
                end
            end
        end

        Test.@testset "Inverse" begin # COV_EXCL_LINE
            for a in a_pts
                ainv = inv(G, a)
                Test.@test isapprox(G, compose(A, a, ainv), e; atol=atol)
                Test.@test isapprox(G, compose(A, ainv, a), e; atol=atol)
                Test.@test isapprox(G, e, compose(A, a, ainv); atol=atol)
                Test.@test isapprox(G, e, compose(A, ainv, a); atol=atol)

                for m in m_pts
                    Test.@test isapprox(
                        M,
                        apply(A, a, m),
                        inverse_apply(A, ainv, m);
                        atol=atol,
                    )
                    Test.@test isapprox(
                        M,
                        apply(A, ainv, m),
                        inverse_apply(A, a, m);
                        atol=atol,
                    )
                end
            end
        end
    end

    test_diff && Test.@testset "apply differential" begin
        for (m, X) in zip(m_pts, X_pts)
            for a in a_pts
                am, aX = apply(A, a, m), apply_diff(A, a, m, X)
                ainvm, ainvv = inverse_apply(A, a, m), inverse_apply_diff(A, a, m, X)
                Test.@test is_vector(M, am, aX, true; atol=atol)
                Test.@test is_vector(M, ainvm, ainvv, true; atol=atol)
            end
        end

        test_mutating_action && Test.@testset "mutating" begin
            for (m, X) in zip(m_pts, X_pts)
                for a in a_pts
                    am = apply(A, a, m)
                    aX = allocate(X)
                    Test.@test apply_diff!(A, aX, a, m, X) === aX
                    ainvm = inverse_apply(A, a, m)
                    ainvv = allocate(X)
                    Test.@test inverse_apply_diff!(A, ainvv, a, m, X) === ainvv
                    Test.@test is_vector(M, am, aX, true; atol=atol)
                    Test.@test is_vector(M, ainvm, ainvv, true; atol=atol)
                end
            end
        end
    end

    test_optimal_alignment && Test.@testset "Center of orbit" begin
        act = center_of_orbit(A, [m_pts[1]], m_pts[2])
        act2 = center_of_orbit(A, [m_pts[1]], m_pts[2], GradientDescentEstimation())
        act_opt = optimal_alignment(A, m_pts[2], m_pts[1])
        Test.@test isapprox(G, act, act_opt; atol=atol)
        Test.@test isapprox(G, act2, act_opt; atol=atol)

        test_mutating_group && Test.@testset "mutating" begin
            act_opt2 = allocate(act_opt)
            optimal_alignment!(A, act_opt2, m_pts[2], m_pts[1])
            Test.@test isapprox(G, act_opt, act_opt2; atol=atol)
        end
    end
    return nothing
end
