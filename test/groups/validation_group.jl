include("../utils.jl")

@testset "Group wrapped in ValidationManifold" begin
    G = SpecialOrthogonal(3)
    M = Rotations(3)
    AG = ValidationManifold(G)
    @test base_group(AG) === G
    @test (@inferred Manifolds.decorator_group_dispatch(AG)) === Val(true)
    @test Manifolds.is_group_decorator(AG)

    eg = Matrix{Float64}(I, 3, 3)
    ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
    p, q = [exp(M, eg, hat(M, eg, ωi)) for ωi in ω]
    X = hat(M, eg, [-1.0, 2.0, 0.5])

    e = Identity(G)
    p2, q2 = ValidationMPoint(p), ValidationMPoint(q)
    @test q2 === Manifolds.array_point(q2) # test that double wraps are avoided.
    X2 = ValidationTVector(X)

    @test Identity(G) isa Identity

    eg = allocate(p2)
    identity_element!(G, eg)
    @test isapprox(G, eg, Identity(G))
    eg = allocate(p2)

    @test inv(AG, p2) isa ValidationMPoint
    @test isapprox(G, inv(AG, p2).value, inv(G, p))
    @test inv(AG, e) isa Identity
    @test_throws DomainError inv(AG, Identity(AG))

    pinvq = allocate(p2)
    inv!(AG, pinvq, p2)
    @test isapprox(G, pinvq.value, inv(G, p))
    eg = allocate(p2)
    inv!(AG, eg, e)
    @test isapprox(G, eg.value, Manifolds.array_value(e))

    @test compose(AG, p2, q2) isa ValidationMPoint
    @test isapprox(G, compose(AG, p2, q2).value, compose(G, p, q))
    @test compose(AG, p2, e) === p2
    @test compose(AG, e, p2) === p2

    pq = allocate(p2)
    compose!(AG, pq, p2, q2)
    @test isapprox(G, pq.value, compose(G, p, q))
    pq = allocate(p2)
    compose!(AG, pq, e, q2)
    @test isapprox(G, pq.value, compose(G, e, q))
    pq = allocate(p2)
    compose!(AG, pq, p2, e)
    @test isapprox(G, pq.value, compose(G, p, e))

    @test exp_lie(AG, X2) isa ValidationMPoint
    @test isapprox(G, exp_lie(AG, X2).value, exp_lie(G, X))
    expX = allocate(p2)
    exp_lie!(AG, expX, X2)
    @test isapprox(G, expX.value, exp_lie(G, X))

    @test log_lie(AG, p2) isa ValidationTVector
    @test isapprox(G, e, log_lie(AG, p2).value, log_lie(G, p))
    logp = allocate(X2)
    log_lie!(AG, logp, p2)
    @test isapprox(G, e, logp.value, log_lie(G, p))

    @test lie_bracket(AG, X, X2) isa ValidationTVector
    Xlb = allocate(X2)
    lie_bracket!(AG, Xlb, X, X2)
    @test isapprox(G, e, Xlb.value, lie_bracket(G, X, X2.value))

    @test adjoint_action(AG, p, X) isa ValidationTVector
    Xaa = allocate(X2)
    adjoint_action!(AG, Xaa, p, X2)
    @test isapprox(G, e, Xaa.value, adjoint_action(G, p, X2.value))

    for conv in (LeftAction(), RightAction())
        @test translate(AG, p2, q2, conv) isa ValidationMPoint
        @test isapprox(G, translate(AG, p2, q2, conv).value, translate(G, p, q, conv))

        pq = allocate(p2)
        translate!(AG, pq, p2, q2, conv)
        @test isapprox(G, pq.value, translate(G, p, q, conv))

        @test inverse_translate(AG, p2, q2, conv) isa ValidationMPoint
        @test isapprox(
            G,
            inverse_translate(AG, p2, q2, conv).value,
            inverse_translate(G, p, q, conv),
        )

        pinvq = allocate(p2)
        inverse_translate!(AG, pinvq, p2, q2, conv)
        @test isapprox(G, pinvq.value, inverse_translate(G, p, q, conv))
    end

    for conv in (LeftAction(), RightAction())
        @test translate_diff(AG, q2, p2, X2, conv; atol=1e-10) isa ValidationTVector
        @test isapprox(
            G,
            translate_diff(AG, q2, p2, X2, conv; atol=1e-10).value,
            translate_diff(G, q, p, X, conv),
        )

        Y = allocate(X2)
        translate_diff!(AG, Y, q2, p2, X2, conv; atol=1e-10)
        @test isapprox(Y.value, translate_diff(G, q, p, X, conv))

        @test inverse_translate_diff(AG, q2, p2, X2, conv; atol=1e-10) isa ValidationTVector
        @test isapprox(
            G,
            inverse_translate_diff(AG, q2, p2, X2, conv; atol=1e-10).value,
            inverse_translate_diff(G, q, p, X, conv),
        )

        Y = allocate(X2)
        inverse_translate_diff!(AG, Y, q2, p2, X2, conv; atol=1e-10)
        @test isapprox(Y.value, inverse_translate_diff(G, q, p, X, conv))
    end
end
