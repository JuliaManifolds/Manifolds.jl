@testset "Group wrapped in ArrayManifold" begin
    G = SpecialOrthogonal(3)
    M = Rotations(3)
    AG = ArrayManifold(G)
    @test base_group(AG) === G
    @test Manifolds.is_decorator_group(AG) === Val(true)

    eg = Matrix{Float64}(I, 3, 3)
    ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
    p, q = [exp(M, eg, hat(M, eg, ωi)) for ωi in ω]
    X = hat(M, eg, [-1.0, 2.0, 0.5])

    e = Identity(AG)
    @test e === Identity(G)
    p2, q2 = ArrayMPoint(p), ArrayMPoint(q)
    X2 = ArrayTVector(X)

    @test identity(AG, p2) isa ArrayMPoint
    @test isapprox(G, identity(AG, p2).value, identity(G, p))
    @test identity(AG, e) === e
    @test_throws DomainError identity(AG, Identity(TranslationGroup(3)))

    eg = similar(p2)
    identity!(AG, eg, p2)
    @test isapprox(G, eg.value, identity(G, p))
    eg = similar(p2)
    identity!(AG, eg, e)
    @test isapprox(G, eg.value, identity(G, p))

    @test inv(AG, p2) isa ArrayMPoint
    @test isapprox(G, inv(AG, p2).value, inv(G, p))
    @test inv(AG, e) === e
    @test_throws DomainError inv(AG, Identity(TranslationGroup(3)))

    xinv = similar(p2)
    inv!(AG, xinv, p2)
    @test isapprox(G, xinv.value, inv(G, p))
    eg = similar(p2)
    inv!(AG, eg, e)
    @test isapprox(G, eg.value, e)

    @test compose(AG, p2, q2) isa ArrayMPoint
    @test isapprox(G, compose(AG, p2, q2).value, compose(G, p, q))
    @test compose(AG, p2, e) === p2
    @test compose(AG, e, p2) === p2

    pq = similar(p2)
    compose!(AG, pq, p2, q2)
    @test isapprox(G, pq.value, compose(G, p, q))
    pq = similar(p2)
    compose!(AG, pq, e, q2)
    @test isapprox(G, pq.value, compose(G, e, q))
    pq = similar(p2)
    compose!(AG, pq, p2, e)
    @test isapprox(G, pq.value, compose(G, p, e))

    for conv in (LeftAction(), RightAction())
        @test translate(AG, p2, q2, conv) isa ArrayMPoint
        @test isapprox(G, translate(AG, p2, q2, conv).value, translate(G, p, q, conv))

        pq = similar(p2)
        translate!(AG, pq, p2, q2, conv)
        @test isapprox(G, pq.value, translate(G, p, q, conv))

        @test inverse_translate(AG, p2, q2, conv) isa ArrayMPoint
        @test isapprox(G, inverse_translate(AG, p2, q2, conv).value, inverse_translate(G, p, q, conv))

        pinvq = similar(p2)
        inverse_translate!(AG, pinvq, p2, q2, conv)
        @test isapprox(G, pinvq.value, inverse_translate(G, p, q, conv))
    end

    for conv in (LeftAction(), RightAction())
        @test translate_diff(AG, q2, p2, X2, conv; atol = 1e-10) isa ArrayTVector
        @test isapprox(G, translate_diff(AG, q2, p2, X2, conv; atol = 1e-10).value, translate_diff(G, q, p, X, conv))

        Y = similar(X2)
        translate_diff!(AG, Y, q2, p2, X2, conv; atol = 1e-10)
        @test isapprox(Y.value, translate_diff(G, q, p, X, conv))

        @test inverse_translate_diff(AG, q2, p2, X2, conv; atol = 1e-10) isa ArrayTVector
        @test isapprox(G, inverse_translate_diff(AG, q2, p2, X2, conv; atol = 1e-10).value, inverse_translate_diff(G, q, p, X, conv))

        Y = similar(X2)
        inverse_translate_diff!(AG, Y, q2, p2, X2, conv; atol = 1e-10)
        @test isapprox(Y.value, inverse_translate_diff(G, q, p, X, conv))
    end
end
