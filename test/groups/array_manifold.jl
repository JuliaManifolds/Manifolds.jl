@testset "Group wrapped in ArrayManifold" begin
    G = SpecialOrthogonal(3)
    M = Rotations(3)
    AG = ArrayManifold(G)
    @test base_group(AG) === G
    @test Manifolds.is_decorator_group(AG) === Val(true)

    eg = Matrix{Float64}(I, 3, 3)
    ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
    x, y = [exp(M, eg, hat(M, eg, ωi)) for ωi in ω]
    v = hat(M, eg, [-1.0, 2.0, 0.5])

    e = Identity(AG)
    @test e === Identity(G)
    x2, y2 = ArrayMPoint(x), ArrayMPoint(y)
    v2 = ArrayTVector(v)

    @test identity(AG, x2) isa ArrayMPoint
    @test isapprox(G, identity(AG, x2).value, identity(G, x))
    @test identity(AG, e) === e
    @test_throws DomainError identity(AG, Identity(TranslationGroup(3)))

    eg = similar(x2)
    identity!(AG, eg, x2)
    @test isapprox(G, eg.value, identity(G, x))
    eg = similar(x2)
    identity!(AG, eg, e)
    @test isapprox(G, eg.value, identity(G, x))

    @test inv(AG, x2) isa ArrayMPoint
    @test isapprox(G, inv(AG, x2).value, inv(G, x))
    @test inv(AG, e) === e
    @test_throws DomainError inv(AG, Identity(TranslationGroup(3)))

    xinv = similar(x2)
    inv!(AG, xinv, x2)
    @test isapprox(G, xinv.value, inv(G, x))
    eg = similar(x2)
    inv!(AG, eg, e)
    @test isapprox(G, eg.value, e)

    @test compose(AG, x2, y2) isa ArrayMPoint
    @test isapprox(G, compose(AG, x2, y2).value, compose(G, x, y))
    @test compose(AG, x2, e) === x2
    @test compose(AG, e, x2) === x2

    xy = similar(x2)
    compose!(AG, xy, x2, y2)
    @test isapprox(G, xy.value, compose(G, x, y))
    xy = similar(x2)
    compose!(AG, xy, e, y2)
    @test isapprox(G, xy.value, compose(G, e, y))
    xy = similar(x2)
    compose!(AG, xy, x2, e)
    @test isapprox(G, xy.value, compose(G, x, e))

    for conv in (LeftAction(), RightAction())
        @test translate(AG, x2, y2, conv) isa ArrayMPoint
        @test isapprox(G, translate(AG, x2, y2, conv).value, translate(G, x, y, conv))

        xy = similar(x2)
        translate!(AG, xy, x2, y2, conv)
        @test isapprox(G, xy.value, translate(G, x, y, conv))

        @test inverse_translate(AG, x2, y2, conv) isa ArrayMPoint
        @test isapprox(G, inverse_translate(AG, x2, y2, conv).value, inverse_translate(G, x, y, conv))

        xinvy = similar(x2)
        inverse_translate!(AG, xinvy, x2, y2, conv)
        @test isapprox(G, xinvy.value, inverse_translate(G, x, y, conv))
    end

    for conv in (LeftAction(), RightAction())
        @test translate_diff(AG, y2, x2, v2, conv; atol = 1e-10) isa ArrayTVector
        @test isapprox(G, translate_diff(AG, y2, x2, v2, conv; atol = 1e-10).value, translate_diff(G, y, x, v, conv))

        vout = similar(v2)
        translate_diff!(AG, vout, y2, x2, v2, conv; atol = 1e-10)
        @test isapprox(vout.value, translate_diff(G, y, x, v, conv))

        @test inverse_translate_diff(AG, y2, x2, v2, conv; atol = 1e-10) isa ArrayTVector
        @test isapprox(G, inverse_translate_diff(AG, y2, x2, v2, conv; atol = 1e-10).value, inverse_translate_diff(G, y, x, v, conv))

        vout = similar(v2)
        inverse_translate_diff!(AG, vout, y2, x2, v2, conv; atol = 1e-10)
        @test isapprox(vout.value, inverse_translate_diff(G, y, x, v, conv))
    end
end
