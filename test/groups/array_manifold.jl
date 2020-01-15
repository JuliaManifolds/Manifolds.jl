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

    @test inv(AG, x2) isa ArrayMPoint
    @test isapprox(G, inv(AG, x2).value, inv(G, x))
    @test inv(AG, e) === e
    @test_throws DomainError inv(AG, Identity(TranslationGroup(3)))

    @test compose(AG, x2, y2) isa ArrayMPoint
    @test isapprox(G, compose(AG, x2, y2).value, compose(G, x, y))
    @test compose(AG, x2, e) === x2
    @test compose(AG, e, x2) === x2

    @test translate(AG, x2, y2, LeftAction()) isa ArrayMPoint
    @test isapprox(G, translate(AG, x2, y2, LeftAction()).value, translate(G, x, y, LeftAction()))

    @test translate(AG, x2, y2, RightAction()) isa ArrayMPoint
    @test isapprox(G, translate(AG, x2, y2, RightAction()).value, translate(G, x, y, RightAction()))

    @test inverse_translate(AG, x2, y2, LeftAction()) isa ArrayMPoint
    @test isapprox(G, inverse_translate(AG, x2, y2, LeftAction()).value, inverse_translate(G, x, y, LeftAction()))

    @test inverse_translate(AG, x2, y2, RightAction()) isa ArrayMPoint
    @test isapprox(G, inverse_translate(AG, x2, y2, RightAction()).value, inverse_translate(G, x, y, RightAction()))

    @test translate_diff(AG, y2, x2, v2, LeftAction(); atol = 1e-10) isa ArrayTVector
    @test isapprox(G, translate_diff(AG, y2, x2, v2, LeftAction(); atol = 1e-10).value, translate_diff(G, y, x, v, LeftAction()))

    @test translate_diff(AG, y2, x2, v2, RightAction(); atol = 1e-10) isa ArrayTVector
    @test isapprox(G, translate_diff(AG, y2, x2, v2, RightAction(); atol = 1e-10).value, translate_diff(G, y, x, v, RightAction()))

    @test inverse_translate_diff(AG, y2, x2, v2, LeftAction(); atol = 1e-10) isa ArrayTVector
    @test isapprox(G, inverse_translate_diff(AG, y2, x2, v2, LeftAction(); atol = 1e-10).value, inverse_translate_diff(G, y, x, v, LeftAction()))

    @test inverse_translate_diff(AG, y2, x2, v2, RightAction(); atol = 1e-10) isa ArrayTVector
    @test isapprox(G, inverse_translate_diff(AG, y2, x2, v2, RightAction(); atol = 1e-10).value, inverse_translate_diff(G, y, x, v, RightAction()))
end
