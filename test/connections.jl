include("utils.jl")

@testset "Connections" begin
    M = Euclidean(2)
    f(x) = [2*x[1]*x[2], x[1]^2]
    F = Manifolds.FunctionLeviCivitaConnection(M, f)

    rb_onb_fwd_diff = RiemannianONBDiffBackend(
        Manifolds.ForwardDiffBackend(),
        Manifolds.ExponentialRetraction(),
        Manifolds.LogarithmicInverseRetraction(),
        DefaultOrthonormalBasis(),
)

    @test isapprox(Manifolds.apply_operator(F, [1, 1], [1, 0]), [2, 2])
    @test isapprox(Manifolds.apply_operator(F, [1, 1], [1, 0], rb_onb_fwd_diff), [2, 2])
    Y = zeros(2)
    Manifolds.apply_operator!(F, Y, [1, 1], [1, 0])
    @test isapprox(Y, [2, 2])
    Y2 = similar(Y)
    Manifolds.apply_operator!(F, Y2, [1, 1], [1, 0], rb_onb_fwd_diff)
    @test isapprox(Y2, [2, 2])
end
