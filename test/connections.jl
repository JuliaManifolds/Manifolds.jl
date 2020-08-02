include("utils.jl")

@testset "Levi-Civita connection on Euclidean" begin
    M = Euclidean(2)
    fY(X) = [X[1]^2, -X[2] * X[1]]
    X1 = [1.0, 0.0]
    X2 = [0.0, 1.0]
    X3 = [2.0, -1.0]
    conn = Manifolds.LeviCivitaConnection(M)
    p = [1.0, 1.0]
    @test Manifolds.apply_operator(conn, p, X1, fY) ≈ [2.0, -1.0]
    @test Manifolds.apply_operator(conn, p, X2, fY) ≈ [0.0, -1.0]
    @test Manifolds.apply_operator(conn, p, X3, fY) ≈ [4.0, -1.0]
    Z = similar(X1)
    Manifolds.apply_operator!(conn, Z, p, X1, fY)
    @test Z ≈ [2.0, -1.0]
    Manifolds.apply_operator!(conn, Z, p, X2, fY)
    @test Z ≈ [0.0, -1.0]
    Manifolds.apply_operator!(conn, Z, p, X3, fY)
    @test Z ≈ [4.0, -1.0]

    rb_onb_fwd_diff = RiemannianONBDiffBackend(
        Manifolds.ForwardDiffBackend(),
        Manifolds.ExponentialRetraction(),
        Manifolds.LogarithmicInverseRetraction(),
        DefaultOrthonormalBasis(),
    )
    @test Manifolds.apply_operator(conn, p, X1, fY, rb_onb_fwd_diff) ≈ [2.0, -1.0]
    Manifolds.apply_operator!(conn, Z, p, X1, fY, rb_onb_fwd_diff)
    @test Z ≈ [2.0, -1.0]
end
