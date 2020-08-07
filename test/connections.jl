include("utils.jl")

@testset "Levi-Civita connection on Euclidean" begin
    M = Euclidean(2)
    fY(p) = [p[1]^2, -p[2] * p[1]] # our vector field
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


@testset "Levi-Civita connection on Sphere" begin
    M = Sphere(2)
    fY(p) = log(M, p, [1.0, 0.0, 0.0]) # our vector field
    conn = LeviCivitaConnection(M)
    p = [0.0, 1.0, 0.0]
    X1 = [0.0, 0.0, 1.0]
    X2 = [2.0, 0.0, 0.0]
    @test isapprox(M, p, apply_operator(conn, p, X1, fY), [0.0, 0.0, 0.0]; atol = 1e-15)
    @test isapprox(M, p, apply_operator(conn, p, X2, fY), [-2.0, 0.0, 0.0])
    Z = similar(X1)
    apply_operator!(conn, Z, p, X1, fY)
    @test isapprox(M, p, Z, [0.0, 0.0, 0.0]; atol = 1e-15)
    apply_operator!(conn, Z, p, X2, fY)
    @test isapprox(M, p, Z, [-2.0, 0.0, 0.0])

    rb_onb_fwd_diff = RiemannianONBDiffBackend(
        Manifolds.ForwardDiffBackend(),
        Manifolds.ExponentialRetraction(),
        Manifolds.LogarithmicInverseRetraction(),
        DefaultOrthonormalBasis(),
    )
    @test isapprox(
        M,
        p,
        apply_operator(conn, p, X1, fY, rb_onb_fwd_diff),
        [0.0, 0.0, 0.0];
        atol = 1e-15,
    )
    @test isapprox(M, p, apply_operator(conn, p, X2, fY, rb_onb_fwd_diff), [-2.0, 0.0, 0.0])
    apply_operator!(conn, Z, p, X1, fY, rb_onb_fwd_diff)
    @test isapprox(M, p, Z, [0.0, 0.0, 0.0]; atol = 1e-15)
    apply_operator!(conn, Z, p, X2, fY, rb_onb_fwd_diff)
    @test isapprox(M, p, Z, [-2.0, 0.0, 0.0])
end

@testset "Hessian on Euclidean" begin
    M = Euclidean(2)
    f(p) = p[1]^2 - p[1] * p[2]^2

    Hop = HessianOperator(LeviCivitaConnection(M))
    p = [1.0, 2.0]
    X1 = [1.0, 0.0]
    X2 = [0.0, 2.0]
    Hp = [2.0 -4.0; -4.0 -2.0]
    @test isapprox(M, p, apply_operator(Hop, p, X1, f), Hp * X1, atol = 1e-8)
    @test isapprox(M, p, apply_operator(Hop, p, X2, f), Hp * X2, atol = 1e-8)

    Z = similar(X1)
    apply_operator!(Hop, Z, p, X1, f)
    @test isapprox(M, p, Z, Hp * X1, atol = 1e-8)
    apply_operator!(Hop, Z, p, X2, f)
    @test isapprox(M, p, Z, Hp * X2, atol = 1e-8)

    rb_onb_fwd_diff = RiemannianONBDiffBackend(
        Manifolds.ForwardDiffBackend(),
        Manifolds.ExponentialRetraction(),
        Manifolds.LogarithmicInverseRetraction(),
        DefaultOrthonormalBasis(),
    )

    @test isapprox(
        M,
        p,
        apply_operator(Hop, p, X1, f, rb_onb_fwd_diff, rb_onb_fwd_diff),
        Hp * X1,
    )
    @test isapprox(
        M,
        p,
        apply_operator(Hop, p, X2, f, rb_onb_fwd_diff, rb_onb_fwd_diff),
        Hp * X2,
    )

end

@testset "Hessian on Sphere" begin
    M = Sphere(2)
    f(p) = p[1]^2 - p[1] * p[2]^2 + 10*p[3]^2*p[1]

    Hop = HessianOperator(LeviCivitaConnection(M))
    p = [1.0, 0.0, 0.0]
    X1 = [0.0, 1.0, 0.0]
    X2 = [0.0, 0.0, 2.0]
    @test isapprox(M, p, apply_operator(Hop, p, X1, f), [0.0, -4.0, 0.0], atol = 1e-8)
    @test isapprox(M, p, apply_operator(Hop, p, X2, f), [0.0, 0.0, 36.0], atol = 1e-8)

    Z = similar(X1)
    apply_operator!(Hop, Z, p, X1, f)
    @test isapprox(M, p, Z, [0.0, -4.0, 0.0], atol = 1e-8)
    apply_operator!(Hop, Z, p, X2, f)
    @test isapprox(M, p, Z, [0.0, 0.0, 36.0], atol = 1e-8)

    rb_onb_fwd_diff = RiemannianONBDiffBackend(
        Manifolds.ForwardDiffBackend(),
        Manifolds.ExponentialRetraction(),
        Manifolds.LogarithmicInverseRetraction(),
        DefaultOrthonormalBasis(),
    )

    @test isapprox(
        M,
        p,
        apply_operator(Hop, p, X1, f, rb_onb_fwd_diff, rb_onb_fwd_diff),
        [0.0, -4.0, 0.0],
    )
    @test isapprox(
        M,
        p,
        apply_operator(Hop, p, X2, f, rb_onb_fwd_diff, rb_onb_fwd_diff),
        [0.0, 0.0, 36.0],
    )

end

