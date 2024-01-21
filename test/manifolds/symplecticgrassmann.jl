include("../header.jl")

@testset "Symplectic Grassmann" begin
    M = SymplecticGrassmann(6, 4)
    Mf = SymplecticGrassmann(6, 4; parameter=:field)
    p = [
        0.0 0.0 -5.0 -1.0
        0.0 0.0 9.0 -2.0
        0.0 0.0 -2.0 1.0
        -2.0 -9.0 -3.0 6.0
        -3.0 -13.0 -21.0 9.0
        -8.0 -36.0 18.0 -6.0
    ]
    q = [
        0.0 0.0 -3.0 1.0
        0.0 0.0 8.0 -3.0
        0.0 0.0 -2.0 1.0
        -1.0 -4.0 -6.0 3.0
        -1.0 -3.0 -21.0 9.0
        -2.0 -6.0 18.0 -6.0
    ]
    X = [
        0.0 0.0 4.25 4.25
        0.0 0.0 0.125 0.125
        0.0 0.0 -1.125 -1.125
        3.625 18.125 -10.875 -10.875
        5.0 25.0 -9.0 -9.0
        13.5 67.5 4.5 4.5
    ]
    Y = [
        -0.02648060 0.00416977 0.01130802 0.01015956
        0.01718954 -0.00680433 0.02364406 -0.00083272
        0.00050392 0.00191916 -0.01035902 -0.00079734
        0.01811917 -0.02307032 -0.04297277 -0.05409099
        -0.02529516 0.00959934 -0.08594555 -0.06117803
        -0.02823014 0.00029946 -0.04196034 -0.04145413
    ]
    @testset "Basics" begin
        @test repr(M) == "SymplecticStiefel(6, 4; field=ℝ)"
        @test repr(Mf) == "SymplecticStiefel(6, 4; field=ℝ; parameter=:field)"
        @test manifold_dimension(M) == 4 * (6 - 4)
        for _M in [M, Mf]
            @test is_point(M, p)
            @test is_vector(_M, p, X)
        end
        @test get_embedding(M) == SymplecticStiefel(6, 4)
    end
    @testset "Embedding / Total Space" begin
        @test get_embedding(M) == SymplecticStiefel(6, 4)
        pE = similar(p)
        embed!(M, pE, p)
        @test p == pE
        embed!(M, pE, StiefelPoint(p))
        @test p == pE
        @test embed(M, StiefelPoint(p)) == p
        XE = similar(X)
        embed!(M, XE, p, X)
        @test XE == X
        embed!(M, XE, StiefelPoint(p), StiefelTVector(X))
        @test XE == X
        @test embed(M, StiefelPoint(p), StiefelTVector(X)) == X
    end
    @testset "Expnential and Retractions" begin
        @test inner(M, p, X, X) ≈ norm(M, p, X)^2
        N = get_embedding(M)
        @test isapprox(N, exp(M, p, X), exp(N, p, X))
        rtm = CayleyRetraction()
        r = retract(M, p, X, rtm)
        @test is_point(M, r)
        irtm = CayleyInverseRetraction()
        X2 = inverse_retract(M, p, r, irtm)
        @test isapprox(M, p, X, X2)
        @test is_vector(M, p, X2)
    end
    @testset "Riemannian Gradient conversion" begin
        A = Matrix{Float64}(I, 6, 6)[:, 1:4]
        Z = riemannian_gradient(M, p, A)
        Z2 = similar(Z)
        riemannian_gradient!(M, Z2, p, A)
        @test isapprox(M, Z2, Z)
        # How can we better test that this is a correct gradient?
        # Or what can we further test here?
    end
end
