using Manifolds, Test

@testset "HermitianPositiveDefinite" begin
    M = HermitianPositiveDefinite(2)
    M2 = HermitianPositiveDefinite(2, ℂ)
    M3 = HermitianPositiveDefinite(2; parameter=:field)

    A = [4.0 -2im; 2.0im 4.0]
    @test is_point(M, A, true)
end
