using Manifolds, Test

@testset "Deprecation tests" begin
    a = [1.0, 2.0, 3.0]
    A = ones(3, 3)
    @test HyperboloidTVector(a) == HyperboloidTangentVector(a)
    @test OrthogonalTVector(A) == OrthogonalTangentVector(A)
    @test PoincareBallTVector(a) == PoincareBallTangentVector(a)
    @test PoincareHalfSpaceTVector(a) == PoincareHalfSpaceTangentVector(a)
    @test ProjectorTVector(A) == ProjectorTangentVector(A)
    @test StiefelTVector(A) == StiefelTangentVector(AbstractAtlas)
    @test TuckerTVector(A, (A, A)) == TuckerTangentVector(A, (A, A))
    @test UMVTVector(A, A, A) == UMVTangentVector(A, A, A)
end
