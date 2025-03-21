using Manifolds, Test

@testset "Invertible Matrices of Determinant one" begin
    M = InvertibleMatricesDeterminantOne(2)
    # is det 1 and inv
    @test is_point(M, [1.0 0.0; 0.0 1.0], true)
    # det 1 but for example not Rotation
    @test is_point(M, [10.0 0.0; 0.0 0.1], true)
    # Not invertible
    @test_throws DomainError is_point(M, [10.0 0.0; 0.0 0.0], true)
    # Det(2)
    @test_throws DomainError is_point(M, [2.0 0.0; 0.0 1.0], true)
end
