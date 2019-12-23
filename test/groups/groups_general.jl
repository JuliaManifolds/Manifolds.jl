
include("../utils.jl")


struct NotImplementedOperation <: AbstractGroupOperation
end

struct NotImplementedManifold <: Manifold
end

@testset "General group tests" begin
    @testset "Not implemented operation" begin
        G = GroupManifold(NotImplementedManifold(), NotImplementedOperation())
        x = [1.0, 2.0]

        @test is_decorator_manifold(G) === Val(true)

        @test_throws ErrorException inv!(G, x, x)
        @test_throws ErrorException inv(G, x)

        @test_throws ErrorException group_id!(G, x, x)
        @test_throws ErrorException group_id(G, x)

        @test compose_left(G, x) === x
        @test_throws ErrorException compose_left(G, x, x)
        @test_throws ErrorException compose_left!(G, x, x, x)
        @test_throws ErrorException compose_left(G, x, x, x)

        @test_throws ErrorException translate(G, x, x)
        @test_throws ErrorException translate(G, x, x, LeftAction())
        @test_throws ErrorException translate(G, x, x, RightAction())
        @test_throws ErrorException translate!(G, x, x, x)
        @test_throws ErrorException translate!(G, x, x, x, LeftAction())
        @test_throws ErrorException translate!(G, x, x, x, RightAction())

        @test_throws ErrorException inverse_translate(G, x, x)
        @test_throws ErrorException inverse_translate(G, x, x, LeftAction())
        @test_throws ErrorException inverse_translate(G, x, x, RightAction())
        @test_throws ErrorException inverse_translate!(G, x, x, x)
        @test_throws ErrorException inverse_translate!(G, x, x, x, LeftAction())
        @test_throws ErrorException inverse_translate!(G, x, x, x, RightAction())
    end

    @testset "Addition operation" begin
        G = GroupManifold(NotImplementedManifold(), Manifolds.AdditionOperation())
        x = [1.0, 2.0]
        ge = Identity(G)
        @test ge + x ≈ x
        @test x + ge ≈ x
        @test ge + ge === ge
        @test -ge === ge
        @test +ge === ge
        @test ge(x) ≈ zero(x)
        @test inv(G, x) ≈ -x
        @test group_id(G, x) ≈ zero(x)
        y = similar(x)
        group_id!(G, y, x)
        @test y ≈ zero(x)
        @test compose_left(G, x, x) ≈ x + x
        compose_left!(G, y, x, x)
        @test y ≈ x + x
    end

    @testset "Multiplication operation" begin
        G = GroupManifold(NotImplementedManifold(), Manifolds.MultiplicationOperation())
        x = [1.0 2.0; 2.0 3.0]
        ge = Identity(G)
        @test ge * x ≈ x
        @test x * ge ≈ x
        @test ge * ge === ge
        @test inv(ge) === ge
        @test *(ge) === ge
        @test ge(x) ≈ one(x)
        @test inv(G, x) ≈ inv(x)
        @test group_id(G, x) ≈ one(x)
        y = similar(x)
        group_id!(G, y, x)
        @test y ≈ one(x)
        @test compose_left(G, x, x) ≈ x * x
        compose_left!(G, y, x, x)
        @test y ≈ x * x
    end
end
