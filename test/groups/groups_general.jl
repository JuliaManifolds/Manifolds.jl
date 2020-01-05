
include("../utils.jl")


struct NotImplementedOperation <: AbstractGroupOperation
end

struct NotImplementedManifold <: Manifold
end

@testset "General group tests" begin
    @testset "Not implemented operation" begin
        G = GroupManifold(NotImplementedManifold(), NotImplementedOperation())
        x = [1.0, 2.0]
        v = [2.0, 3.0]
        eg = Identity(G)

        @test is_decorator_manifold(G) === Val(true)

        @test_throws ErrorException inv!(G, x, x)
        @test_throws ErrorException inv!(G, x, eg)
        @test_throws ErrorException inv(G, x)
        @test inv(G, eg) == eg

        @test_throws ErrorException identity!(G, x, x)
        @test_throws ErrorException identity(G, x)

        @test compose(G, eg, eg) == eg
        @test_throws ErrorException compose(G, x, x)
        @test compose(G, eg, x) == x
        @test compose(G, x, eg) == x
        @test_throws ErrorException compose!(G, x, eg, x)
        @test_throws ErrorException compose!(G, x, x, eg)
        @test_throws ErrorException compose!(G, x, x, x)
        @test_throws ErrorException compose!(G, x, eg, eg)

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

        @test_throws ErrorException translate_diff(G, x, x, v)
        @test_throws ErrorException translate_diff(G, x, x, v, LeftAction())
        @test_throws ErrorException translate_diff(G, x, x, v, RightAction())
        @test_throws ErrorException translate_diff!(G, v, x, x, v)
        @test_throws ErrorException translate_diff!(G, v, x, x, v, LeftAction())
        @test_throws ErrorException translate_diff!(G, v, x, x, v, RightAction())

        @test_throws ErrorException inverse_translate_diff(G, x, x, v)
        @test_throws ErrorException inverse_translate_diff(G, x, x, v, LeftAction())
        @test_throws ErrorException inverse_translate_diff(G, x, x, v, RightAction())
        @test_throws ErrorException inverse_translate_diff!(G, v, x, x, v)
        @test_throws ErrorException inverse_translate_diff!(G, v, x, x, v, LeftAction())
        @test_throws ErrorException inverse_translate_diff!(G, v, x, x, v, RightAction())
    end

    @testset "Action direction" begin
        @test switch_direction(LeftAction()) == RightAction()
        @test switch_direction(RightAction()) == LeftAction()
    end

    @testset "Addition operation" begin
        G = GroupManifold(NotImplementedManifold(), Manifolds.AdditionOperation())
        test_group(G, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        x = [1.0, 2.0]
        ge = Identity(G)
        @test zero(ge) === ge
        @test ge + x ≈ x
        @test x + ge ≈ x
        @test ge + ge === ge
        @test -ge === ge
        @test +ge === ge
        @test ge(x) ≈ zero(x)
        @test inv(G, x) ≈ -x
        @test identity(G, x) ≈ zero(x)
        y = similar(x)
        identity!(G, y, x)
        @test y ≈ zero(x)
        @test compose(G, x, x) ≈ x + x
        @test compose(G, x, ge) ≈ x
        @test compose(G, ge, x) ≈ x
        @test compose(G, ge, ge) == ge
        compose!(G, y, x, x)
        @test y ≈ x + x
        compose!(G, y, x, ge)
        @test y ≈ x
        compose!(G, y, ge, x)
        @test y ≈ x
    end

    @testset "Multiplication operation" begin
        G = GroupManifold(NotImplementedManifold(), Manifolds.MultiplicationOperation())
        test_group(G, [[1.0 2.0; 3.0 4.0], [2.0 3.0; 4.0 5.0], [3.0 4.0; 5.0 6.0]])

        x = [1.0 2.0; 2.0 3.0]
        ge = Identity(G)
        @test one(ge) === ge
        @test ge * x ≈ x
        @test x * ge ≈ x
        @test ge * ge === ge
        @test inv(ge) === ge
        @test *(ge) === ge

        @test x / ge ≈ x
        @test ge \ x ≈ x
        @test ge / ge === ge
        @test ge \ ge === ge
        @test ge / x ≈ inv(x)
        @test x \ ge ≈ inv(x)

        @test ge(x) ≈ one(x)
        @test inv(G, x) ≈ inv(x)
        @test inv(G, ge) === ge
        @test identity(G, x) ≈ one(x)
        y = similar(x)
        identity!(G, y, x)
        @test y ≈ one(x)
        @test compose(G, x, x) ≈ x * x
        @test compose(G, x, ge) ≈ x
        @test compose(G, ge, x) ≈ x
        @test compose(G, ge, ge) == ge
        compose!(G, y, x, x)
        @test y ≈ x * x
        compose!(G, y, x, ge)
        @test y ≈ x
        compose!(G, y, ge, x)
        @test y ≈ x

        @testset "identity optimization" begin
            x2 = copy(x)
            identity!(G, x2, x)
            x3 = copy(x)
            invoke(identity!, Tuple{AbstractGroupManifold{Manifolds.MultiplicationOperation}, Any, AbstractMatrix}, G, x3, x)
            @test isapprox(G, x2, x3)
        end
    end
end

struct NotImplementedAction <: AbstractGroupAction{LeftAction}
end

@testset "General group action tests" begin
    @testset "Not implemented operations" begin
        A = NotImplementedAction()
        x = [1.0, 2.0]
        a = [1.0, 2.0]

        @test_throws ErrorException base_group(A)
        @test_throws ErrorException g_manifold(A)
        @test_throws ErrorException apply(A, x, a)
        @test_throws ErrorException apply!(A, x, x, a)
        @test_throws ErrorException compose(A, a, a)
        @test_throws ErrorException compose!(A, a, a, a)
        @test_throws ErrorException optimal_alignment(A, x, x)
        @test_throws ErrorException optimal_alignment!(A, a, x, x)
    end
end
