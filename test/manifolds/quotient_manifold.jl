include("../header.jl")

struct DummyManifold <: AbstractManifold{ℝ} end
struct DummyTotalSpace <: AbstractManifold{ℝ} end
struct DummyOrbit <: AbstractManifold{ℝ} end

@testset "QuotientManifold" begin
    Q = DummyManifold()
    T = DummyTotalSpace()
    M = QuotientManifold(Q, T)
    @test get_total_space(M) == DummyTotalSpace()
    @test repr(M) == "QuotientManifold($(Q), $(T))"

    @test decorated_manifold(M) == Q
end
