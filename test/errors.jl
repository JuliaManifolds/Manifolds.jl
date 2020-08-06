#
# Test specific error methods
#
include("utils.jl")

@testset "Test specific Errors and their format." begin
    e = DomainError(1.0, "Norm not zero.") # a dummy
    e2 = ComponentManifoldError(1, e)

    s1 = sprint(showerror, e)
    s2 = sprint(showerror, e2)
    @test s2 == "At #1: $(s1)"

    e3 = CompositeManifoldError()
    s3 = sprint(showerror, e3)
    @test s3 == "CompositeManifoldError()\n"
    @test length(e3) == 0
    @test isempty(e3)
    @test repr(e3) == "CompositeManifoldError()"

    e4 = CompositeManifoldError([e2])
    @test repr(e4) == "CompositeManifoldError([$(repr(e2)), ])"
    s4 = sprint(showerror, e4)
    @test s4 == "CompositeManifoldError: $(s2)"
end
