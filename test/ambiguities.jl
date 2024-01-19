"""
    has_type_in_signature(sig, T::Type)

Test whether the signature `sig` has an argument of type `T` as one of its parameters.
"""
function has_type_in_signature(sig, T::Type)
    return any(map(Base.unwrap_unionall(sig.sig).parameters) do x
        xw = Base.rewrap_unionall(x, sig.sig)
        return (xw isa Type ? xw : xw.T) <: T
    end)
end

@testset "Ambiguities" begin
    if VERSION.prerelease == () && !Sys.iswindows() && VERSION < v"1.10.0"
        mbs = Test.detect_ambiguities(ManifoldsBase)
        # Interims solution until we follow what was proposed in
        # https://discourse.julialang.org/t/avoid-ambiguities-with-individual-number-element-identity/62465/2
        fmbs = filter(x -> !any(has_type_in_signature.(x, Identity)), mbs)
        FMBS_LIMIT = 34
        println("Number of ManifoldsBase.jl ambiguities: $(length(fmbs))")
        @test length(fmbs) <= FMBS_LIMIT
        if length(fmbs) > FMBS_LIMIT
            for amb in fmbs
                println(amb)
                println()
            end
        end
        ms = Test.detect_ambiguities(Manifolds)
        # Interims solution until we follow what was proposed in
        # https://discourse.julialang.org/t/avoid-ambiguities-with-individual-number-element-identity/62465/2
        fms = filter(x -> !any(has_type_in_signature.(x, Identity)), ms)
        FMS_LIMIT = 47
        println("Number of Manifolds.jl ambiguities: $(length(fms))")
        if length(fms) > FMS_LIMIT
            for amb in fms
                println(amb)
                println()
            end
        end
        @test length(fms) <= FMS_LIMIT
        # this test takes way too long to perform regularly
        # @test length(our_base_ambiguities()) <= 4
    else
        @info "Skipping Ambiguity tests for pre-release versions"
    end
end
