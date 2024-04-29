using Aqua, Manifolds, Test

@testset "Aqua.jl" begin
    Aqua.test_all(
        Manifolds;
        ambiguities=(
            exclude=[
                *,
                ==,
                allocate_result,
                inv,
                inv!,
                mean,
                Manifolds.mul!,
                reshape,
                Manifolds.SemidirectProductOperation,
                setindex!,
                Manifolds.Manifolds.TranslationGroup,
            ],
            broken=false,
        ),
    )
end
