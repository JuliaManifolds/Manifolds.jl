using Aqua, Manifolds, Test, StaticArrays # Last package temporary

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
                Manifolds.TranslationGroup,
                Manifolds.GroupManifold,
            ],
            broken=false,
        ),
        piracies=(
            treat_as_own=[
                AbstractDecoratorManifold, # MAybe fix?
                AbstractManifold, # Maybe fix?
                AbstractNumbers,  # Maybe fix?
                ProductManifold,  # Maybe fix?
                ExtrinsicEstimation, # already deprecated
                Manifolds.EmptyTrait, # Maybe fix?
                Manifolds.TraitList, # Maybe fix?
                Manifolds.GeodesicInterpolationWithinRadius, # Probably fix
                # StaticArray, # Definetly fix!
                allocate, # Maybe fix
            ],
        ),
    )
end
