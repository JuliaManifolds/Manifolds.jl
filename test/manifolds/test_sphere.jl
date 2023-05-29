using Manifolds, Test, JLD2
using Manifolds: test_manifold, has_feature_expectations
# Generate or load Test Scenario – can be set here or globally by env
generate_test = false;
config_file = (@__DIR__) * "/config/sphere.jld2"
generate = get(ENV, "TEST_MANIFOLD_GENERATE_TESTS", generate_test)

M = Sphere(2)
if generate || !isfile(config_file)
    # Generate (semi-automatically) and save setup
    using Manifolds: find_manifold_functions, ManifoldFeatures, ManifoldExpectations
    features = ManifoldFeatures(M)
    expectations = ManifoldExpectations(
        values=Dict(:manifold_dimension => 2, :repr_manifold => "Sphere(2, ℝ)"),
        tolerances=Dict(:exp_atol => 1e-9),
    )
    jldsave(config_file; features, expectations)
    @warn "Configuration for the Sphere regenerated. This should not be actve by default."
else
    file = jldopen(config_file)
    features = file["features"]
    expectations = file["expectations"]
    close(file)
end

ps = [[1.0, 0.0, 0.0], 1 / sqrt(2) .* [1.0, 1.0, 0.0], 1 / sqrt(2) .* [1.0, 0.0, 1.0]]

Xs = [1 / sqrt(2) .* [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]

test_manifold(
    M;
    points=ps,
    tangent_vectors=Xs,
    features=features,
    expectations=expectations,
)
