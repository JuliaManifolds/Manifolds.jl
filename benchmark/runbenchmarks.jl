using PkgBenchmark

#--track-allocation=all
config = BenchmarkConfig(
    id = nothing,
    juliacmd = `julia -O3`,
    env = Dict("JULIA_NUM_THREADS" => 4),
)

results = benchmarkpkg("Manifolds", config)
export_markdown("benchmark/results.md", results)
