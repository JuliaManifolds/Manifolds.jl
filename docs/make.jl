#!/usr/bin/env julia
#
#

if "--help" ∈ ARGS
    println(
        """
        docs/make.jl

        Render the `Manifolds.jl` documentation with optional arguments

        Arguments
        * `--exclude-tutorials` - exclude the tutorials from the menu of Documenter,
          This can be used if not all tutorials are rendered and you want to therefore exclude links
          to these, especially the corresponding menu. This option should not be set on CI.
          Locally this is also set if `--quarto` is not set and not all tutorials are rendered.
        * `--help`              - print this help and exit without rendering the documentation
        * `--prettyurls`        – toggle the pretty urls part to true, which is always set on CI
        * `--quarto`            – (re)run the Quarto notebooks from the `tutorials/` folder before
          generating the documentation. If they are generated once they are cached accordingly.
          Then you can spare time in the rendering by not passing this argument.
          If quarto is not run, some tutorials are generated as empty files, since they
          are referenced from within the documentation.
        """,
    )
    exit(0)
end

run_quarto = "--quarto" in ARGS
run_on_CI = (get(ENV, "CI", nothing) == "true")
tutorials_in_menu = !("--exclude-tutorials" ∈ ARGS)
#
#
# (a) setup the tutorials menu – check whether all files exist
tutorials_menu =
    "How to..." => [
    "🚀 Get Started with `Manifolds.jl`" => "tutorials/getstarted.md",
    "work in charts" => "tutorials/working-in-charts.md",
    "perform Hand gesture analysis" => "tutorials/hand-gestures.md",
    "integrate on manifolds and handle probability densities" => "tutorials/integration.md",
    "explore curvature without coordinates" => "tutorials/exploring-curvature.md",
]
# Check whether all tutorials are rendered, issue a warning if not (and quarto if not set)
all_tutorials_exist = true
for (name, file) in tutorials_menu.second
    fn = joinpath(@__DIR__, "src/", file)
    if !isfile(fn) || filesize(fn) == 0 # nonexistent or empty file
        global all_tutorials_exist = false
        if !run_quarto
            @warn "Tutorial $name does not exist at $fn."
            if (!isfile(fn)) && (endswith(file, "getstarted.md"))
                @warn "Generating empty file, since this tutorial is linked to from the documentation."
                touch(fn)
            end
        end
    end
end
if !all_tutorials_exist && !run_quarto && !run_on_CI
    @warn """
        Not all tutorials exist. Run `make.jl --quarto` to generate them. For this run they are excluded from the menu.
    """
    tutorials_in_menu = false
end
if !tutorials_in_menu
    @warn """
    You are either explicitly or implicitly excluding the tutorials from the documentation.
    You will not be able to see their menu entries nor their rendered pages.
    """
    run_on_CI &&
        (@error "On CI, the tutorials have to be either rendered with Quarto or be cached.")
end
#
# (b) if docs is not the current active environment, switch to it
# (from https://github.com/JuliaIO/HDF5.jl/pull/1020/) 
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

# (c) If quarto is set, or we are on CI, run quarto
if run_quarto || run_on_CI
    @info "Rendering Quarto"
    tutorials_folder = (@__DIR__) * "/../tutorials"
    # instantiate the tutorials environment if necessary
    Pkg.activate(tutorials_folder)
    # For a breaking release -> also set the tutorials folder to the most recent version
    Pkg.instantiate()
    Pkg.activate(@__DIR__) # but return to the docs one before
    run(`quarto render $(tutorials_folder)`)
end

# (d) load necessary packages for the docs
using Plots, RecipesBase, Manifolds, ManifoldsBase, Documenter, PythonPlot
using DocumenterCitations, DocumenterInterLinks
# required for loading methods that handle differential equation solving
using OrdinaryDiffEq, BoundaryValueDiffEq, DiffEqCallbacks
using NLsolve
# required for loading the manifold tests functions
using Test, FiniteDifferences
ENV["GKSwstype"] = "100"

# (e) add CONTRIBUTING.md and NEWS.md to docs

function add_links(line::String, url::String = "https://github.com/JuliaManifolds/Manifolds.jl")
    # replace issues (#XXXX) -> ([#XXXX](url/issue/XXXX))
    while (m = match(r"\(\#([0-9]+)\)", line)) !== nothing
        id = m.captures[1]
        line = replace(line, m.match => "([#$id]($url/issues/$id))")
    end
    # replace ## [X.Y.Z] -> with a link to the release [X.Y.Z](url/releases/tag/vX.Y.Z)
    while (m = match(r"\#\# \[([0-9]+.[0-9]+.[0-9]+)\] (.*)", line)) !== nothing
        tag = m.captures[1]
        date = m.captures[2]
        line = replace(line, m.match => "## [$tag]($url/releases/tag/v$tag) ($date)")
    end
    return line
end

generated_path = joinpath(@__DIR__, "src", "misc")
base_url = "https://github.com/JuliaManifolds/Manifolds.jl/blob/master/"
isdir(generated_path) || mkdir(generated_path)
for fname in ["CONTRIBUTING.md", "NEWS.md"]
    open(joinpath(generated_path, fname), "w") do io
        # Point to source license file
        println(
            io,
            """
            ```@meta
            EditURL = "$(base_url)$(fname)"
            ```
            """,
        )
        # Write the contents out below the meta block
        for line in eachline(joinpath(dirname(@__DIR__), fname))
            println(io, add_links(line))
        end
    end
end

# (f) final step: render the docs
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style = :alpha)
links = InterLinks(
    "ManifoldsBase" => ("https://juliamanifolds.github.io/ManifoldsBase.jl/stable/"),
)

modules = [
    Manifolds,
    Base.get_extension(Manifolds, :ManifoldsBoundaryValueDiffEqExt),
    Base.get_extension(Manifolds, :ManifoldsNLsolveExt),
    Base.get_extension(Manifolds, :ManifoldsOrdinaryDiffEqDiffEqCallbacksExt),
    Base.get_extension(Manifolds, :ManifoldsOrdinaryDiffEqExt),
    Base.get_extension(Manifolds, :ManifoldsRecipesBaseExt),
    Base.get_extension(Manifolds, :ManifoldsTestExt),
    Manifolds.Test,
]

if modules isa Vector{Union{Nothing, Module}}
    error("At least one module has not been properly loaded: ", modules)
end

makedocs(;
    format = Documenter.HTML(
        prettyurls = (get(ENV, "CI", nothing) == "true") || ("--prettyurls" ∈ ARGS),
        assets = ["assets/favicon.ico", "assets/citations.css", "assets/link-icons.css"],
        search_size_threshold_warn = 1000 * 2^10, # raise slightly from 500 to 1 MiB
        size_threshold_warn = 200 * 2^10, # raise slightly from 100 to 200 KiB
        size_threshold = 300 * 2^10,      # raise slightly 200 to 300 KiB
    ),
    modules = modules,
    authors = "Seth Axen, Mateusz Baran, Ronny Bergmann, and contributors.",
    sitename = "Manifolds.jl",
    pages = [
        "Home" => "index.md",
        (tutorials_in_menu ? [tutorials_menu] : [])...,
        "Manifolds" => [
            "Basic manifolds" => [
                "Centered matrices" => "manifolds/centeredmatrices.md",
                "Cholesky space" => "manifolds/choleskyspace.md",
                "Circle" => "manifolds/circle.md",
                "Determinant one matrices" => "manifolds/determinantone.md",
                "Elliptope" => "manifolds/elliptope.md",
                "Essential manifold" => "manifolds/essentialmanifold.md",
                "Euclidean" => "manifolds/euclidean.md",
                "Fixed-rank matrices" => "manifolds/fixedrankmatrices.md",
                "Flag" => "manifolds/flag.md",
                "Generalized Stiefel" => "manifolds/generalizedstiefel.md",
                "Generalized Grassmann" => "manifolds/generalizedgrassmann.md",
                "Grassmann" => "manifolds/grassmann.md",
                "Hamiltonian" => "manifolds/hamiltonian.md",
                "Heisenberg matrices" => "manifolds/heisenberg.md",
                "Hyperbolic space" => "manifolds/hyperbolic.md",
                "Hyperrectangle" => "manifolds/hyperrectangle.md",
                "Invertible matrices" => "manifolds/invertible.md",
                "Lorentzian manifold" => "manifolds/lorentz.md",
                "Multinomial doubly stochastic matrices" => "manifolds/multinomialdoublystochastic.md",
                "Multinomial matrices" => "manifolds/multinomial.md",
                "Multinomial symmetric matrices" => "manifolds/multinomialsymmetric.md",
                "Multinomial symmetric positive definite matrices" => "manifolds/multinomialsymmetricpositivedefinite.md",
                "Oblique manifold" => "manifolds/oblique.md",
                "Probability simplex" => "manifolds/probabilitysimplex.md",
                "Positive numbers" => "manifolds/positivenumbers.md",
                "Projective space" => "manifolds/projectivespace.md",
                "Orthogonal and Unitary Matrices" => "manifolds/generalunitary.md",
                "Rotations" => "manifolds/rotations.md",
                "Segre" => "manifolds/segre.md",
                "Shape spaces" => "manifolds/shapespace.md",
                "Skew-Hermitian matrices" => "manifolds/skewhermitian.md",
                "Spectrahedron" => "manifolds/spectrahedron.md",
                "Sphere" => "manifolds/sphere.md",
                "Stiefel" => "manifolds/stiefel.md",
                "Symmetric matrices" => "manifolds/symmetric.md",
                "Symmetric positive definite" => "manifolds/symmetricpositivedefinite.md",
                "SPD, fixed determinant" => "manifolds/spdfixeddeterminant.md",
                "Symmetric positive semidefinite fixed rank" => "manifolds/symmetricpsdfixedrank.md",
                "Symplectic Grassmann" => "manifolds/symplecticgrassmann.md",
                "Symplectic matrices" => "manifolds/symplectic.md",
                "Symplectic Stiefel" => "manifolds/symplecticstiefel.md",
                "Torus" => "manifolds/torus.md",
                "Tucker" => "manifolds/tucker.md",
                "Unit-norm symmetric matrices" => "manifolds/spheresymmetricmatrices.md",
            ],
            "Combined manifolds" => [
                "Fiber bundle" => "manifolds/fiber_bundle.md",
                "Graph manifold" => "manifolds/graph.md",
                "Power manifold" => "manifolds/power.md",
                "Product manifold" => "manifolds/product.md",
                "Vector bundle" => "manifolds/vector_bundle.md",
            ],
            "Manifold decorators" => [
                "Connection manifold" => "manifolds/connection.md",
                "Metric manifold" => "manifolds/metric.md",
            ],
        ],
        "Features on Manifolds" => [
            "Atlases and charts" => "features/atlases.md",
            "Differentiation" => "features/differentiation.md",
            "Distributions" => "features/distributions.md",
            "Integration" => "features/integration.md",
            "Statistics" => "features/statistics.md",
            "Utilities" => "features/utilities.md",
        ],
        "Miscellanea" => [
            "About" => "misc/about.md",
            "Changelog" => "misc/NEWS.md",
            "Contributing" => "misc/CONTRIBUTING.md",
            "Internals" => "misc/internals.md",
            "Test suite" => "misc/testsuite.md",
            "Notation" => "misc/notation.md",
            "References" => "misc/references.md",
        ],
    ],
    plugins = [bib, links],
)
deploydocs(repo = "github.com/JuliaManifolds/Manifolds.jl.git", push_preview = true)
#back to main env
Pkg.activate()
