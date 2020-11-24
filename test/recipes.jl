using RecipesBase, VisualRegressionTests, Plots
include("utils.jl")
# Note that the `false` in the test avoids to start Gtk
# I (kellertuer)  think it also avoids asking whether a new reference should be
# created but I am not 100% sure

@testset "Recipes Test" begin
    function Hyp2PB_plot()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0,0.0], [0.0,1.0]])
        p2 = convert.( Ref(PoincareBallPoint), p)
        plot(M,p2)
    end
    @plottest Hyp2PB_plot "assets/Hyp2PBPlot.png" false

    function Hyp2PB_quiver()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0,0.0], [0.0,0.0], [0.0,1.0]])
        p2 = convert.( Ref(PoincareBallPoint), p)
        X = [ log(M,p2[2],p2[1]), log(M,p2[1],p2[3]) ]
        plot(M,[p2[2],p2[1]], X)
    end
    @plottest Hyp2PB_quiver "assets/Hyp2PBQuiver.png" false

    function Hyp2PH_plot()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0,0.0], [0.0,1.0]])
        p2 = convert.( Ref(PoincareHalfSpacePoint), p)
        plot(M,p2)
    end
    @plottest Hyp2PH_plot "assets/Hyp2PHPlot.png" false

    function Hyp2PH_quiver()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0,0.0], [0.0,0.0], [0.0,1.0]])
        p2 = convert.( Ref(PoincareHalfSpacePoint), p)
        X = [ log(M,p2[2],p2[1]), log(M,p2[1],p2[3]) ]
        plot(M,[p2[2],p2[1]], X)
    end
    @plottest Hyp2PH_quiver "assets/Hyp2PHQuiver.png" false
end