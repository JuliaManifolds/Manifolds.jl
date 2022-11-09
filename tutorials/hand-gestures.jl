using Manifolds
using CSV
using DataFrames
using Plots
using MultivariateStats

function load_hands()
    hands_url = "https://raw.githubusercontent.com/geomstats/geomstats/master/geomstats/datasets/data/hands/hands.txt"
    hand_labels_url = "https://raw.githubusercontent.com/geomstats/geomstats/master/geomstats/datasets/data/hands/labels.txt"

    hands = Matrix(CSV.read(download(hands_url), DataFrame, header=false))
    hands = reshape(hands, size(hands, 1), 3, 22)
    hand_labels = CSV.read(download(hand_labels_url), DataFrame, header=false).Column1
    return hands, hand_labels
end

function hand_analysis()
    # Let's load dataset of hand gestures, described here: https://geomstats.github.io/notebooks/14_real_world_applications__hand_poses_analysis_in_kendall_shape_space.html
    hands, hand_labels = load_hands()

    # This plots a sample gesture as a 3D scatter plot of points.
    scatter3d(hands[1, 1, :], hands[1, 2, :], hands[1, 3, :])

    # Each gesture is represented by 22 landmarks in ℝ³, so we use the appropriate
    # Kendall's shape space
    Mshape = KendallsShapeSpace(3, 22)
    # Hands read from the dataset are projected to the shape space to remove translation
    # and scaling variability. Rotational variability is then handled using quotient
    # structure of KendallsShapeSpace
    hands_projected = [project(Mshape, hands[i, :, :]) for i in axes(hands, 1)]

    # Now let's do tangent space PCA. This starts with computing a mean point and computing
    # logithmic maps at mean to each point in the dataset.
    mean_hand = mean(Mshape, hands_projected)
    hand_logs = [log(Mshape, mean_hand, p) for p in hands_projected]
    # For tangent PCA, we need coordinates in a basis.
    # Some libraries skip this step because the representation of tangent vectors
    # forms a linear subspace of an Euclidean space so PCA automatically detects
    # which directions have no variance but this is a more principled way to solve
    # this issue.
    B = get_basis(Mshape, mean_hand, ProjectedOrthonormalBasis(:svd))
    hand_log_coordinates = [get_coordinates(Mshape, mean_hand, X, B) for X in hand_logs]
    
    # This code prepares data for MultivariateStats -- mean=0 is set because we've centered
    # the data geometrically to mean_hand in the code above.
    red_coords = reduce(hcat, hand_log_coordinates)
    fp = fit(PCA, red_coords; mean=0)
    # Now let's show explained variance of each principal component.
    plot(principalvars(fp), title="explained variance", label="Tangent PCA")

    # Also, let's look at how projections on the first two pricipal components look like.

    fig = plot(; title="coordinates per gesture of the first two principal components")
    for label_num in [0, 1]
        mask = hand_labels .== label_num
        cur_hand_logs = red_coords[:, mask]
        cur_t = MultivariateStats.transform(fp, cur_hand_logs)
        scatter!(fig, cur_t[1, :], cur_t[2, :], label="gesture " * string(label_num))
    end
    xlabel!(fig, "principal component 1")
    ylabel!(fig, "principal component 2")
    fig

    # Now let's compute pairwise distances between gestures
    # we can use them for clustering, classification, etc.
    hand_distances = [
        distance(Mshape, hands_projected[i], hands_projected[j]) for
        i in eachindex(hands_projected), j in eachindex(hands_projected)
    ]
    return heatmap(hand_distances, aspect_ratio=:equal)
end
