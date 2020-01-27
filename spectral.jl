# eigenvector centrality for reconstruction of zebrafish hemi-hindbrain
# Centrality is defined as the product of left and right principal eigenvectors
# written by Sebastian Seung for Julia 1.3

using MAT, Plots, LinearAlgebra, SparseArrays, DataFrames, CSV

function powermethod(A)
    # power method for finding principal eigenvector of A
    niter = 1000
    AA = Float32.(A)
    u = rand(Float32, size(A, 1))
    for iter = 1:niter
        u = AA*u
        u /= norm(u)
    end
    return u
end

# W_ij is number of synapses for i <- j connection.  i is Post and j is Pre
fname = matopen("ConnMatrixPre.mat")
W = read(fname, "ConnMatrixPre")
close(fname)
W = Int.(W)

# dictionary keys are cell classes
# values are arrays of Eyewire IDs
fname = matopen("cellIDs.mat")
cellIDs = read(fname, "cellIDs")
close(fname)
for key in keys(cellIDs)
    cellIDs[key] = Int.(cellIDs[key])[:]  # turn into vector of integers
end

fname = matopen("AllCells.mat")
AllCells = read(fname, "AllCells")
close(fname)
AllCells = round.(Int, vec(AllCells))

integratorinds = findall(map(s -> occursin("Integrator", s), CellClass))
abdminds = findall(map(s -> occursin("Abducens_M", s), CellClass))
abdiinds = findall(map(s -> occursin("Abducens_I", s), CellClass))

# zero outgoing connections from ABD as they are partner assignment errors
W[:, abdminds] .= 0  
W[:, abdiinds] .= 0

# IDs of r78contra dendrites (unique). Each dendrite should have no outgoing synapses
dendrites = [76199, 76200, 76182, 76183, 76185, 76186, 76189, 76191, 76188, 77582, 77605, 79040, 76399, 76828, 76829, 76826, 76289, 76542, 76832, 76838, 76877]
# best matches with orphan axons (some are repeated). Each axon should have no incoming synapses
axons = [78687, 78651, 76666, 80219, 76666, 76666, 78677, 79950, 77869, 80219, 79134, 78923, 76675, 78903, 81682, 80241, 80248, 78615, 77773, 78615, 80242]

# synthesize bilateral model
# mirror so there are twice as many segments
# glue together axons and dendrites from opposite sides (correspondences above)
# orphan axons should have zero incoming synapses to start with
# this will remove their outgoing synapses too
Wipsi = copy(W)
Crossing = zeros(Int, size(W))    # synthesize connections crossing the midline
for (dendriteID, axonID) in zip(dendrites, axons)
    dendriteind = findall(AllCells .== dendriteID)
    axonind = findall(AllCells .== axonID)
    # transfer outgoing connections from axon to corresponding dendrite on other side
    Crossing[:, dendriteind] = Wipsi[:, axonind]
    # zero out outgoing connections from axon
    Wipsi[:, axonind] .= 0
end

# This is the 2n x 2n connection matrix for the bilateral model
Wbilateral = [Wipsi Crossing; Crossing Wipsi]

#vr = powermethod(Wbilateral)
#vl = powermethod(Wbilateral')

# principal eigenvector is last one since eigenvalues are sorted
vr = real(eigvecs(float(Wbilateral))[:,end])
vl = real(eigvecs(float(Wbilateral'))[:,end])

# eigenvector centrality
centrality = vl.*vr
# choose sign to make centrality nonnegative
if any( centrality .< -1e-6 )  # could be negative entries due to roundoff error
    centrality = - centrality
end

n = size(W, 1)  # number of neurons without mirroring
ind = sortperm(centrality[1:n], rev=true)  # sort centralities on one side only

# various synapse numbers
tointegrator = vec(sum(Wbilateral[integratorinds, :], dims=1))
fromintegrator = vec(sum(Wbilateral[:, integratorinds], dims=2))
fanout = vec(sum(Wbilateral, dims=1))
fanin = vec(sum(Wbilateral, dims=2))
toabdm = vec(sum(Wbilateral[abdminds, :], dims=1))
toabdi = vec(sum(Wbilateral[abdiinds, :], dims=1))

CSV.write("ranking.csv", DataFrame([ind [AllCells fanout[1:n] fanin[1:n] tointegrator[1:n] fromintegrator[1:n] toabdm[1:n] CellClass centrality[1:n]][ind,:]]), writeheader = false)
