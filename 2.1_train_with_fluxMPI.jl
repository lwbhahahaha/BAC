using Pkg
Pkg.activate("/home/molloi-lab/Desktop/Project BAC/BAC project/libs/")
using Lux, Random, NNlib, Zygote, LuxCUDA, CUDA, FluxMPI, JLD2, DICOM
using Images
using MLUtils
using MPI
using Optimisers
using ImageMorphology, ChainRulesCore, Statistics

CUDA.allowscalar(false)

patch_size = 256
patch_size_half = round(Int, patch_size/2);

FluxMPI.Init()

# Rank(similar to threadID) of the current process.
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
dev = gpu_device()
dev_cpu = cpu_device()

_conv = (in, out) -> Conv((3, 3), in=>out, pad=1)

conv1 = (in, out) -> Chain(_conv(in, out), BatchNorm(out, leakyrelu))
conv2 = (in, out) -> Chain(Conv((1, 1), in=>out), sigmoid)

_tran = (in, out) -> ConvTranspose((2, 2), in => out, stride = 2)
tran = (in, out) -> Chain(_tran(in, out), BatchNorm(out, leakyrelu))

my_cat = (x, y) -> cat(x, y; dims=Val(3))

function unet2D(in_chs, lbl_chs)    
    # Contracting layers
    l1 = Chain(conv1(in_chs, 64), conv1(64, 64))
    l2 = Chain(l1, MaxPool((2,2), stride=2), conv1(64, 128), conv1(128, 128))
    l3 = Chain(l2, MaxPool((2,2), stride=2), conv1(128, 256), conv1(256, 256))
    l4 = Chain(l3, MaxPool((2,2), stride=2), conv1(256, 512), conv1(512, 512))
    l5 = Chain(l4, MaxPool((2,2), stride=2), conv1(512, 1024), conv1(1024, 1024), tran(1024, 512))
    
    # Expanding layers
    l6 = Chain(Parallel(my_cat,l5,l4), conv1(512+512, 512), conv1(512, 512), tran(512, 256))
    l7 = Chain(Parallel(my_cat,l6,l3), conv1(256+256, 256), conv1(256, 256), tran(256, 128))
    l8 = Chain(Parallel(my_cat,l7,l2), conv1(128+128, 128), conv1(128, 128), tran(128, 64))
    l9 = Chain(Parallel(my_cat,l8,l1), conv1(64+64, 64), conv1(64, 64), conv2(64, lbl_chs))
end

function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function bce_dice_and_hd_loss(ŷ,  y, epoch; ϵ=1f-10)
    num_batches = size(ŷ)[4]
    hd_weight = max(0, epoch-5)*1f-3

    # dice
    @inbounds loss_dice = 
        1f0 - (muladd(2f0, sum(ŷ .* y), ϵ) / (sum(ŷ .^ 2) + sum(y .^ 2) + ϵ))

    # binarycrossentropy
    bce_loss = mean(@. -xlogy(y, ŷ .+ ϵ) - xlogy(1f0 .- y, 1f0 .- ŷ .+ ϵ))

    # HD
    ŷ_dtm = Array{Float32, 4}(undef, 256, 256, 1, num_batches)
    y_dtm = Array{Float32, 4}(undef, 256, 256, 1, num_batches)
    ŷ_cpu = ŷ |> dev_cpu
    y_cpu = y |> dev_cpu
    ignore_derivatives() do
        for chan_idx = 1:1
            for batch_idx = 1 : num_batches
                ŷ_cpu_round = round.(ŷ_cpu[:,:, chan_idx, batch_idx])
                if sum(ŷ_cpu_round) > 0f0
                    ŷ_dtm[:,:, chan_idx, batch_idx] = 
                    distance_transform(feature_transform(Bool.(ŷ_cpu_round)))
                else
                    ŷ_dtm[:,:, chan_idx, batch_idx] = fill(1f3, (256, 256))
                end
                if sum(y_cpu[:,:, chan_idx, batch_idx]) > 0f0
                    y_dtm[:,:, chan_idx, batch_idx] = 
                    distance_transform(feature_transform(Bool.(round.(y_cpu[:,:, chan_idx, batch_idx]))))
                else
                    y_dtm[:,:, chan_idx, batch_idx] = fill(1f3, (256, 256))
                end
            end
        end
    end
    loss_hd = mean(((ŷ_cpu .- y_cpu) .^ 2) .* (ŷ_dtm .^ 4 .+ y_dtm .^ 4))

    # summary
    loss_total = loss_dice + bce_loss + loss_hd * hd_weight * 1f-8
    FluxMPI.fluxmpi_println("epoch#$epoch: DICE = $loss_dice, BCE = $bce_loss, HD = $loss_hd")
    return loss_total
end

function compute_loss(x, y, model, ps, st, epoch)
    ŷ, st = model(x, ps, st)
    loss = bce_dice_and_hd_loss(ŷ, y, epoch)
    return loss, ŷ, st
end

function auto_continue_training()
    i = 0
    while isfile("saved_train_info_$i.jld2")
        i += 1
    end
    return max(0, i-1)
end

function train(start_epoch_idx, epoch_target, ps, st, opt, st_opt, model, train_loader)
    st_opt = FluxMPI.synchronize!(st_opt; root_rank = 0)
    FluxMPI.fluxmpi_println("Start training...")
    for epoch in start_epoch_idx : epoch_target
        for (x_cpu, y_cpu) in train_loader
            global ps, st, st_opt
            x, y = x_cpu |> dev, y_cpu |> dev
            
            (loss, _, st), back = pullback(p -> compute_loss(x, y, model, p, st, epoch), ps)
            gs = back((one(loss), nothing, nothing))[1]

            st_opt, ps = Optimisers.update(st_opt, ps, gs)
        end
        FluxMPI.fluxmpi_println("============================================================")
        if rank == 0
            global ps, st
            local ps_save, st_save = ps |> dev_cpu, st |> dev_cpu
            @save "saved_train_info_$epoch.jld2" ps_save st_save
        end
    end
end

# read data
if rank == 0
    @load "train_loader_1_small.jld2" train_loader
    global train_loader_ = train_loader
end
if rank == 1
    @load "train_loader_2_small.jld2" train_loader
    global train_loader_ = train_loader
end
if rank == 2
    @load "train_loader_3_small.jld2" train_loader
    global train_loader_ = train_loader
end
if rank == 3
    @load "train_loader_4_small.jld2" train_loader
    global train_loader_ = train_loader
end

# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)
l_r = 0.001
model = unet2D(1, 1)

start_epoch_idx = auto_continue_training() # Replace 0 with other epoch idx if training on a saved model
epoch_target = 500 # end epoch idx
if start_epoch_idx == 0
    # train new model 
    ps, st = Lux.setup(rng, model)

    ps = ps |> dev
    st = st |> dev

    ps = FluxMPI.synchronize!(ps; root_rank = 0)
    st = FluxMPI.synchronize!(st; root_rank = 0)

    opt = DistributedOptimizer(NAdam(l_r))
    st_opt = Optimisers.setup(opt, ps)

    if rank == 0
        local ps_save, st_save = ps |> dev_cpu, st |> dev_cpu
        @save "saved_train_info_0.jld2" ps_save st_save
    end
    train(start_epoch_idx+1, epoch_target, ps, st, opt, st_opt, model, train_loader_)
else
    # load saved model 
    @load "saved_train_info_$start_epoch_idx.jld2" ps_save st_save
    ps = ps_save |> dev
    st = st_save |> dev

    opt = DistributedOptimizer(NAdam(l_r))
    st_opt = Optimisers.setup(opt, ps)

    train(start_epoch_idx+1, epoch_target, ps, st, opt, st_opt, model, train_loader_)
end


