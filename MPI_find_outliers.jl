using Pkg
Pkg.activate("/home/molloi-lab/Desktop/Project BAC/BAC project/libs/")

using Lux, Random, NNlib, Zygote, LuxCUDA, CUDA, FluxMPI, JLD2, DICOM
using Images
using MLUtils
using MPI
using Optimisers
using ImageMorphology, ChainRulesCore, Statistics
using Plots

CUDA.allowscalar(false)

data_dir = "/media/molloi-lab/2TB/Clean_Dataset_full"
saved_model_dir = "."
output_pred_dir = "/media/molloi-lab/1TB/Output"
output_images_dir = "/media/molloi-lab/1TB/images_Output"

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
# conv2 = (in, out) -> Chain(_conv(in, out), x -> softmax(x; dims = 3))
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

function zoom_pxiel_values(img)
    a, b = minimum(img), maximum(img)
    img_ = (img .- a) ./ (b - a)
    return img_
end

function normalize_img(img)
    m = maximum(img)
    img = m .- img
    a = mean(img)
    s = std(img)
    img = (img .- a) ./ s 
    # println("mean = $(mean(img)), std = $(std(img))")
    return img
end

function patch_image(img, mask, num_patches; thd = 0.666)
    # img_max = maximum(img)
    # img = img_max .- img
    img = normalize_img(img)
    s = size(img)
    # @info "num_patches = $num_patches"
    x = ceil(Int, s[1]/patch_size) + floor(Int, (s[1]-patch_size_half)/patch_size)
    y = ceil(Int, s[2]/patch_size) + floor(Int, (s[2]-patch_size_half)/patch_size)

    # img_patches = Array{Float32, 4}(undef, patch_size, patch_size, 1, num_patches)
    # locations = Array{Tuple{Int64, Int64, Int64, Int64}, 1}(undef, num_patches)
    img_patches = []
    locations = []
    ct = 0
    for i = 1 : x-1
        x_start = 1+(i-1)*patch_size_half
        x_end = x_start+patch_size-1
        for j = 1 : y-1
            y_start = 1+(j-1)*patch_size_half
            y_end = y_start+patch_size-1
            # check patch
            if mean(mask[x_start:x_end, y_start:y_end]) > thd
                # save patch
                ct += 1
                push!(img_patches, img[x_start:x_end, y_start:y_end])
                push!(locations, (x_start, x_end, y_start, y_end))
                # img_patches[:, :, 1, ct] = img[x_start:x_end, y_start:y_end]
                # locations[ct] = (x_start, x_end, y_start, y_end)
            end
        end
        # right col
        y_start, y_end = s[2]-patch_size+1, s[2]
        # check patch
        if mean(mask[x_start:x_end, y_start:y_end]) > thd
            # save patch
            ct += 1
            push!(img_patches, img[x_start:x_end, y_start:y_end])
            push!(locations, (x_start, x_end, y_start, y_end))
            # img_patches[:, :, 1, ct] = img[x_start:x_end, y_start:y_end]
            # locations[ct] = (x_start, x_end, y_start, y_end)
        end
    end
    # last row
    x_start, x_end = s[1]-patch_size+1, s[1]
    for j = 1 : y-1
        y_start = 1+(j-1)*patch_size_half
        y_end = y_start+patch_size-1
        # check patch
        if mean(mask[x_start:x_end, y_start:y_end]) > thd
            # save patch
            ct += 1
            push!(img_patches, img[x_start:x_end, y_start:y_end])
            push!(locations, (x_start, x_end, y_start, y_end))
            # img_patches[:, :, 1, ct] = img[x_start:x_end, y_start:y_end]
            # locations[ct] = (x_start, x_end, y_start, y_end)
        end
    end
    # right col
    y_start, y_end = s[2]-patch_size+1, s[2]
    # check patch
    if mean(mask[x_start:x_end, y_start:y_end]) > thd
        # save patch
        ct += 1
        push!(img_patches, img[x_start:x_end, y_start:y_end])
        push!(locations, (x_start, x_end, y_start, y_end))
        # img_patches[:, :, 1, ct] = img[x_start:x_end, y_start:y_end]
        # locations[ct] = (x_start, x_end, y_start, y_end)
    end
    # return
    return img_patches, locations, ct
end

function pred_img(img_in, model_info, num_patches)
    # get binary mask
    # @info "Getting binary mask..."
    img = deepcopy(img_in)
    img_out = zeros(Float32, size(img))
    img_mask = 1 .- round.(zoom_pxiel_values(img_in))
    
    # patch img_in
    # @info "Patching image..."
    img_patches, locations, ct = patch_image(img, img_mask, num_patches)
    pred_patches = Array{Float32, 3}(undef, patch_size, patch_size, ct)

    # load model
    # @info "Loading model..."
    model, ps, st = model_info
    
    # apply model on all patches
    # @info "Running predictions..."
    for i = 1 : ct
        curr_patch = (reshape(img_patches[i], 256, 256, 1, 1)) |> dev # 256*256*1*batchsize
        curr_pred = model(curr_patch, ps, st)[1] |> cpu # 256*256*1*batchsize
        pred_patches[:, :, i] = curr_pred[:,:,1,:]
    end

    # combines patch to pred_out
    # @info "Gathering output"
    Threads.@threads for i = 1 : ct
        x_start, x_end, y_start, y_end = locations[i]
        for x = x_start : x_end
            for y = y_start : y_end
                if pred_patches[x-x_start+1, y-y_start+1, i] >= 0.5
                    img_out[x,y] = 1f0
                end
            end
        end
    end
    
    return img_out
end


# Run through all images(runtime ~= 37min for 200 images)
GC.gc(true)
epoch_idx = 55

# load model
model_name = "saved_train_info_$epoch_idx.jld2"
model_path = joinpath(saved_model_dir, model_name)
@load model_path ps_save st_save
ps = ps_save |> dev
st_ = st_save |> dev
st = Lux.testmode(st_)
model_info = (unet2D(1, 1), ps, st)

SIDs = readdir(data_dir)
s = size(SIDs)[1]
for i =  rank+1 : 4 : s
    printed = false
    sid = SIDs[i]
    curr_dir = joinpath(data_dir, sid)
    if isdir(curr_dir)
        for f in readdir(curr_dir)
            f_name, f_ext = splitext(f)
            if f_ext == ".dcm"
                # set up path
                output_dir = joinpath(output_pred_dir, sid)
                isdir(output_dir) || mkdir(output_dir)
                output_path = joinpath(output_dir, f_name*".pred.png")
                if !isfile(output_path)
                    if !printed
                        printed = true
                        FluxMPI.fluxmpi_println(i)
                    end
                    # process image
                    a,b = split(rsplit(f_name, '.'; limit = 2)[2], '_')
                    num_patches = parse(Int64, a) + parse(Int64, b)
                    img_path = joinpath(curr_dir, f)
                    lbl_path = joinpath(curr_dir, f_name*".png")
                    lbl = Float32.(Images.load(lbl_path))
                    img = Float32.(dcm_parse(img_path)[(0x7fe0, 0x0010)])
                    pred = pred_img(img, model_info, num_patches)
                    # save
                    save(output_path, pred)
                end
            end
        end
    end
end