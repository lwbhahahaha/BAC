### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 18f6276e-1531-11ee-151f-357a93fee464
begin
	using Pkg
	Pkg.activate("libs/")
end

# ╔═╡ 900f4615-5ffa-46cf-b4c4-ef35d77e3fd1
begin
	using CSV
	using JLD2
	using CUDA
	using Glob
	using Dates
	using DICOM
	using Images
	using CondaPkg
	using ImageView
	using ImageDraw
	using Statistics
	using DataFrames
	using PythonCall
	using StaticArrays
	using MLDataPattern
	using ChainRulesCore
	using FastAI, FastVision, Flux, Metalhead
	import CairoMakie; CairoMakie.activate!(type="png")
end

# ╔═╡ 84e9d89b-d43d-4eed-9eaf-ea9a9c699a2c
md"""
# Create black masks for all images.
"""

# ╔═╡ 000580e2-394d-42ae-a107-17607f3a5c94
@load "clean_set.jld2" train_set valid_set

# ╔═╡ 6b139a97-6612-4db6-b36e-47912fedc6e6
begin
	Threads.@threads for i = 1 : size(train_set)[1]
		for j = 1 : 4
			# Read dicom image
			dcm_data = dcm_parse(train_set[i][1][j])
			img, UID = dcm_data[(0x7fe0, 0x0010)], dcm_data[(0x0008,0x0018)]
			# Create pure black images.
			lbl = zeros(Gray{N0f8}, size(img))
			# Set png path
			splited = split(train_set[i][1][j], "\\")
			splited[1] *= "\\"
			splited[5] = "label"
			splited[7] = splited[7][1:end-3]*"png"
			png_path = joinpath(splited)
			# save png
			save(png_path, lbl)
			# Change path to roi
			train_set[i][2][j] = png_path
		end
	end
end

# ╔═╡ f3aedd03-ff76-401f-8ac7-fd0e67e313f4
begin
	Threads.@threads for i = 1 : size(valid_set)[1]
		for j = 1 : 4
			# Read dicom image
			dcm_data = dcm_parse(valid_set[i][1][j])
			img, UID = dcm_data[(0x7fe0, 0x0010)], dcm_data[(0x0008,0x0018)]
			# Create pure black images.
			lbl = zeros(Gray{N0f8}, size(img))
			# Set png path
			splited = split(valid_set[i][1][j], "\\")
			splited[1] *= "\\"
			splited[5] = "label"
			splited[7] = splited[7][1:end-3]*"png"
			png_path = joinpath(splited)
			# save png
			save(png_path, lbl)
			# Change path to roi
			valid_set[i][2][j] = png_path
		end
	end
end

# ╔═╡ f8c0a0ff-a396-4a06-85b7-48742f8ca5f1
@save "clean_set_step2.jld2" train_set valid_set

# ╔═╡ Cell order:
# ╠═18f6276e-1531-11ee-151f-357a93fee464
# ╠═900f4615-5ffa-46cf-b4c4-ef35d77e3fd1
# ╟─84e9d89b-d43d-4eed-9eaf-ea9a9c699a2c
# ╠═000580e2-394d-42ae-a107-17607f3a5c94
# ╠═6b139a97-6612-4db6-b36e-47912fedc6e6
# ╠═f3aedd03-ff76-401f-8ac7-fd0e67e313f4
# ╠═f8c0a0ff-a396-4a06-85b7-48742f8ca5f1
