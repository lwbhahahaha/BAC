### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ e6d23e80-0bd2-11ee-1785-73e34754591b
# run with julia 1.9.1
begin
	using Pkg
	Pkg.activate("libs/")
	Pkg.instantiate()
	Pkg.add("DICOM")
	Pkg.add("XLSX")
	Pkg.add("Plots")
end

# ╔═╡ b98df1ab-ca99-4802-acc2-02fffe8d4284
begin
	Pkg.add("JLD2")
	using JLD2
end

# ╔═╡ 268ceef5-2c2d-475b-b18d-246da9f2c9c0
begin
	using DICOM
	using XLSX
	using Plots
end

# ╔═╡ 7975a099-49d4-496f-bbd9-ae6c9c90e507
html"""<style>
main {
    max-width: 1500px;
}
"""

# ╔═╡ 80335714-4f14-4e4b-b944-23a3041272e7
# Set global path vars here.
begin
	BAC_dir = raw"C:\BAC_dataset"
	valid_data_dir = raw"C:\BAC_dataset\collected_dataset_for_ML\valid"
	train_data_dir = raw"C:\BAC_dataset\collected_dataset_for_ML\train"
end;

# ╔═╡ f11b0ed3-8ad8-4677-b663-972849f9387a
#=
	- format for valid_set and train_set:

		idx = 1: 4 images' paths
		idx = 2: 4 labels' paths
		idx = 3: 4 UIDs
		idx = 4: SID
		idx = 5: L_ave_mass
		idx = 6: R_ave_mass
		idx = 7: sum_mass
		idx = 8: 4 masses

	- Index within arrays:
		"L CC" => 1, "L MLO" => 2, "R CC" => 3, "R MLO" => 4
=#

# ╔═╡ ff43e82f-61bc-4082-b14d-8f3162435349
"""
	This function reads the excel file(Ca mass calculation Final.xlsx) and retrives data.
"""
function read_excel_and_build_data()
	println("Reading the excel...")
	# Read
	xf = XLSX.readxlsx(raw"C:\Users\wenbl13\OneDrive - UCI Health\Desktop\BAC project\Ca mass calculation Final.xlsx")
	SID = xf["Final!C:C"][2:end]
	L1_mass = xf["Final!O:O"][2:end] # L CC mass
	L2_mass = xf["Final!S:S"][2:end] # L MLO mass
	R1_mass = xf["Final!F:F"][2:end] # R CC mass
	R2_mass = xf["Final!J:J"][2:end] # R MLO mass
	L_mass = xf["Final!T:T"][2:end]
	R_mass = xf["Final!K:K"][2:end]
	sum_mass = xf["Final!U:U"][2:end]
	xf = nothing
	image_ct = size(SID)[1]
	println("\tFound $image_ct patients...")

	# Analysis
	println("\nAnalyzing...")
	num_intervals, interval = 5, 20
	interval_count = zeros(Int, num_intervals)
	data_groups = Array{Any}(undef, num_intervals)
	Threads.@threads for i = 1:num_intervals
		data_groups[i] = [] # init data_groups
	end
	for i = 1 : image_ct
		mass = sum_mass[i]
		idx = min(num_intervals, floor(Int, mass/interval)+1)
		interval_count[idx] += 1
		push!(data_groups[idx], [Array{String}(undef, 4), Array{String}(undef, 4), Array{String}(undef, 4), SID[i], L_mass[i], R_mass[i], sum_mass[i], [L1_mass[i], L2_mass[i], R1_mass[i], R2_mass[i]]])
	end
	
	for i = 1 : num_intervals
		println("\t$((i-1)*interval) ≤ mass$(i<num_intervals ? " < $(i*interval):  \t" : ":\t\t\t")$(interval_count[i]) patients")
	end

	# Candidating validation dataset
	println("\nCandidating validation set...")
	validation_size = 50
	target_each_interval = ceil(Int, validation_size/num_intervals)
	valid_set = []
	println("\tValidation set(Total $validation_size) patients:")
	for i = 1:num_intervals
		ct = 0
		flag = false
		if target_each_interval > interval_count[i]
			# supersampling
			t = floor(Int, target_each_interval/interval_count[i])
			for supersampling_times = 1 : t
				for data in data_groups[i]
					if data != nothing
						push!(valid_set, data)
					end
				end
			end
			ct = t*interval_count[i]
			flag = true
		end
		picked_idx = []
		while ct < target_each_interval
			idx = rand(1:interval_count[i])
			if idx ∉ picked_idx
				push!(picked_idx, idx)
				push!(valid_set, deepcopy(data_groups[i][idx]))
				data_groups[i][idx] = nothing
				ct += 1
			end
		end
		interval_count[i] -= ct
		println("\t\t$((i-1)*interval) ≤ mass$(i<num_intervals ? " < $(i*interval):  \t" : ":\t\t\t")$(ct) patients$(flag ? "(Supersampled)" : "")")
	end
	
	# Candidating training dataset
	println("\nCandidating training set...")
	training_size = 300
	target_each_interval = ceil(Int, training_size/num_intervals)
	train_set = []
	println("\tTraining set(Total $training_size) patients:")
	for i = 1:num_intervals
		ct = 0
		flag = false
		if target_each_interval > interval_count[i]
			# supersampling
			t = floor(Int, target_each_interval/interval_count[i])
			for supersampling_times = 1 : t
				for data in data_groups[i]
					if data != nothing
						push!(train_set, data)
					end
				end
			end
			ct = t*interval_count[i]
			flag = true
		end
		picked_idx = []
		while ct < target_each_interval
			idx = rand(1:size(data_groups[i])[1])
			if idx ∉ picked_idx && data_groups[i][idx] != nothing
				push!(picked_idx, idx)
				push!(train_set, data_groups[i][idx])
				ct += 1
			end
		end
		println("\t\t$((i-1)*interval) ≤ mass$(i<num_intervals ? " < $(i*interval):  \t" : ":\t\t\t")$(ct) patients$(flag ? "(Supersampled)" : "")")
	end
	return valid_set, train_set
end

# ╔═╡ 849b3530-f419-4e22-a0d8-7a622f04f6b8
"""
	This function recursively search folders for files. This function uses multithreads to speed up.
"""
function rec_search(root_dir, mode)
	files=[]
	
	# collect curr folder info
	sub_files, sub_dirs = [], []
	for f in readdir(root_dir)
		curr_path = joinpath(root_dir, f)

		if isdir(curr_path)
			push!(sub_dirs, curr_path)
		else
			push!(sub_files, curr_path)
		end
	end
	dir_ct = size(sub_dirs)[1]
	files_found = Array{Any}(undef, dir_ct)
	
	# folders: rec calls
	Threads.@threads for i = 1 : dir_ct
		files_found[i] = rec_search(sub_dirs[i], mode)
	end
	
	# Append after multithreading to prevent data races
	for i = 1 : dir_ct
		append!(files, files_found[i])
	end

	# files: check if it's DICOM image
	for f in sub_files
		 
		if f[end-3:end] == (mode == "label" ? ".roi" : ".dcm")
			push!(files, f)
		end
	end
	
	return files
end

# ╔═╡ 76c957f4-cb09-4fa8-bdd0-f1835319ccfc
"""
	This function recursively search folders for `file_name`. This function uses multithreads to speed up. Helper function for `search_for_specific_file`
"""
function rec_search_for_specific_file(root_dir, file_name)
	# collect curr folder info
	sub_files, sub_dirs = [], []
	for f in readdir(root_dir)
		curr_path = joinpath(root_dir, f)

		if isdir(curr_path)
			push!(sub_dirs, curr_path)
		else
			push!(sub_files, curr_path)
		end
	end
	dir_ct = size(sub_dirs)[1]
	files_found = Array{Any}(undef, dir_ct)
	
	# folders: rec calls
	Threads.@threads for i = 1 : dir_ct
		rec_search_for_specific_file(sub_dirs[i], file_name)
	end

	# files: check if it's target_file
	for f in sub_files
		if rsplit(f, "\\"; limit = 2)[2] == file_name
			@info f
		end
	end
end

# ╔═╡ 1adb1c22-6e59-47bd-b38e-dcd3e5aecde8
"""
	This function search for `file_name`.
"""
function search_for_specific_file(file_name)
	rec_search_for_specific_file(joinpath(BAC_dir, "Patientstudy_BAC_2013a", "data_from_KP"), file_name)
	rec_search_for_specific_file(joinpath(BAC_dir, "Patientstudy_BAC_2013b", "data_from_KP"), file_name)
	rec_search_for_specific_file(joinpath(BAC_dir, "Patientstudy_BAC_2014", "data_from_KP"), file_name)
	rec_search_for_specific_file(joinpath(BAC_dir, "Patientstudy_BAC_2015a", "data_from_KP"), file_name)
	rec_search_for_specific_file(joinpath(BAC_dir, "Patientstudy_BAC_2015b", "data_from_KP"), file_name)
	rec_search_for_specific_file(joinpath(BAC_dir, "Patientstudy_BAC_2016", "Data from KP"), file_name)
end

# ╔═╡ 70bd3f84-97f4-4618-9893-aa7e55481ea8
"""
	This function deals with the file system and locates all the images.
"""
function Locate_images()
	println("\nLocating files...")
	
	# shared_drive_BAC_dir = raw"Z:\Research Projects\BAC"
	# # Check connection to shared drive
	# isdir(shared_drive_BAC_dir) || (printstyled("Error: Failed to establish a connection with the shared drive: $shared_drive_BAC_dir.\n"; color = :red);)

	# Locate year dirs
	data_from_kp_dict = Dict()
	year_dir = ["Patientstudy_BAC_2013a", "Patientstudy_BAC_2013b", "Patientstudy_BAC_2014", "Patientstudy_BAC_2015a", "Patientstudy_BAC_2015b", "Patientstudy_BAC_2016"]
	for year_path in year_dir
		isdir(joinpath(BAC_dir, year_path)) || (printstyled("\tError: Failed to locate the folder: $year_path.\n"; color = :red);return)
		print("\tFound \"")
		printstyled(year_path; color = :yellow)
		print("\"; ")
		data_from_kp = ""
		for data_from_kp_ in readdir(joinpath(BAC_dir, year_path))
			data_from_kp_splited = split(lowercase(data_from_kp_), [' ', '_'])
			if "data" in data_from_kp_splited && "kp" in data_from_kp_splited
				data_from_kp = data_from_kp_
				break
			end
		end
		if data_from_kp==""
			printstyled("Error: data_from_kp not found\n"; color = :red)
			return
		else
			data_from_kp_dict[year_path] = data_from_kp
			print("Found \"")
			printstyled(data_from_kp; color = :yellow)
			println("\".")
		end
	end

	# Locate all images
	print("\tRecursively searching images...")
	images_paths = Array{Any}(undef, 6) 
	Threads.@threads for i = 1:6
	# for i = 1:6
		year_path = year_dir[i]
		curr_path = joinpath(year_path, data_from_kp_dict[year_path])
		images_paths[i] = rec_search(joinpath(BAC_dir, curr_path), "image")
	end
	images = []
	# Append after multithreading to prevent data races
	for i = 1:6
		append!(images, images_paths[i])
	end
	
	image_ct = size(images)[1]
	println("Done. Found $image_ct images.")
	return images, image_ct
end

# ╔═╡ 941de9bb-2790-4a9e-a534-3c83083de75a
"""
	This function deals with the file system and locates all the images.
"""
function Locate_labels()
	println("\nLocating labels...")

	# Locate year dirs
	roi_manual_dict = Dict()
	year_dir = ["Patientstudy_BAC_2013a", "Patientstudy_BAC_2013b", "Patientstudy_BAC_2014", "Patientstudy_BAC_2015a", "Patientstudy_BAC_2015b", "Patientstudy_BAC_2016"]
	for year_path in year_dir
		isdir(joinpath(BAC_dir, year_path)) || (printstyled("\tError: Failed to locate the folder: $year_path.\n"; color = :red);return)
		print("\tFound \"")
		printstyled(year_path; color = :yellow)
		print("\"; ")
		roi_manual = ""
		for roi_manual_ in readdir(joinpath(BAC_dir, year_path))
			roi_manual_splited = split(lowercase(roi_manual_), [' ', '_'])
			if "roi" in roi_manual_splited && "manual" in roi_manual_splited
				roi_manual = roi_manual_
				break
			end
		end
		if roi_manual==""
			printstyled("Error: ROI_manual not found\n"; color = :red)
			return
		else
			roi_manual_dict[year_path] = roi_manual
			print("Found \"")
			printstyled(roi_manual; color = :yellow)
			println("\".")
		end
	end

	# Locate all images
	print("\tRecursively searching labels...")
	labels_paths = Array{Any}(undef, 6) 
	Threads.@threads for i = 1:6
		year_path = year_dir[i]
		curr_path = joinpath(year_path, roi_manual_dict[year_path])
		labels_paths[i] = rec_search(joinpath(BAC_dir, curr_path), "label")
	end
	labels = []
	# Append after multithreading to prevent data races
	for i = 1:6
		append!(labels, labels_paths[i])
	end
	
	label_ct = size(labels)[1]
	println("Done. Found $label_ct labels.")
	return labels, label_ct
end

# ╔═╡ dfe74c22-8f72-40bb-9047-fdd7d868b067
"""
	This function deals with the file system and locates all images within train and valid dataset and copy them to `collected_dataset_for_ML`.
"""
function locate_and_copy_images!(valid_set, train_set, images_paths, image_ct; overwrite = false)
	println("Matching images...")
	V_P_dict = Dict("L CC" => 1, "L MLO" => 2, "R CC" => 3, "R MLO" => 4)
	validation_size = size(valid_set)[1]
	training_size = size(train_set)[1]
	
	# Valid set - create folders if needed
	valid_image_dir = joinpath(valid_data_dir, "image")
	if isdir(valid_image_dir) && overwrite
		rm(valid_image_dir, recursive=true)
	end
	isdir(valid_image_dir) || mkdir(valid_image_dir)
	for idx_set = 1 : validation_size
		dir = joinpath(valid_image_dir, valid_set[idx_set][4])
		isdir(dir) || mkdir(dir)
	end
	
	
	# Train set - create folders if needed
	train_image_dir = joinpath(train_data_dir, "image")
	if isdir(train_image_dir) && overwrite
		rm(train_image_dir, recursive=true)
	end
	isdir(train_image_dir) || mkdir(train_image_dir)
	for idx_set = 1 : training_size
		dir = joinpath(train_image_dir, train_set[idx_set][4])
		isdir(dir) || mkdir(dir)
	end

	# Start matching
	for idx_images = 1:image_ct
	# for idx_images = 1:100
		curr_path = images_paths[idx_images]

		# Read
		dcm_data = dcm_parse(images_paths[idx_images])
		LvsR = ""
		try
			LvsR = uppercase(dcm_data[(0x0020,0x0062)])
		catch e
			println("----------------------")
			println(e)
			println(curr_path)
			println("----------------------")
		end
		V_P = ""
		try
			V_P = uppercase(dcm_data[(0x0018,0x5101)])
		catch e
			println("----------------------")
			println(e)
			println(curr_path)
			println("----------------------")
		end
		curr_key = LvsR*" "*V_P
		
		# Check V_P
		curr_key in keys(V_P_dict) || continue
		
		SID = dcm_data[(0x0010,0x0020)]
		UID = dcm_data[(0x0008,0x0018)]
		curr_acq_time = dcm_data[(0x0008,0x0032)]

		# Valid image
		Threads.@threads for idx_set = 1 : validation_size
			SID_target = valid_set[idx_set][4]
			# Check SID
			SID == SID_target || continue
			# # Copy image to `collected_dataset_for_ML` folder
			new_path = joinpath(valid_image_dir, SID, UID*".dcm")

			#if existed already
			if isfile(new_path)
				prev_acq_time = dcm_parse(new_path)[(0x0008,0x0032)]
				if curr_acq_time > prev_acq_time
					rm(new_path)
					cp(curr_path, new_path)
					valid_set[idx_set][3][V_P_dict[curr_key]] = UID
				end
			else
				cp(curr_path, new_path)
				valid_set[idx_set][3][V_P_dict[curr_key]] = UID
				valid_set[idx_set][1][V_P_dict[curr_key]] = new_path
			end
			break
		end

		# Train image
		Threads.@threads for idx_set = 1 : training_size
			SID_target = train_set[idx_set][4]
			# Check SID
			SID == SID_target || continue
			# Copy image to `collected_dataset_for_ML` folder
			new_path = joinpath(train_image_dir, SID, UID*".dcm")

			#if existed already
			if isfile(new_path)
				prev_acq_time = dcm_parse(new_path)[(0x0008,0x0032)]
				if curr_acq_time > prev_acq_time
					rm(new_path)
					cp(curr_path, new_path)
					train_set[idx_set][3][V_P_dict[curr_key]] = UID
				elseif curr_acq_time == prev_acq_time
					train_set[idx_set][3][V_P_dict[curr_key]] = UID
					train_set[idx_set][1][V_P_dict[curr_key]] = new_path
				end
			else
				cp(curr_path, new_path)
				train_set[idx_set][3][V_P_dict[curr_key]] = UID
				train_set[idx_set][1][V_P_dict[curr_key]] = new_path
			end
		end
	end
end

# ╔═╡ f1712674-d34f-43a4-9a67-1b08b26ea3fa
"""
	This function deals with the file system and locates all labels within train and valid dataset and copy them to `collected_dataset_for_ML`.
"""
function locate_and_copy_labels!(valid_set, train_set, labels_paths, label_ct; overwrite = false)
	println("Matching labels...")
	V_P_dict = Dict("L CC" => 1, "L MLO" => 2, "R CC" => 3, "R MLO" => 4)
	validation_size = size(valid_set)[1]
	training_size = size(train_set)[1]

	# dict
	data_from_kp_dict = Dict(
	"Patientstudy_BAC_2013a" => "data_from_KP",
	"Patientstudy_BAC_2013b" => "data_from_KP",
	"Patientstudy_BAC_2014" => "data_from_KP",
	"Patientstudy_BAC_2015a" => "data_from_KP",
	"Patientstudy_BAC_2015b" => "data_from_KP",
	"Patientstudy_BAC_2016" => "Data from KP")
	
	# Valid set - create folders if needed
	valid_label_dir = joinpath(valid_data_dir, "label")
	if isdir(valid_label_dir) && overwrite
		rm(valid_label_dir, recursive=true)
	end
	isdir(valid_label_dir) || mkdir(valid_label_dir)
	for idx_set = 1 : validation_size
		dir = joinpath(valid_label_dir, valid_set[idx_set][4])
		isdir(dir) || mkdir(dir)
	end
	
	# Train set - create folders if needed
	train_label_dir = joinpath(train_data_dir, "label")
	if isdir(train_label_dir) && overwrite
		rm(train_label_dir, recursive=true)
	end
	isdir(train_label_dir) || mkdir(train_label_dir)
	for idx_set = 1 : training_size
		dir = joinpath(train_label_dir, train_set[idx_set][4])
		isdir(dir) || mkdir(dir)
	end
	
	# Start matching
	for idx_labels = 1:label_ct
		curr_path = labels_paths[idx_labels]
		curr_UID = split(rsplit(curr_path, "\\"; limit = 2)[2], ['-','_'])[end][1:end-4]
		if length(curr_UID)<10
			# special case 1
			year, rest = split(curr_path, "\\"; limit = 4)[3:4]
			rest = split(rest, "\\")
			pos = 0
			for (i, d) in enumerate(rest[2:end])
				rest_ = split(d, "_")
				if "patientdata" in rest_
					pos = i+1
					break
				end
			end
			if pos==0 && rest[end] == "LMLO_1.2.840.113681.2230565808.960.3551698729.29-1.roi"
				# exclude this one
				continue
			end
			date = rest[pos]
			correct_dir = joinpath(BAC_dir, year, data_from_kp_dict[year], date)
			for p in rest[pos+1 : end-1]
				correct_dir = joinpath(correct_dir, p)
			end
			if isdir(correct_dir) && size(readdir(correct_dir))[1] == 1
				# End case 1
				curr_UID = readdir(correct_dir)[1][1:end-4]
				# println("Special case 1:\n\tUID = $curr_UID\nCorrect_dir = $correct_dir\n--------------\n")
			else
				# special case 2
				if rest[pos+1] == "all" && lowercase(rest[pos+2][1:3]) == "sid"
					vp = rest[end][1:end-4]
					correct_dir = joinpath(BAC_dir, year, data_from_kp_dict[year], date, split(rest[pos+2], ['_', '-'])[end], vp)
				else
					# ignore "sep"
					continue
				end
				if isdir(correct_dir) && size(readdir(correct_dir))[1] == 1
					# End case 2
					curr_UID = readdir(correct_dir)[1][1:end-4]
					# println("Special case 2:\n\tUID = $curr_UID\nCorrect_dir = $correct_dir\n--------------\n")
				else
					# special case 3
					correct_dir = joinpath(BAC_dir, year, data_from_kp_dict[year], date, split(rest[pos+2], ['_', '-'])[end])
					for d in readdir(correct_dir)
						curr_UID_name = readdir(joinpath(correct_dir, d))[1][1:end-4]
						dcm_data = dcm_parse(joinpath(correct_dir, d, curr_UID_name*".dcm"))
						LvsR = uppercase(dcm_data[(0x0020,0x0062)])
						V_P = uppercase(dcm_data[(0x0018,0x5101)])
						curr_key = LvsR*V_P
						if curr_key == uppercase(vp)
							correct_dir = joinpath(correct_dir, d)
							curr_UID = curr_UID_name
							break
						end
					end
					if isdir(correct_dir) && size(readdir(correct_dir))[1] == 1
						# End case 3
						# println("Special case 3:\n\tUID = $curr_UID\nCorrect_dir = $correct_dir\n--------------\n")
					else
						# Others
						println("Unknown edge case:")
						println(curr_path)
						println("\n--------------\n")
					end
				end
					
			end
		end
		# Now we got the correct UID, matching valid set
		Threads.@threads for idx_set = 1 : validation_size
			target_UIDs = valid_set[idx_set][3]
			SID = valid_set[idx_set][4]
			for i = 1:4
				if curr_UID == target_UIDs[i]
					# found match
					new_path = joinpath(valid_label_dir, SID, curr_UID*".roi")
					cp(curr_path, new_path; force=true)
					valid_set[idx_set][2][i] = new_path
				end
			end
		end

		# Train set
		for idx_set = 1 : training_size
			target_UIDs = train_set[idx_set][3]
			SID = train_set[idx_set][4]
			for i = 1:4
				if curr_UID == target_UIDs[i]
					# found match
					new_path = joinpath(train_label_dir, SID, curr_UID*".roi")
					isfile(new_path) || (cp(curr_path, new_path); 
						train_set[idx_set][2][i] = new_path)
				end
			end
		end
	end
end

# ╔═╡ 4849ac0d-c1ba-42cd-acfe-995362fc9cd7
"""
	This function checks if all images, lables are found.
"""
function check_dataset(valid_set, train_set)
	println("\nChecking dataset...")
	# valid_set
	println("\tChecking validation set...")
	for (idx, (images, labels, UIDs, SID, L_mass, R_mass, sum_mass, masses)) in enumerate(valid_set)
		error_msg = "\t\tError: valid_set[$idx]"
		error = false
		
		# Check images' paths
		for i = 1:4
			isassigned(images, i) || (error = true; printstyled("$error_msg -> image #$i not found!\n"; color = :yellow); break)
		end
		error && continue

		# Check UIDs
		for i = 1:4
			isassigned(UIDs, i) || (error = true; printstyled("$error_msg -> UID #$i not found!\n"; color = :yellow); break)
		end
		error && continue

		# Check if labels match mass
		# Left 1
		if masses[1] != 0 && !isassigned(labels, 1)
			error = true
			printstyled("$error_msg -> L1 label not found!\n"; color = :yellow)
		end
		error && continue
		
		# Left 2
		if masses[2] != 0 && !isassigned(labels, 2)
			error = true
			printstyled("$error_msg -> L2 label not found!\n"; color = :yellow)
		end
		error && continue
		
		# Right 1
		if masses[3] != 0 && !isassigned(labels, 3)
			error = true
			printstyled("$error_msg -> R1 label not found!\n"; color = :yellow)
		end
		error && continue
		
		# Right 2
		if masses[4] != 0 && !isassigned(labels, 4)
			error = true
			printstyled("$error_msg -> R2 label not found!\n"; color = :yellow)
		end
		error && continue
	end
	
	# train_set
	println("\tChecking training set...")
	for (idx, (images, labels, UIDs, SID, L_mass, R_mass, sum_mass, masses)) in enumerate(train_set)
		error_msg = "\t\tError: train_set[$idx]"
		error = false
		
		# Check images' paths
		for i = 1:4
			isassigned(images, i) || (error = true; printstyled("$error_msg -> image #$i not found!\n"; color = :yellow); break)
		end
		error && continue

		# Check UIDs
		for i = 1:4
			isassigned(UIDs, i) || (error = true; printstyled("$error_msg -> UID #$i not found!\n"; color = :yellow); break)
		end
		error && continue

		# Check if labels match mass
		# Left 1
		if masses[1] != 0 && !isassigned(labels, 1)
			error = true
			printstyled("$error_msg -> L1 label not found!\n"; color = :yellow)
		end
		error && continue
		
		# Left 2
		if masses[2] != 0 && !isassigned(labels, 2)
			error = true
			printstyled("$error_msg -> L2 label not found!\n"; color = :yellow)
		end
		error && continue
		
		# Right 1
		if masses[3] != 0 && !isassigned(labels, 3)
			error = true
			printstyled("$error_msg -> R1 label not found!\n"; color = :yellow)
		end
		error && continue
		
		# Right 2
		if masses[4] != 0 && !isassigned(labels, 4)
			error = true
			printstyled("$error_msg -> R2 label not found!\n"; color = :yellow)
		end
		error && continue
	end
	println("Done!")
end

# ╔═╡ 0d5ef811-ed51-4955-a57c-90b0d1780362
"""
	This function builds the dataset. One should run this function ONLY ONCE, as running again will duplicate the copying process. If one needs to remove a current dataset and build a new data. Run with `overwrite = true`.

	Dataset will be stored to "C:/BAC_dataset/collected_dataset_for_ML/"
"""
function build_dataset(;overwrite = false)
	valid_set, train_set = read_excel_and_build_data()
	images_paths, image_ct = Locate_shared_drive_files()
	locate_and_copy!(valid_set, train_set, images_paths, image_ct; overwrite = overwrite)
end

# ╔═╡ c1582911-15c5-48e3-ad16-42d868a218ba
begin
	# valid_set, train_set = read_excel_and_build_data()
	# images_paths, image_ct = Locate_images()
	# labels_paths, label_ct = Locate_labels()
end

# ╔═╡ 5a771c7f-3baa-4b97-a105-e02b5c4aa405
# locate_and_copy_images!(valid_set, train_set, images_paths, image_ct; overwrite = true)

# ╔═╡ c530472c-3357-4a88-9bd0-0efcc6216506
# locate_and_copy_labels!(valid_set, train_set, labels_paths, label_ct; overwrite = true)

# ╔═╡ 8b079e28-26f7-426f-ab6f-3632df2f2d2d
# check_dataset(valid_set, train_set)

# ╔═╡ 696d3cc3-547e-4313-b76b-d2599e9a0418
begin
	idx_inspect = 8
	set_inspect = valid_set
end;

# ╔═╡ cb69de73-5826-4cb6-b179-3c0bbc58f0c6
set_inspect[idx_inspect]

# ╔═╡ b23ba6f3-fe17-4a05-9413-4280fd7ffbce
 search_for_specific_file(rsplit(set_inspect[idx_inspect][1][3], "\\"; limit = 2)[2])

# ╔═╡ 6688a648-165f-4c5e-bff8-8fb26c6a6637
#=

	"L CC" => 1
	"L MLO" => 2
	"R CC" => 3
	"R MLO" => 4

=#

# ╔═╡ 0f285f76-4654-41fd-9fa1-663bbb9dc4bd
#=

	Note:
		1. Missing L MLO roi: SID-120759; action: deleted from excel
			C:\BAC_dataset\Patientstudy_BAC_2014\data_from_KP\patientdata_12232014\SID120759_9999.252003244562861319872442637369620381507

		2. Missing all 4 rois: SID-128938; action: deleted from excel
			C:\BAC_dataset\Patientstudy_BAC_2015a\data_from_KP\patientdata_01122015_ver1\9999.130605962513579093035510871191652416231\9999.218619105550672546193892237031945440228\1.2.840.113681.2230565208.1097.3583754939.216.dcm

		3. Missing all 4 rois: SID-128938; action: deleted from excel
			C:\BAC_dataset\Patientstudy_BAC_2015a\data_from_KP\patientdata_01122015_ver1\9999.53463995667350864452014324255175774271\9999.233785338442764634253758139824078959002\1.2.840.113681.2229466748.940.3586096671.24.dcm

		4. Missing all 4 rois: SID-127571; action: deleted from excel
			C:\BAC_dataset\Patientstudy_BAC_2015a\data_from_KP\patientdata_01122015_ver1\9999.265068198994854859029991995932891730345\9999.84618478665538419285284336328047798347\1.2.840.113681.2230562826.969.3582806501.39.dcm

		5. Missing MLO scan: SID-113972; deleted from excel
			C:\BAC_dataset\Patientstudy_BAC_2014\data_from_KP\patientdata_03272014\9999.14612740449131251210073765295840472676\9999.211186508647625948716036028091784929679

		6. Missing LCC and LMO rois: SID-102396; deleted from excel
			C:\BAC_dataset\Patientstudy_BAC_2014\data_from_KP\patientdata_03072014\9999.175540248468875467872357098993829966005\9999.335085199231000338936299033793071085496\1.2.840.113681.2230563618.926.3569601867.188.dcm

		7. Missing all 4 rois: SID-102110; action: deleted from excel
			C:\BAC_dataset\Patientstudy_BAC_2015a\data_from_KP\patientdata_01122015_ver2\9999.50744375181391260492816331922453396346\9999.291942512761586632929450535718253070565\1.2.840.113681.2230563618.924.3595507856.377.dcm

		8. Missing all 2 rois: SID-118436; action: deleted from excel
			C:\BAC_dataset\Patientstudy_BAC_2014\data_from_KP\patientdata_06252014\9999.34907754794302922421578725131621415740\9999.46151628837242401531189400843473630833\1.2.840.113619.2.255.273315310659.1976140130132650.81.dcm

		9. Missing L CC image: SID-113038; action: deleted from excel
			C:\BAC_dataset\Patientstudy_BAC_2014\data_from_KP\patientdata_02102014\9999.133539637703847430287877112789083304645\9999.142914532170936754689712010215492030228\1.2.840.113619.2.255.273316596199.2734130830095237.527.dcm

=#

# ╔═╡ 2553cbc3-736b-4800-b91d-7627993814ac
md"""
# Save clean dataset
"""

# ╔═╡ 5b127c56-9302-4633-8be3-00a05c74f24b
@save "clean_set.jld2" train_set valid_set

# ╔═╡ Cell order:
# ╠═e6d23e80-0bd2-11ee-1785-73e34754591b
# ╠═7975a099-49d4-496f-bbd9-ae6c9c90e507
# ╠═268ceef5-2c2d-475b-b18d-246da9f2c9c0
# ╠═80335714-4f14-4e4b-b944-23a3041272e7
# ╠═f11b0ed3-8ad8-4677-b663-972849f9387a
# ╠═ff43e82f-61bc-4082-b14d-8f3162435349
# ╟─849b3530-f419-4e22-a0d8-7a622f04f6b8
# ╟─1adb1c22-6e59-47bd-b38e-dcd3e5aecde8
# ╟─76c957f4-cb09-4fa8-bdd0-f1835319ccfc
# ╟─70bd3f84-97f4-4618-9893-aa7e55481ea8
# ╟─941de9bb-2790-4a9e-a534-3c83083de75a
# ╟─dfe74c22-8f72-40bb-9047-fdd7d868b067
# ╟─f1712674-d34f-43a4-9a67-1b08b26ea3fa
# ╠═4849ac0d-c1ba-42cd-acfe-995362fc9cd7
# ╟─0d5ef811-ed51-4955-a57c-90b0d1780362
# ╠═c1582911-15c5-48e3-ad16-42d868a218ba
# ╠═5a771c7f-3baa-4b97-a105-e02b5c4aa405
# ╠═c530472c-3357-4a88-9bd0-0efcc6216506
# ╠═8b079e28-26f7-426f-ab6f-3632df2f2d2d
# ╠═696d3cc3-547e-4313-b76b-d2599e9a0418
# ╠═cb69de73-5826-4cb6-b179-3c0bbc58f0c6
# ╠═b23ba6f3-fe17-4a05-9413-4280fd7ffbce
# ╠═6688a648-165f-4c5e-bff8-8fb26c6a6637
# ╠═0f285f76-4654-41fd-9fa1-663bbb9dc4bd
# ╟─2553cbc3-736b-4800-b91d-7627993814ac
# ╠═b98df1ab-ca99-4802-acc2-02fffe8d4284
# ╠═5b127c56-9302-4633-8be3-00a05c74f24b
