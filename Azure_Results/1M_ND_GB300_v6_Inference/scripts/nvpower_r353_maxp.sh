#!/bin/bash

#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


function set_socfamily()
{
	if [ -e "/proc/device-tree/compatible" ]; then
		machine="$(tr -d '\0' < /proc/device-tree/compatible)"
		if [[ "${machine}" =~ "e3900" ]]; then
			machine="e3900"
		elif [[ "${machine}" =~ "jetson-xavier-industrial" ]]; then
			machine="jetson-xavier-industrial"
		elif [[ "${machine}" =~ "jetson-xavier" ]]; then
			machine="jetson-xavier"
		elif [[ "${machine}" =~ "p2972-0006" ]]; then
			machine="p2972-0006"
		elif [[ "${machine}" =~ "p3668" ]]; then
			if [[ "${machine}" =~ "nvidia,p3668-emul" ]]; then
				machine="p3668-emul"
			else
				machine="p3668"
			fi
		elif [[ "${machine}" =~ "p3701" ]]; then
			if [[ "${machine}" =~ "p3701-0000-as-p3767-0000" ]]; then
				machine="p3701-0000-as-p3767-0000"
			elif [[ "${machine}" =~ "p3701-0000-as-p3767-0001" ]]; then
				machine="p3701-0000-as-p3767-0001"
			elif [[ "${machine}" =~ "p3701-0000-as-p3767-0003" ]]; then
				machine="p3701-0000-as-p3767-0003"
			elif [[ "${machine}" =~ "p3701-0000-as-p3767-0004" ]]; then
				machine="p3701-0000-as-p3767-0004"
			elif [[ "${machine}" =~ "p3701-0000-as-pxxxx" ]]; then
				machine="p3701-0000-as-pxxxx"
			elif [[ "${machine}" =~ "p3701-0002" ]]; then
				machine="p3701-0002"
			elif [[ "${machine}" =~ "p3701-0000-as-p3701-0004" ]]; then
				machine="p3701-0000-as-p3701-0004"
			elif [[ "${machine}" =~ "p3701-0004" ]]; then
				machine="p3701-0004"
			else
				machine="p3701-0000"
			fi
		elif [[ "${machine}" =~ "p3767" ]]; then
			if [[ "${machine}" =~ "p3767-0000-as-p3767-0001" ]]; then
				machine="p3767-0000-as-p3767-0001"
			elif [[ "${machine}" =~ "p3767-0000-as-p3767-0003" ]]; then
				machine="p3767-0000-as-p3767-0003"
			elif [[ "${machine}" =~ "p3767-0000-as-p3767-0004" ]]; then
				machine="p3767-0000-as-p3767-0004"
			elif [[ "${machine}" =~ "p3767-0001" ]]; then
				machine="p3767-0001"
			elif [[ "${machine}" =~ "p3767-0002" ]]; then
				machine="p3767-0002"
			elif [[ "${machine}" =~ "p3767-0003" ]]; then
				machine="p3767-0003"
			elif [[ "${machine}" =~ "p3767-0004" ]]; then
				machine="p3767-0004"
			elif [[ "${machine}" =~ "p3767-0005" ]]; then
				machine="p3767-0005"
			else
				machine="p3767-0000"
			fi
		elif [[ "${machine}" =~ "e2421-1099-as-pxxxx" ]]; then
				machine="e2421-1099-as-pxxxx"
		else
			machine="$(cat /proc/device-tree/model)"
		fi

		CHIP="$(tr -d '\0' < /proc/device-tree/compatible)"
		if [[ "${CHIP}" =~ "tegra194" ]]; then
			SOCFAMILY="tegra194"
		elif [[ "${CHIP}" =~ "tegra234" ]]; then
			SOCFAMILY="tegra234"
		fi
	fi
}

function set_power_state_perm()
{
	# set power state permission
	if [ -e "/sys/power/state" ]; then
		chmod 0666 "/sys/power/state"
	fi
}

function create_nvpmodel_symlink()
{
	conf_file=""
	# create /etc/nvpmodel.conf symlink
	if [ ! -e "/etc/nvpmodel.conf" ]; then
		if [ "${SOCFAMILY}" = "tegra194" ]; then
			if [ "${machine}" = "e3900" ]; then
				if [ -d "/sys/devices/gpu.0" ] &&
					[ -d "/sys/devices/17000000.gv11b" ]; then
					conf_file="/etc/nvpmodel/nvpmodel_t194_e3900_iGPU.conf"
				else
					conf_file="/etc/nvpmodel/nvpmodel_t194_e3900_dGPU.conf"
				fi
			elif [ "${machine}" = "p2972-0006" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_t194_8gb.conf"
			elif [ "${machine}" = "p3668" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_t194_p3668.conf"
			elif [ "${machine}" = "p3668-emul" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_t194_p3668_emul.conf"
			elif [ "${machine}" = "jetson-xavier-industrial" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_t194_agxi.conf"
			else
				conf_file="/etc/nvpmodel/nvpmodel_t194.conf"
			fi
		elif [ "${SOCFAMILY}" = "tegra234" ]; then
			if [ "${machine}" = "p3701-0000-as-p3767-0000" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_p3767_0000.conf"
			elif [ "${machine}" = "p3701-0000-as-p3767-0001" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_p3767_0001.conf"
			elif [ "${machine}" = "p3701-0000-as-p3767-0003" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_p3767_0003.conf"
			elif [ "${machine}" = "p3701-0000-as-p3767-0004" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_p3767_0004.conf"
			elif [ "${machine}" = "p3701-0000-as-pxxxx" ] || \
				[ "${machine}" = "e2421-1099-as-pxxxx" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_pxxxx.conf"
			elif [ "${machine}" = "p3701-0002" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_p3701_0002.conf"
			elif [ "${machine}" = "p3701-0000-as-p3701-0004" ] || \
				[ "${machine}" = "p3701-0004" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_p3701_0004.conf"
			elif [ "${machine}" = "p3767-0000" ] || \
				[ "${machine}" = "p3767-0002" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_p3767_0000.conf"
			elif [ "${machine}" = "p3767-0000-as-p3767-0001" ] || \
				[ "${machine}" = "p3767-0001" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_p3767_0001.conf"
			elif [ "${machine}" = "p3767-0003" ] || \
				[ "${machine}" = "p3767-0005" ] || \
				[ "${machine}" = "p3767-0000-as-p3767-0003" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_p3767_0003.conf"
			elif [ "${machine}" = "p3767-0004" ] || \
				[ "${machine}" = "p3767-0000-as-p3767-0004" ]; then
				conf_file="/etc/nvpmodel/nvpmodel_p3767_0004.conf"
			else
				conf_file="/etc/nvpmodel/nvpmodel_p3701_0000.conf"
			fi
		fi

		if [ "${conf_file}" != "" ]; then
			if [ -e "${conf_file}" ]; then
				ln -sf "${conf_file}" /etc/nvpmodel.conf
			else
				echo "${SCRIPT_NAME} - WARNING: file ${conf_file} not found!"
			fi
		fi
	fi
}

function create_nvfancontrol_symlink()
{
	conf_file=""
	if [ ! -e "/etc/nvfancontrol.conf" ]; then
		if [ "${SOCFAMILY}" = "tegra194" ]; then
			if [ "${machine}" = "e3900" ]; then
				conf_file="/etc/nvpower/nvfancontrol/nvfancontrol_e3900.conf"
			elif [ "${machine}" = "p3668" ]; then
				conf_file="/etc/nvpower/nvfancontrol/nvfancontrol_p3668.conf"
			else
				conf_file="/etc/nvpower/nvfancontrol/nvfancontrol_p2888.conf"
			fi
		fi

		if [ "${SOCFAMILY}" = "tegra234" ]; then
			if [ "${machine}" = "p3701-0002" ]; then
				conf_file="/etc/nvpower/nvfancontrol/nvfancontrol_p3701_0002.conf"
			elif [[ "${machine}" =~ "p3767" ]]; then
				conf_file="/etc/nvpower/nvfancontrol/nvfancontrol_p3767_0000.conf"
			else
				conf_file="/etc/nvpower/nvfancontrol/nvfancontrol_p3701_0000.conf"
			fi
		fi

		if [ "${conf_file}" != "" ]; then
			if [ -e "${conf_file}" ]; then
				ln -sf "${conf_file}" /etc/nvfancontrol.conf
			else
				echo "${SCRIPT_NAME} - WARNING: file ${conf_file} not found!"
			fi
		fi
	fi
}

function cpu_hotplug()
{
	# CPU hotplug
	if [ "$SOCFAMILY" = "tegra194" ]; then
		if [ "$machine" = "p2972-0006" -o "$machine" = "p3668" ]; then
			if [ -e /sys/devices/system/cpu/cpu6/online ]; then
				echo 0 > /sys/devices/system/cpu/cpu6/online
			fi
			if [ -e /sys/devices/system/cpu/cpu7/online ]; then
				echo 0 > /sys/devices/system/cpu/cpu7/online
			fi
		fi
	fi
}

function set_cpu_governor()
{
	# CPU governor setting
	CPU_INTERACTIVE_GOV=0
	CPU_SCHEDUTIL_GOV=0

	KERNEL_VERSION=$(uname -r | cut -d '.' -f-2)

	if [ -e /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors ]; \
		then
		read governors < \
			/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors

		case $governors in
			*interactive*)
				CPU_INTERACTIVE_GOV=1
			;;
		esac

		case $governors in
			*schedutil*)
				# latest kernel is using the upstream driver and need to avoid setting
				# schedutil governor on kstable.
				if [ "$(echo "${KERNEL_VERSION} < 5.11" | bc)" -eq 1 ]; then
					CPU_SCHEDUTIL_GOV=1
				fi
	        ;;
	    esac
	fi

	SCHEDUTIL="/sys/devices/system/cpu/cpufreq/schedutil"
	RATE_LIMIT_US="${SCHEDUTIL}/rate_limit_us"
	UP_LIMIT_US="${SCHEDUTIL}/up_rate_limit_us"
	DOWN_LIMIT_US="${SCHEDUTIL}/down_rate_limit_us"
	CAPACITY_MARGIN="${SCHEDUTIL}/capacity_margin"

	case "${SOCFAMILY}" in
		tegra210 | tegra186 | tegra194 | tegra234)
			if [ "${CPU_SCHEDUTIL_GOV}" = "1" ]; then
				for scaling_governor in \
					/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
					echo schedutil > "${scaling_governor}"
				done
				if [ -e "${RATE_LIMIT_US}" ]; then
					echo 2000 > "${RATE_LIMIT_US}"
				fi
				if [ -e "${UP_LIMIT_US}" ]; then
					echo 0 > "${UP_LIMIT_US}"
				fi
				if [ -e "${DOWN_LIMIT_US}" ]; then
					echo 500 > "${DOWN_LIMIT_US}"
				fi
				if [ -e "${CAPACITY_MARGIN}" ]; then
					echo 1024 > "${CAPACITY_MARGIN}"
				fi
			elif [ "${CPU_INTERACTIVE_GOV}" = "1" ]; then
				for scaling_governor in \
					/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
					echo interactive > "${scaling_governor}"
				done
			fi
			;;
		*)
			;;
	esac
}

function lock_se_frequency()
{
	# Lock SE clock at MinFreq to reduce vdd_soc power
	if [ "${SOCFAMILY}" = "tegra194" ]; then
		if [ -d "/sys/kernel/debug/bpmp/debug/clk/nafll_se" ]; then
			echo 1 > /sys/kernel/debug/bpmp/debug/clk/nafll_se/mrq_rate_locked
			cat /sys/kernel/debug/bpmp/debug/clk/nafll_se/min_rate > \
				/sys/kernel/debug/bpmp/debug/clk/nafll_se/rate
		fi
	fi
}

function enable_vic_actmon()
{
	# Enable VIC actmon by setting wmark_active devfreq governor
	# Configure wmark_active parameters.
	DEVFREQ_WMARK_ACTIVE_GOV=0

	case "${SOCFAMILY}" in
		tegra186 | tegra194 | tegra234)
			VIC_GOV_PARAM_PATH="/sys/devices/platform/host1x/15340000.vic"
			VIC_ACTMON_PATH="/sys/kernel/debug/vic"
			VIC_DEV_PATH="${VIC_GOV_PARAM_PATH}/devfreq_dev"

			if [ -e "${VIC_DEV_PATH}/available_governors" ]; then
				read governors < "${VIC_DEV_PATH}/available_governors"

				case "${governors}" in
					*wmark_active*)
						DEVFREQ_WMARK_ACTIVE_GOV=1
					;;
				esac
			fi

			if [ "${DEVFREQ_WMARK_ACTIVE_GOV}" -eq 1 ]; then
				if [ -e "${VIC_DEV_PATH}/governor" ]; then
					echo wmark_active > "${VIC_DEV_PATH}/governor"
				fi
				if [ -e "${VIC_GOV_PARAM_PATH}/load_target" ]; then
					echo 700 > "${VIC_GOV_PARAM_PATH}/load_target"
				fi
				if [ -e "${VIC_GOV_PARAM_PATH}/load_max" ]; then
					echo 900 > "${VIC_GOV_PARAM_PATH}/load_max"
				fi
				if [ -e "${VIC_GOV_PARAM_PATH}/block_window" ]; then
					echo 0 > "${VIC_GOV_PARAM_PATH}/block_window"
				fi
				if [ -e "${VIC_GOV_PARAM_PATH}/smooth" ]; then
					echo 0 > "${VIC_GOV_PARAM_PATH}/smooth"
				fi
				if [ -e "${VIC_GOV_PARAM_PATH}/freq_boost_en" ]; then
					echo 0 > "${VIC_GOV_PARAM_PATH}/freq_boost_en"
				fi
			fi

			if [ -e "${VIC_ACTMON_PATH}/actmon_k" ]; then
				echo 4 > "${VIC_ACTMON_PATH}/actmon_k"
			fi
			if [ -e "${VIC_ACTMON_PATH}/actmon_sample_period_norm" ]; then
				echo 1500 > "${VIC_ACTMON_PATH}/actmon_sample_period_norm"
			fi
			;;
		*)
			;;
	esac
}

# validate the correct nvpmodel.conf file for the xavier-nx platform
function check_nvpmodel_param()
{
	if ! grep -q "< PARAM TYPE=${1} NAME=${2} >" "/etc/nvpmodel.conf"; then
		echo "nvpmodel: ${2} cap is not set" > /dev/kmsg
		return 1
	fi

	return 0
}

function validate_nvpmodel_conf()
{
	if [ "${SOCFAMILY}" == "tegra194" ]; then
		if [ "${machine}" == "p3668" ]; then
			state=0
			SOC_CLOCKS=("NVENC" "NVENC1" "NVDEC" "NVDEC1" "NVJPG" "SE1" "SE2" "SE3" "SE4")

			for clk in "${SOC_CLOCKS[@]}"; do
				check_nvpmodel_param "CLOCK" "${clk}"
				state=$((state | $?))
			done

			check_nvpmodel_param "HWMON" "VDDIN_OC_LIMIT"
			state=$((state | $?))
			if [ "${state}" == 1 ]; then
				echo "nvpmodel: incompatible nvpmodel.conf file. Regenerate latest conf file \
using PowerEstimator web tool - https://jetson-tools.nvidia.com/powerestimator" > /dev/kmsg
			fi
		fi
	fi
}

function configure_tmp451_sensor()
{
	i2c_addr="$1"
	temp1_min="$2"
	temp1_max="$3"
	temp1_crit="$4"
	temp2_min="$5"
	temp2_max="$6"
	temp2_crit="$7"
	temp2_offset="$8"
	update_interval="$9"

	hwmon_path=$(find "/sys/bus/i2c/devices/${i2c_addr}/hwmon/" -name 'hwmon[0-9]*' 2>/dev/null)

	if [ -n "$hwmon_path" ]; then
		echo "$temp1_min" > "${hwmon_path}/temp1_min"
		echo "$temp1_max" > "${hwmon_path}/temp1_max"
		echo "$temp1_crit" > "${hwmon_path}/temp1_crit"
		echo "$temp2_min" > "${hwmon_path}/temp2_min"
		echo "$temp2_max" > "${hwmon_path}/temp2_max"
		echo "$temp2_crit" > "${hwmon_path}/temp2_crit"
		echo "$temp2_offset" > "${hwmon_path}/temp2_offset"
		echo "$update_interval" > "${hwmon_path}/update_interval"
	else
		echo "tmp451 thermal sensor not found!"
	fi
}

# HWMON device configurations helper function
# * Parameters:
# * ${1}: device node name
# * ${2}: setup node name
# * ${3}: setup value
# * ${4}: condition (optional)
# * ${5}: power limit in mW (optional)
# * ${6}: bus voltage channel to calculate critical current (optional)
# case ina3221: use first channel label name as the condition
#               since there could be multiple ina3221 nodes.
function config_hwmon()
{
	for dir in /sys/class/hwmon/*
	do
		# Match device node name
		if [ "$(< ${dir}/name)" != "${1}" ]; then continue; fi
		# Check the optional condition
		if [[ ! -z "${4}" ]]
		then
			case "${1}" in
			ina3221)
				if [ "$(< ${dir}/in1_label)" != "${4}" ]
				then continue
				fi
				;;
			*)
				echo "unsupported condition"
				;;
			esac
		fi

		# if power limit and bus voltage channel are provided
		# in $5 & $6, then calculate critical current limit
		# or else directly write $3 value into $2 node.
		if [[ ! -z "${5}" && ! -z "${6}" ]]; then
			power_limit="${5}";
			bus_volt="$(cat ${dir}/${6})"
			crit_limit="$((power_limit * 1000 / bus_volt))"
			echo "${crit_limit}" > "${dir}/${2}"
		else
			echo "${3}" > "${dir}/${2}"
		fi
	done
}

function setup_hwmon()
{
	case "${SOCFAMILY}" in
	tegra194 | tegra234)
		# Set averaging mode to use 512 samples
		config_hwmon ina3221 samples 512
		# Set update interval to 430ms
		# (2 * 140us convertion time * 3 channels * 512)
		config_hwmon ina3221 update_interval 430
		if [[ "${machine}" = "p3701-0000" || "${machine}" = "p3701-0005" ]]; then
			config_hwmon ina3221 curr4_crit 3380 VDD_GPU_SOC 150000 in1_input
		elif [ "${machine}" = "p3701-0004" ]; then
			config_hwmon ina3221 curr4_crit 3380 VDD_GPU_SOC 45000 in1_input
		elif [ "${machine}" = "p3767-0000" ] || [ "${machine}" = "p3767-0002" ]; then
			config_hwmon ina3221 curr1_max 1315 VDD_IN 30000 in1_input
			config_hwmon ina3221 curr1_crit 1578 VDD_IN 45000 in1_input
		elif [ "${machine}" = "p3767-0001" ]; then
			config_hwmon ina3221 curr1_max 1052 VDD_IN 20000 in1_input
			config_hwmon ina3221 curr1_crit 1315 VDD_IN 25000 in1_input
		elif [ "${machine}" = "p3767-0003" ] || [ "${machine}" = "p3767-0005" ]; then
			config_hwmon ina3221 curr1_max 3000 VDD_IN 15000 in1_input
			config_hwmon ina3221 curr1_crit 4000 VDD_IN 20000 in1_input
		elif [ "${machine}" = "p3767-0004" ]; then
			config_hwmon ina3221 curr1_max 2000 VDD_IN 10000 in1_input
			config_hwmon ina3221 curr1_crit 3000 VDD_IN 15000 in1_input
		fi
		;;
	*)
		# no need to set
		;;
	esac
}

SOCFAMILY=""

set_socfamily
set_power_state_perm
create_nvpmodel_symlink
create_nvfancontrol_symlink
cpu_hotplug
set_cpu_governor
lock_se_frequency
enable_vic_actmon
validate_nvpmodel_conf
setup_hwmon
