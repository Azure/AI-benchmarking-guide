import docker
import os
import shlex
import subprocess
import json
import pandas as pd
import csv
from prettytable import PrettyTable
import json

class LLMBenchmark:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "LLMBenchmark"
        self.config = self.get_config(config_path)
        self.hf_username = self.config['credentials']['hf_username']
        self.hf_password = self.config['credentials']['hf_password']
        self.dir_path = dir_path
        self.precision = "float16"
        self.container = None
        self.machine = machine

    def get_config(self, path: str):
        file = open(path)
        data = json.load(file)
        file.close()
        try:
            return data[self.name]
        except KeyError:
            raise KeyError("no value found")
    
    def create_container(self):
        client = docker.from_env()

        
        # Define the Docker run options
        docker_run_options = {
            'runtime': 'nvidia',
            'volumes': {str(self.dir_path): {'bind': str(self.dir_path), 'mode': 'rw'}},
            'entrypoint': '/bin/bash',
            'tty': True,
            'detach': True
        }

        #if existing container exists
        # self.container = client.containers.get('c34fc0616f7a')

        # Creates new Docker container
        self.container = client.containers.run('nvidia/cuda:12.1.0-devel-ubuntu22.04', **docker_run_options)

        print(f"Docker Container ID: {self.container.id}")

        # Install Necessary Libraries and Repositories
        self.install_requirements()

        # # Create required Folders
        if not os.path.exists(f'{self.dir_path}/models'):
            self.container.exec_run(f"mkdir {self.dir_path}/models")
        if not os.path.exists(f'{self.dir_path}/checkpoints'):
            self.container.exec_run(f"mkdir {self.dir_path}/checkpoints")
        
    
    def install_requirements(self):
        # Update package lists first
        i1 = self.container.exec_run("apt-get update", stderr=True)
        if i1.exit_code != 0:
            print(i1.output.decode('utf-8'))
        
        # Install required packages
        print("Installing Required Packages")
        i2 = self.container.exec_run("apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git", stderr=True)
        if i2.exit_code != 0:
            print(i2.output.decode('utf-8'))

        # Install tensorrt-llm package
        i3 = self.container.exec_run("pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com ")
        if i3.exit_code != 0:
            print(i3.output.decode('utf-8'))

        i4 = self.container.exec_run("pip3 install --upgrade transformers")
        if i4.exit_code != 0:
            print(i4.output.decode('utf-8'))
        print("Cloning TensorRT-LLM reopsitory from https://github.com/NVIDIA/TensorRT-LLM.git") # Add a tag
        
        # Clone TensorRT-LLM repo
        if not os.path.exists(os.path.join(self.dir_path, 'TensorRT-LLM')):
            i4 = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path} && git clone https://github.com/NVIDIA/TensorRT-LLM.git"' , stderr=True)
            if i4.exit_code != 0:
                print(i4.output.decode('utf-8'))
            else:
                print('TensorRT-LLM Cloned')

        else:
            print("TensorRT-LLM already exists")
        i6 = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/TensorRT-LLM && git checkout a681853d3803ee5893307e812530b5e7004bb6e1"')

            

    def download_models(self):
        # Install git-lfs
        dm1 = self.container.exec_run("apt-get install git-lfs")
        if dm1.exit_code != 0:
            print(dm1.output.decode('utf-8'))
            
        dm2 = self.container.exec_run("git lfs install")
        if dm2.exit_code != 0:
            print(dm2.output.decode('utf-8'))

        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model']:
                model_type = self.config['models'][model_name]['type']
                model_url = self.config['models'][model_name]['hf_url']

                # Install Model Requirements
                dm3 = self.container.exec_run(f'pip install -r {self.dir_path}/TensorRT-LLM/examples/{model_type}/requirements.txt ')
                if dm3.exit_code != 0:
                    print(dm3.output.decode('utf-8'))

                # Clone required models.
                print("Downloading", model_name)
                if not os.path.exists(os.path.join(self.dir_path, 'models', model_name)):
                    new_url = model_url.replace("https://", '')
                    self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/models && git clone https://{self.hf_username}:{self.hf_password}@{new_url}"')
                    print(model_name, "downloaded successfully")
                else: # Add Error handling
                    print(model_name, 'already exists')
    

    def build_engines(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model']:
                model_precision = self.config['models'][model_name]['precision']
                model_type = self.config['models'][model_name]['type']
                for tp_size in self.config['models'][model_name]['tp_sizes']:
                    # Convert Checkpoints

                    if not os.path.exists(f'{self.dir_path}/engines/{model_name}/tp_{tp_size}/{self.precision}/rank0.engine'):
                        print(f"Converting Checkpoints for {model_name}, TP size:{tp_size}") 
                        if tp_size == 1:
                            if model_precision == "fp8":
                                convert_checkpoints_command = f'''
                                python3 {self.dir_path}/TensorRT-LLM/examples/quantization/quantize.py \
                                    --model_dir {self.dir_path}/models/{model_name}\
                                    --qformat fp8 \
                                    --kv_cache_dtype fp8 \
                                    --output_dir {self.dir_path}/checkpoints/{model_name}/tp_{tp_size}/{self.precision}\
                                    --dtype {self.precision}
                                '''
                            else:
                                convert_checkpoints_command = f'''
                                python3 {self.dir_path}/TensorRT-LLM/examples/{model_type}/convert_checkpoint.py \
                                    --model_dir {self.dir_path}/models/{model_name}\
                                    --output_dir {self.dir_path}/checkpoints/{model_name}/tp_{tp_size}/{self.precision}\
                                    --dtype float16 \
                                '''

                        else:
                            if model_precision == "fp8":
                                convert_checkpoints_command = f'''
                                python3 {self.dir_path}/TensorRT-LLM/examples/quantization/quantize.py \
                                    --model_dir {self.dir_path}/models/{model_name}\
                                    --qformat fp8 \
                                    --kv_cache_dtype fp8 \
                                    --output_dir {self.dir_path}/checkpoints/{model_name}/tp_{tp_size}/{self.precision}\
                                    --dtype {self.precision}\
                                    --tp_size {tp_size}
                                '''
                            else:
                                convert_checkpoints_command = f'''
                                python3 {self.dir_path}/TensorRT-LLM/examples/{model_type}/convert_checkpoint.py \
                                    --model_dir {self.dir_path}/models/{model_name}\
                                    --output_dir {self.dir_path}/checkpoints/{model_name}/tp_{tp_size}/{self.precision}\
                                    --dtype {model_precision} \
                                    --load_by_shard \
                                    --workers 8 \
                                    --tp_size 8 \
                                    --pp_size 1
                                '''

                        be1 = self.container.exec_run(convert_checkpoints_command)
                        if be1.exit_code != 0:
                            print(be1.output.decode('utf-8'))

                        # Build Engines
                        print(f"Building Engine for {model_name} , TP size:{tp_size}")
                        build_engine_command = f'''
                            trtllm-build \
                            --checkpoint_dir {self.dir_path}/checkpoints/{model_name}/tp_{tp_size}/{self.precision}\
                            --output_dir {self.dir_path}/engines/{model_name}/tp_{tp_size}/{self.precision} \
                            --workers {tp_size} \
                            --gemm_plugin auto
                        '''
                        # build_engine_command = f'''
                        #     trtllm-build \
                        #     --checkpoint_dir {self.dir_path}/checkpoints/{model_name}/tp_{tp_size}/{self.precision}\
                        #     --output_dir {self.dir_path}/engines/{model_name}/tp_{tp_size}/{self.precision} \
                        #     --max_num_tokens 4096 \
                        #     --max_input_len 64000 \
                        #     --max_seq_len 65000 \
                        #     --use_paged_context_fmha enable \
                        #     --workers 8 
                        # '''

                        be2 = self.container.exec_run(build_engine_command)
                        if be2.exit_code != 0:
                            print(be2.output.decode('utf-8'))

        # Delete Converted Checkpoints
        be3 = self.container.exec_run(f'''rm -rf {self.dir_path}/checkpoints''')
        if be3.exit_code != 0:
                print(be3.output.decode('utf-8'))

    def run_benchmark(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model']:
                for tp_size in self.config['models'][model_name]['tp_sizes']:
                    for batch_size in self.config['models'][model_name]['batch_sizes']:
                        for input_output_size in  self.config['models'][model_name]['input_output_sizes']:
                            warmup = self.config['models'][model_name]['warmup']
                            number_of_runs = self.config['models'][model_name]['number_of_runs']
                            print(f"Benchmarking {model_name} | TP Size: {tp_size} | Batch Size: {batch_size} | Input Size: {input_output_size.split(',')[0]} | Output Size: {input_output_size.split(',')[1]}")

                            if tp_size == 1:
                                
                                print(model_name, tp_size, batch_size, input_output_size)
                                run_benchmark_command = f'''
                                    python3 {self.dir_path}/TensorRT-LLM/benchmarks/python/benchmark.py \
                                                --batch_size {batch_size} \
                                                --input_output_len {input_output_size} \
                                                --warm_up {warmup} \
                                                --num_runs {number_of_runs} \
                                                --engine_dir {self.dir_path}/engines/{model_name}/tp_{tp_size}/{self.precision} \
                                                -m dec
                                '''
                            else:
                                run_benchmark_command = f'''
                                    mpirun -n {tp_size} --allow-run-as-root python3 {self.dir_path}/TensorRT-LLM/benchmarks/python/benchmark.py \
                                                --batch_size {batch_size} \
                                                --dtype float16
                                                --input_output_len {input_output_size} \
                                                --warm_up {warmup} \
                                                --num_runs {number_of_runs} \
                                                --engine_dir {self.dir_path}/engines/{model_name}/tp_{tp_size}/{self.precision} \
                                                -m dec
                                '''

                            cmd = (
                                    f"nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,"
                                    + f"power.draw,enforced.power.limit,clocks.current.sm,"
                                    + f"clocks.current.memory --format=csv --id=0 -lms 100 -f {self.dir_path}/Outputs/{model_name}_power.csv"
                                )


                            cmd = shlex.split(cmd)

                            power_results = subprocess.Popen(
                                cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True,
                                cwd=self.dir_path
                            )
                        
                            # c14_e = self.container.exec_run(c14)
                            rb1 = self.container.exec_run(run_benchmark_command)
                            if rb1.exit_code != 0:
                                print(rb1.output.decode('utf-8'))
                            # print(rb1.output.decode('utf-8'))
                            
                            cmd = "pkill -9 nvidia-smi"
                            cmd = shlex.split(cmd)

                            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=self.dir_path)
                            power_results.communicate()
                            result = rb1.output.decode('utf-8')


                            df = self.parse_results(result, model_name)
                            table1 = PrettyTable()

                            table1.add_row(['Model Named', model_name])
                            table1.add_row(['Input/Output lengths', f"{df['input_length'][-1]}; {df['output_length'][-1]}"])
                            table1.add_row(['World Size (TP size)', df['world_size'][-1]])
                            table1.add_row(['Batch Size', df['batch_size'][-1]])
                            table1.add_row(['Throughput (tokens/sec)', df['tokens_per_sec'][-1]])
                            table1.add_row(['Latency (ms)', df['latency(ms)'][-1]])

                            print(table1.get_string(header=False))

                            table2 = PrettyTable()

                            table2.field_names = ['Metrics', 'Maximum', 'Average'] 
                            table2.add_row(['Memory Usage', df['max_memory_usage'][-1], df['avg_memory_usage'][-1]])
                            table2.add_row(['Power Draw', df['max_power_usage'][-1], df['avg_power_usage'][-1]])
                            table2.add_row(['Clock Speed', df['max_clock_speed'][-1], df['avg_clock_speed'][-1]])

                            print(table2)

                            # self.run_model_sizes(model_name, batch_size)

    
    def parse_results(self,output: str, model_name: str):
        n_readings = len(output.split('[BENCHMARK]'))
        readings = output.split('[BENCHMARK]')
        if not os.path.exists(os.path.join(self.dir_path, 'Outputs', f"LLMBenchmark_{self.machine}.csv")):
            df = dict()
        else:
            df = pd.read_csv(os.path.join(self.dir_path, 'Outputs', f"LLMBenchmark_{self.machine}.csv")).to_dict(orient='list')
        for i in range(n_readings):
            if 'model_name' in readings[i]:
                result = readings[i].split('[TensorRT-LLM]')[0]
                result = result.strip().replace('dec', model_name).split()
                # print(result)
                for i in range(0,len(result),2):
                    
                    if result[i] not in df:
                        df[result[i]] = []
                    # print(df[result[i]])
                    df[result[i]].append(result[i+1])
                if 'machine' not in df:
                    df['machine'] = []
                

                df['machine'].append(self.machine)
                telemetry = self.get_telemetry(model_name)
                for param in telemetry:
                    if param not in df:
                        df[param] = []
                    df[param].append(telemetry[param])
                # print(df)
                pd.DataFrame(df).to_csv(os.path.join(self.dir_path, 'Outputs', f"LLMBenchmark_{self.machine}.csv"), index = False)
                return df


    def get_telemetry(self, model_name):
        result = dict()
        df_power = pd.read_csv(f'{self.dir_path}/Outputs/{model_name}_power.csv' ,on_bad_lines='skip')
        df_power = df_power[df_power[' utilization.gpu [%]'] != ' 0 %'].reset_index()

        memory_used_vals = [0]
        for i in df_power[' memory.used [MiB]'].tolist():
            try:
                memory_used_vals.append(float(i.split()[0]))
            except:
                pass
        max_memory = max(memory_used_vals)
        avg_memory = sum(memory_used_vals)/len(memory_used_vals)

        memory_total_vals = [0]
        for i in df_power[' memory.total [MiB]'].tolist():
            try:
                memory_total_vals.append(float(i.split()[0]))
            except:
                pass
        total_memory = max(memory_total_vals)

        clocks_current_vals = [0]
        for i in df_power[' clocks.current.sm [MHz]'].tolist():
            try:
                clocks_current_vals.append(float(i.split()[0]))
            except:
                pass
        max_clocks = max(clocks_current_vals)
        avg_clocks = sum(clocks_current_vals)/ len(clocks_current_vals)

        power_current_vals = [0]
        for i in df_power[' power.draw [W]'].tolist():
            try:
                power_current_vals.append(float(i.split()[0]))
            except:
                pass
        max_power = max(power_current_vals)
        avg_power = sum(power_current_vals)/ len(power_current_vals)

        power_total_vals = [0]
        for i in df_power[' enforced.power.limit [W]'].tolist():
            try:
                power_total_vals.append(float(i.split()[0]))
            except:
                pass
        total_power = max(power_total_vals)

        result['max_memory_usage'] = round(max_memory, 1)
        result['avg_memory_usage'] = round(avg_memory, 1)
        result['total_memory_usage'] = total_memory

        result['max_clock_speed'] = round(max_clocks, 1)
        result['avg_clock_speed'] = round(avg_clocks, 1)

        result['max_power_usage'] = round(max_power, 1)
        result['avg_power_usage'] = round(avg_power, 1)
        result['enforces_power_limit'] = total_power

        return result


    def run_model_sizes(self, model_name, batch_size):

        # if not os.path.isfile(f'Outputs/{model_name}_GEMMCublasLt_Performance_' + self.machine + '_'  + '_'+self.precision+'.csv'):
            
            model_config_path = f'{self.dir_path}/models/{model_name}/config.json'
            with open(model_config_path, 'r') as file:
                model_config = json.load(file)
            hidden_dim = model_config['hidden_size']
            intermediate_dim = model_config['intermediate_size']

            kv_dim = (model_config['hidden_size'] * model_config['num_key_value_heads']) / model_config['num_attention_heads']
            m_dims = [batch_size, batch_size, batch_size, batch_size]
            n_dims = [hidden_dim, kv_dim, intermediate_dim, hidden_dim]
            k_dims = [hidden_dim,  hidden_dim, hidden_dim, intermediate_dim]
            b_dims = [1, 1, 1, 1]  
            os.chdir(f"{self.dir_path}/bin")
            
            buffer = []
            
            for i in range(len(m_dims)):
                # change i, b and w
                results = subprocess.run(
                    [
                        "./cublaslt_gemm",
                        "-m",
                        str(m_dims[i]),
                        "-n",
                        str(n_dims[i]),
                        "-k",
                        str(k_dims[i]),
                        "-b",
                        str(b_dims[i]),
                        "-i",
                        str(10),
                        "-w",
                        str(10),
                        "-t",
                        str('fp8e4m3'),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                log = results.stdout.decode('utf-8').split()
                buffer.append(log)
            operation_name = [["Attention: Query Projection"]+buffer[0], ["Attention: Key Projection"]+buffer[1], ["Attention: Value Projection"]+buffer[1], ["Feed Forward: Up projection 1"]+buffer[2], ["Feed Forward: Up projection 2"]+buffer[2], ["Feed Forward: Down Projection"]+buffer[3]]
            

            with open(f'{self.dir_path}/Outputs/{model_name}_GEMMCublasLt_Performance_' + self.machine + '_'  + '_'+self.precision+'.csv', 'a') as csvFile:
                writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Operation", "M", "N", "K", "Batch", "Time(us)", "TFLOPS"])
                for item in operation_name:
                    writer.writerow(item)
            
            table = PrettyTable()

            table.field_names = ["Operation", "M", "N", "K", "Batch", "Time(us)", "TFLOPS"]
            for row in operation_name:
                table.add_row(row)
            print(table)
