import os
import datetime
import subprocess
import json
pwd = os.getcwd() + "/Outputs/log.txt"
curr = os.getcwd()

def create_dir(name: str):
    current = os.getcwd()
    outdir = os.path.join(str(current), name)
    isdir = os.path.isdir(outdir)
    if not isdir:
        os.mkdir(outdir)
    return outdir

def write_log(message: str, filename: str = pwd):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}]\n {message}\n"

    with open(filename, "a") as file:
        file.write(log_entry)

def check_error(results):
    if results.stderr:
        return results.stderr.decode("utf-8")
    return results.stdout.decode("utf-8")

def get_os_version():
    results = subprocess.run("lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    ubuntu = results.stdout.decode('utf-8').strip().split("\t")[1]
    return "Ubuntu"+ubuntu

def get_hostname():
    results = subprocess.run(["hostname"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if results.stderr:
        return ""
    return results.stdout.decode("utf-8").strip()

def prettytable_to_markdown(table):
    if table is None:
        return ""
    header = "| " + " | ".join(table.field_names) + " |"
    separator = "| " + " | ".join("---" for _ in table.field_names) + " |"
    rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table.rows]
    return "\n".join([header, separator] + rows) 

def export_markdown(title, description, table = None):
    table = prettytable_to_markdown(table)
    filename = curr + "/Outputs/" + get_hostname() + "_summary.md"
    with open(filename, "a") as file:
        if title is not None:
            file.write("## " + title + "\n\n")
        file.write(description + "\n")
        file.write(table)
        file.write("\n\n")
        
def create_bm_entry(bmName, appName, sku, result):
    if os.getenv("BM_UPLOAD") == "1":    
        id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        ubuntu = get_os_version()
        return {
            "jobId": id,
            "appName": appName,
            "bmName": bmName,
            "nodes": "1",
            "cores": "",
            "sockets": "",
            "result": result,
            "totalRunTime": "",
            "skuGen": "",
            "sku": sku,
            "os": ubuntu,
            "BIOS": "",
            "user": "azureuser",
            "runCategory": "best",
            "Run Date": "",
            "notes": "",
            "appVersion": "string"
        }
    return None   

def post_benchmark_entry(entry):
    if os.getenv("BM_UPLOAD") == "1":
        url = ""
        json_data = json.dumps(entry)
        curl_command = [
            "curl",
            "-X", "POST",
            url,
            "-H", "Content-Type: application/json",
            "-d", json_data
        ]
        
        result = subprocess.run(curl_command, capture_output=True, text=True)
        return result.stdout, result.stderr
