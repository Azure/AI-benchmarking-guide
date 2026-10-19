import json

from prettytable import PrettyTable

from Infra import tools


def run_fio():
    tools.progress("Running FIO Tests...")
    tests = [
        ("read", "1M"), ("read", "512k"), ("read", "1k"),
        ("write", "1M"), ("write", "512k"), ("write", "1k"),
        ("randwrite", "1k"), ("randread", "1k"),
    ]
    output_dir = tools.get_output_dir()
    table = PrettyTable(["Test", "Block Size", "Bandwidth (MiB/s)"])
    with tools.runtime.temporary_directory(prefix="fio-", dir=output_dir) as directory:
        for operation, block_size in tests:
            tools.progress(f"FIO: {operation}, block size {block_size}")
            result = tools.run_command(
                [
                    "fio", f"--bs={block_size}", "--ioengine=libaio",
                    "--iodepth=255", f"--directory={directory}", "--direct=1",
                    "--runtime=300", "--numjobs=4", f"--rw={operation}",
                    "--name=test", "--group_reporting", "--gtod_reduce=1",
                    "--size=10G", "--output-format=json",
                ],
                capture_output=True, text=True, check=True,
            )
            jobs = json.loads(result.stdout)["jobs"]
            if not jobs:
                raise tools.runtime.OutputError("FIO returned no job results")
            direction = "write" if "write" in operation else "read"
            bandwidth = 0
            for job in jobs:
                if job["error"] != 0:
                    raise tools.runtime.OutputError(f"FIO job failed: {job}")
                bandwidth += job[direction]["bw_bytes"]
            table.add_row([operation, block_size, bandwidth / (1024 ** 2)])
    tools.detail(table)
    tools.export_markdown("FIO Tests", "", table)
