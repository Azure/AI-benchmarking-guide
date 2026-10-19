from Infra.fio import run_fio


class FIO:
    def __init__(self, path: str, machine: str):
        self.name = "FIO"
        self.machine_name = machine

    def run(self):
        run_fio()
