import time
import psutil as ps
import logging

logger = logging.getLogger("Resource_Manager")
logger.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

warning_handler = logging.FileHandler('../logs/runtime_checkup.log')
warning_handler.setLevel(logging.WARNING)
warning_handler.setFormatter(formatter)

logger.addHandler(warning_handler)

class Resource_Manager(object):
    def __init__(self, cpu_TH:float = 90, memory_TH:float = 80):
        self.cpu_TH = cpu_TH
        self.memory_TH = memory_TH

    def cpu_checkup(self) -> bool:
        current_cpu_usage = ps.cpu_percent()
        if current_cpu_usage >= self.cpu_TH:
            logger.exception("CPU usage is way too high {}%, Cool down".format(current_cpu_usage))
            time.sleep(2)
            return True

        elif (current_cpu_usage < self.cpu_TH) and (current_cpu_usage >= self.cpu_TH - 10.0):
            logger.error("CPU usage is going to increase, {}% to BROKE!".format(self.cpu_TH - current_cpu_usage))
            return False
        

    def ram_checkup(self) -> bool:
        current_ram_usage = ps.virtual_memory().percent
        if current_ram_usage >= self.memory_TH:
            logger.exception("RAM usage is way too high {}%, Cool down".format(current_ram_usage))
            time.sleep(2)
            return True

        elif (current_ram_usage < self.memory_TH) and (current_ram_usage >= self.memory_TH - 10.0):
            logger.error("RAM usage is going to increase, {}% to BROKE!".format(self.memory_TH - current_ram_usage))
            return False


    def monitor(self):
        return [self.cpu_checkup(), self.ram_checkup()]