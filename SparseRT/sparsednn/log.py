class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


def warn(text):
    print(f"{bcolors.WARNING}{text}{bcolors.ENDC}")

def fail(text):
    print(f"{bcolors.FAIL}{text}{bcolors.ENDC}")

def info(text):
    print(f"{bcolors.HEADER}{text}{bcolors.ENDC}")

def done(text):
    print(f"{bcolors.OKBLUE}{text}{bcolors.ENDC}")
