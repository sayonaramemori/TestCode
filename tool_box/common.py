import platform

def get_current_os():
    os_name = platform.system()
    if os_name == "Windows":
        print("Running on Windows")
        return 0
    elif os_name == "Linux":
        print("Running on Linux")
        return 1
    elif os_name == "Darwin":
        print("Running on macOS")
        return 2
    else:
        print("Unknown OS:", os_name)
    # return os_name

if __name__ == '__main__':
    get_current_os()
