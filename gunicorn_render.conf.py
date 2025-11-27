import multiprocessing

timeout = 600
workers = multiprocessing.cpu_count() * 2 + 1
bind = "0.0.0.0:10000"
accesslog = "-"
errorlog = "-"
capture_output = True
enable_stdio_inheritance = True
