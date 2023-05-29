# Per lanciare
# python run_experiment_qsub.py -f "/home/lorenzoserina/hello.py" -l /home/lorenzoserina/
import os
import getopt
import sys
if __name__ == '__main__':
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "p:a:l:f:n:", ['sh'])
    python_path = "/home/lorenzoserina/.conda/envs/Liberty/bin/python"#liberty
    file_path = '/home/lorenzoserina/script/'
    scripts_path = '/home/lorenzoserina/python_script'
    logger_dir = "/home/lorenzoserina/log/"
    script_name = 'script.sh'
    arguments = ''
    nodes = 4
    run_sh = False
    for opt, arg in opts:
        if opt == "-p":
            python_path = arg
        elif opt == '-a':
            arguments = arg
        elif opt == '-l':
            logger_dir = arg
        elif opt == '-f':
            file_path = arg
        elif opt == '-n':
            nodes = int(arg)
        elif opt == '--sh':
            print('OK')
            run_sh = True
    if os.path.isdir(scripts_path):
        os.system(f'rm -r {scripts_path}')
    os.system(f'mkdir -p {scripts_path}')
    
    with open(f'{scripts_path}{script_name}', 'w') as f:
        if run_sh:
            f.write(f'/bin/bash {file_path} {arguments} > {logger_dir}live_log.txt')
        else:
            f.write(f'CUDA_LAUNCH_BLOCKING=1 {python_path} {file_path} {arguments} > {logger_dir}live_log.txt')
        f.close()
    os.system(f'qsub -o {logger_dir}out.log -e {logger_dir}out.err.log -q longbatch -p 1023 -l walltime=168:00:00,nodes=minsky.ing.unibs.it:ppn={nodes}  {scripts_path}{script_name}') #verylong