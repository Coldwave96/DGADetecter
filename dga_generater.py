import os
import random
import datetime
import argparse
import subprocess

parser = argparse.ArgumentParser(description="DGA Domain Generater")

default_name = "bigviktor,ccleaner,chinad,enviserv,mydoom,padcrypt,shiotob,vidro,xshellghost"
parser.add_argument('-n', '--name', type=str, default=default_name, 
                    help="Family name of the DGA domain to be generated, use comma to seperate multiple names. (default: {})".format(default_name))

args = parser.parse_args()
family_names = args.name.split(',')

# bigviktor,ccleaner,vidro,xshellghost: time + nr
# chinad: time + nr + lenth_of_SLD
# enviserv: seed(string) + nr + TLDs
# mydoom: time + nr + seed(hex) + TLDs
# padcrypt: date(%Y-%m-%d) + version
# shiotob: domain + version
# suppobox: word_list(file) + time(%Y-%m-%d %H:%M:%S)
nr = 1000

# seconds since January 1, 1970 UTC
time_now = datetime.datetime.now()
time_command = "date +%s -d " + '\"' + str(time_now) + '\"'
seconds = subprocess.check_output(time_command, shell=True).decode('utf-8').strip()

# date
date = time_now.strftime("%Y-%m-%d")
date_time = time_now.strftime("%Y-%m-%d %H:%M:%S")

# seed
seed_str = 'papa_koli'
seed_hex = '0xfa8'

# doamin
initial_domain = 'panisdar.com'

# TLD
tlds_enviserv = 'com-net-org-info-biz-in'
tlds_mydoom = 'com-biz-us-net-org-ws-info-in'

for name in family_names:
    commands = []
    command = 'python3 DGA/' + name + '/dga.py '
    output_dir = 'Datasets/Malicious/' + name + '/example.csv'
    match name:
        case 'chinad':
            command += '--time ' + str(seconds) + ' --nr ' + str(nr) + ' --len ' + str(random.randint(10, 19))
            commands.append(command)
        case 'enviserv':
            command += '--seed ' + seed_str + ' --nr ' + str(nr) + ' --tlds ' + tlds_enviserv
            commands.append(command)
        case 'mydoom':
            command += '--time ' + str(seconds) + ' --nr ' + str(nr) + ' --seed ' + seed_hex + ' --tlds ' + tlds_mydoom
            commands.append(command)
        case 'padcrypt':
            command1 = command + '--date ' + date + ' --version ' + '2.2.86.1'
            command2 = command + '--date ' + date + ' --version ' + '2.2.97.0'
            command3 = command + '--date ' + date + ' --version ' + '11.37.0.0'
            commands.append(command1)
            commands.append(command2)
            commands.append(command3)
        case 'shiotob':
            command1 = command + '--domain ' + initial_domain + ' --version ' + str(1)
            command2 = command + '--domain ' + initial_domain + ' --version ' + str(2)
            commands.append(command1)
            commands.append(command2)
        case _:
            command += '--time ' + str(seconds) + ' --nr ' + str(nr)
            commands.append(command)
    print("[*]Generating DGA family {}".format(name))
    for index, command in enumerate(commands):
        if index > 0:
            command += ' >> ' + output_dir
        else:
            command += ' > ' + output_dir
        print("Command: {}".format(command))
        os.system(command)
        
