import random
import argparse

parser = argparse.ArgumentParser(description="DGA Domain Generater")

default_name = "bigviktor,ccleaner,chinad,enviserv,mydoom,padcrypt,shiotob,suppobox,vidro,xshellghost"
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

nr = '1000'
for name in family_names:
    command = 'python3 ./DGA/' + name + '/dga.py '
    output_dir = './Datasets/Malicious/' + name + '/example.txt'
    match name:
        case 'chinad':
            command += '--time ' + str(random.randint(0, 1000000)) + ' --nr ' + nr + ' --len ' + str(random.randint(0, 20))
            command += ' >> ' + output_dir
            print(command)
        case _:
            command = ''
