'''
    DGA of XshellGhost
'''

import random
import argparse
from datetime import datetime
from ctypes import c_uint

def dga(year, month, nr, tlds):
    for _ in range(nr):
        year = random.randint(1970, year)
        month = random.randint(1, 12)

        _year = c_uint(year)
        _month = c_uint(month)
        seed = c_uint(0)

        seed.value = 0x90422a3a * _month.value
        seed.value -= 0x39d06f76 * _year.value
        seed.value -= 0x67b7fc6f
        
        sld_len = seed.value % 6 + 10
        sld = ''
        for i in range(sld_len):
            sld += chr(seed.value % 0x1a + ord('a'))
            seed.value = 29 * seed.value + 19

        domain = sld + '.' + tlds[0]
        print(domain)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', help="Seconds since January 1, 1970 UTC")
    parser.add_argument("-n", "--nr", help="nr of domains to generate")
    args = parser.parse_args()

    tlds = ['com']
        
    d = datetime.utcfromtimestamp(int(args.time))
    dga(d.year, d.month, int(args.nr), tlds)
