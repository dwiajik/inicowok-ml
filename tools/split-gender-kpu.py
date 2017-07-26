import argparse
import csv
from os.path import join, dirname

parser = argparse.ArgumentParser(description='Split datasets/data-pemilih-kpu.csv based on gender into two files.')
parser.add_argument('-i', '--input', default='datasets/data-pemilih-kpu.csv', help='File path input, e.g. datasets/data-pemilih-kpu.csv')
parser.add_argument('-m', '--male', default='male.txt', help='File name for output CSV containing male names, e.g. male.csv')
parser.add_argument('-f', '--female', default='female.txt', help='File name for output CSV containing female names, e.g. female.csv')
args = parser.parse_args()

print(args)

with open(join(dirname(__file__), '..', args.input), newline='\n') as csv_input:
    dataset = csv.reader(csv_input, delimiter=',', quotechar='"')
    both = [(line[0], line[1]) for line in dataset]
    males = [line[0] for line in both if line[1] == 'Laki-Laki']
    females = [line[0] for line in both if line[1] == 'Perempuan']

print('{} male names'.format(len(males)))
print('{} female names'.format(len(females)))

with open(join(dirname(__file__), '..', args.male), 'a', newline='\n') as output:
    for male in males:
        output.write('{}\n'.format(male))

with open(join(dirname(__file__), '..', args.female), 'a', newline='\n') as output:
    for female in females:
        output.write('{}\n'.format(female))

print('Done, output: {}, {}'.format(args.male, args.female))