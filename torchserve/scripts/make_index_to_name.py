# Python imports
import sys

if len(sys.argv) < 2:
    sys.exit("No input filename is given.")
input_filename = sys.argv[1]

if len(sys.argv) < 3:
    sys.exit("No output filename is given")
output_filename = sys.argv[2]

input_file = open(input_filename)
tags = [line for line in input_file]
input_file.close()

output_filename = open(output_filename, 'w')
output_filename.write('{')
index = 0
for tag in tags:
    output_filename.write(f'"{index}":"{tag[:-1]}"')
    if index < len(tags)-1:
        output_filename.write(',')
    index += 1
output_filename.write('}')
output_filename.close()
