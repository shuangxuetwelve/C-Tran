# Python imports
import sys

# Third-party imports
import torch

if len(sys.argv) < 2:
    sys.exit("No input filename is given.")
input_filename = sys.argv[1]

if len(sys.argv) < 3:
    sys.exit("No output filename is given")
output_filename = sys.argv[2]

checkpoint = torch.load(input_filename)
state_dict = checkpoint['state_dict']
torch.save(state_dict, output_filename)
