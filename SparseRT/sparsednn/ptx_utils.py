import random
import struct
from utils import half_to_hex


def parse_ptx(filename, A_blocks):
    ptx = open(filename, "r").readlines()
    reg_names = [[]for i in range(A_blocks)]
    global_address_names = [0]*A_blocks
    line = 0
    while line < len(ptx):
        if 'START' in ptx[line]:
            block = int(ptx[line].split('B')[1].split('G')[0])
            block_reg_list = reg_names[block]
            line += 1
            while 'ld.f32' not in ptx[line]:
                line += 1
            global_address_names[block] = ptx[line].split("[")[1].split("]")[0]
            line += 1
            while 'END' not in ptx[line]:
                if 'fma.rn.f32' in ptx[line]:
                    reg_name = ptx[line].split()[1][2:-1]
                    block_reg_list.append(int(reg_name))
                line += 1
        line += 1
    # print('reg_names')
    # print(reg_names)
    return reg_names, global_address_names


HALF_BIAS_RELU_BB = """
    mov.b32 bias_reg, BIAS;
    add.f16x2 temp_reg, SOURCE, bias_reg;
    set.gt.ftz.f16x2.f16x2 pred_reg, temp_reg, zero_reg;
    mul.rn.f16x2 DEST, pred_reg, temp_reg;
"""

HALF_BIAS_BB = """
    mov.b32 bias_reg, BIAS;
    add.f16x2 DEST, SOURCE, bias_reg;
"""


def hex_to_float(hex):
    return struct.unpack('!f', bytes.fromhex(hex))[0]


def insert_ptx(in_ptx_file, out_ptx_file, block_ptxs, relu=True, blurb=None, id=None):
    ptx_code = open(in_ptx_file, "r").readlines()
    new_file = open(out_ptx_file, "w")
    i = 0
    mads = 0
    reg_defined = False
    while i < len(ptx_code):
        line = ptx_code[i]
        #print("change ptx utils line 64 for your problems")
        if blurb and "mad.lo.s32" in line and " " + str(id) + "," in line:
            if mads == 0:
                stuff = line.replace("\n", "").split(",")
                x_reg = stuff[1]
                y_reg = stuff[3].replace(";", "")
                new_file.write(line)
                new_file.write(blurb.replace(
                    "X_REG", x_reg).replace("Y_REG", y_reg))
            else:
                new_file.write(line)
            mads += 1
        elif "START" in line:
            if not reg_defined:
                reg_defined = True
                new_file.write('\t.reg .f32 load_reg;\n\t')
            my_block = int(line.split("B")[1].split("G")[0])
            my_group = int(line.split("G")[1].split(";")[0])
            my_ptx = block_ptxs[my_block][my_group]

            # we have to deal with cases where the load instruction doesn't
            # immediately follow the inline assembly marker
            # we can't miss those instructions in between!
            new_file.write(line)

            for j in range(i+2, len(ptx_code)):
                if "ld.f32" in ptx_code[j]:
                    break
                else:
                    new_file.write(ptx_code[j])

            for ptx_line in my_ptx:
                new_file.write(ptx_line)

            new_file.write("\n")
            j = i
            next_i = None
            while True:
                if "END" in ptx_code[j]:
                    next_i = j + 1
                    break
                j += 1

            i = next_i

        # TODO: this is really hacky, you should not be using this.
        # we are parsing the ptx, finding the part where we add bias and perform relu
        # and manually swapping out that block for a handcoded ptx block that does the equivalent.
        # we literally parse the hex literal bias value, convert it to float, then to float16, then to hex again
        # we should change this approach.
        elif "add.f32" in line and "mov.f32" in ptx_code[i+1] and "max.f32" in ptx_code[i+2]:
            SOURCE = line.split(",")[-2]
            DEST = ptx_code[i+2].split(",")[-3].split()[-1]
            BIAS = half_to_hex(hex_to_float(line.split(
                ",")[-1].split(";")[0].replace("0f", "").replace(" ", "")))
            if relu:
                new_file.write(HALF_BIAS_RELU_BB.replace(
                    "SOURCE", SOURCE).replace("DEST", DEST).replace("BIAS", BIAS))
            else:
                new_file.write(HALF_BIAS_BB.replace("SOURCE", SOURCE).replace(
                    "DEST", DEST).replace("BIAS", BIAS))
            i += 2
        elif "add.f32" in line and "max.f32" in ptx_code[i+1]:
            SOURCE = line.split(",")[-2]
            DEST = ptx_code[i+1].split(",")[-3].split()[-1]
            BIAS = half_to_hex(hex_to_float(line.split(
                ",")[-1].split(";")[0].replace("0f", "").replace(" ", "")))
            if relu:
                new_file.write(HALF_BIAS_RELU_BB.replace(
                    "SOURCE", SOURCE).replace("DEST", DEST).replace("BIAS", BIAS))
            else:
                new_file.write(HALF_BIAS_BB.replace("SOURCE", SOURCE).replace(
                    "DEST", DEST).replace("BIAS", BIAS))
            i += 1
        else:
            new_file.write(line)

        i += 1
