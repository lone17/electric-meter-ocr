"""
evaluate.py

Run: python evaluate.py --output_file output.csv --target_file train.csv

Description: Compute the metrics
"""

import sys

# Two constant for metrics, which will be updated later
SCORE1_CONSTANT = 0.8       # Mean accuracy
SCORE2_CONSTANT = 100       # Mean error

def read_file(fname):
    results = {}
    with open(fname, "r") as f:

        header = f.readline()
        for line in f.readlines():
            w = line.strip().split(',')

            fname = w[0]
            number = w[1]

            results[fname] = number

    return results

# If wrong output, assume return value 0
def convert_to_int(s):
    if s.isdigit():
        return (int)(s)
    else:
        return 0


def process(output_file, target_file):
    target = read_file(target_file)
    output = read_file(output_file)

    error1 = 0.
    total_digit = 0.

    error2 = 0.

    for fname in target:
        if fname in output:
            target_s = target[fname]
            target_v = (int)(target[fname])

            # Additional check for non-number exception
            s = output[fname]
            v = convert_to_int(output[fname])

            total_digit += len(target_s)
            for i in range(len(target_s)):
                if (i >= len(s)):
                    error1 += 1
                else:
                    if target_s[i] != s[i]:
                        error1 += 1

            diff = abs(v - target_v)
            error2 += diff

        else:
            error2 += SCORE2_CONSTANT
            error1 += len(target[fname])
            total_digit += len(target[fname])

    score1 = 1.0*error1/total_digit
    score2 = error2/len(target)

    print('Accuracy metrics:')
    print('\tTotal: %i' % total_digit)
    print('\tError1: %i' % error1)
    score1 = 7.5*SCORE1_CONSTANT/score1
    print('\tScore1: %.2f' % (score1))

    print('Distance metrics:')
    score2 = 7.5*SCORE2_CONSTANT/score2
    print('\tError2: %.2f' % error2)
    print('\tScore2: %.2f' % (score2))


    total_score = score1 + score2
    print("Total = %.2f / 20" % total_score)


if __name__=="__main__":

    # Get input parameters
    output_file = sys.argv[1]
    target_file = sys.argv[2]

    process(output_file, target_file)




