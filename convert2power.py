import argparse
import voice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')

    arg = parser.parse_args()

    voice.convert2power(arg.input, arg.output)