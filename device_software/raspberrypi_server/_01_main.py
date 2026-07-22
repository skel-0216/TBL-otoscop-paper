import _02_funcs as endoscope
import sys

json_path = "_03_config.json"

if __name__ == '__main__':
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

while True:
    state = endoscope.main(json_path)

    if state == 1:
        print("\n************ Restart\n")

    if state == 2:
        print("\n************ please restart the code\n")
        break

