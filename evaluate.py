import argparse
from metrics.pope_calculator import PopeCalculator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA model on POPE benchmark")
    parser.add_argument("--ans_file", type=str, required=True, help="The file containing the predictions")
    parser.add_argument("--label_file", type=str, required=True, help="The file containing the answers")
    args = parser.parse_args()
    calc = PopeCalculator()
    calc.parse(args.ans_file, args.label_file)
    print(calc.calculate_results())