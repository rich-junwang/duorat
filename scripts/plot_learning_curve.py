# @Vu Hoang & Thanh Vu @ Oracle

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
import numpy as np


def plot(x, y, xlabel, output_file, use_logscale=True):
    """
    Plotting the line chart
    """
    plt.plot(x, y, '-o')
    if use_logscale:
        plt.xlabel(xlabel + " - LogScale")
        plt.ylabel("Eval 1-ExactMatchSetAccuracy - LogScale")
        plt.yscale("log")
        plt.xscale("log")
    else:
        plt.xlabel(xlabel)
        plt.ylabel("Eval 1-ExactMatchSetAccuracy")
    plt.tight_layout()
    plt.savefig(f"{output_file}")
    plt.close()


def func(x, a, b, c):
    """
    Function for the curve_fit
    """
    return a + b * (x ** c)


def func_zero_a(x, b, c):
    """
    Function for the curve_fit when a is set to 0
    """
    return b * (x ** c)


def get_x_from_y(e, a, b, c):
    """
    Get total number of examples needed to get the error e
    """
    if e - a <= 0:
        return - 1
    n = ((e - a) / b) ** (1 / c)
    return n


def get_sigma(xdata, ydata):
    # sigma = 1 / (n / (e * (1 - e) ) )
    # where n is the number of training examples for that data point and e is the (gold) error rate for that data item.
    output = []
    for i in range(len(xdata)):
        output.append(1 / (xdata[i] / (ydata[i] * (1 - ydata[i]))))
    return output


def normalise(xdata, ydata, threshold):
    """
    To remove data points with y value is larger than the threshold
    """
    output = []
    for i in range(len(ydata)):
        if ydata[i] <= threshold:
            output.append(xdata[i])
    return output


def run_regression(xdata, ydata, output_file, xlabel, use_logscale=True, set_a_to_zero=True):
    sigma = get_sigma(xdata, ydata)
    a = 0
    if set_a_to_zero:
        popt, pcov = curve_fit(func_zero_a, xdata, ydata, sigma=sigma)
        plt.plot(xdata, ydata, 'o--', label='Few-shot learning performance')
        plt.plot(xdata, func_zero_a(xdata, *popt), 'g-',
                 label='Least-squares fit: b=%5.3f, c=%5.3f' % tuple(popt))
        b, c = tuple(popt)
    else:
        popt, pcov = curve_fit(func, xdata, ydata, sigma=sigma, method="trf", bounds=(-1, 1), maxfev=2000)
        plt.plot(xdata, ydata, 'o--', label='Few-shot learning performance')
        plt.plot(xdata, func(xdata, *popt), 'g-',
                 label='Least-squares fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        a, b, c = tuple(popt)
    if use_logscale:
        plt.xlabel(f"{xlabel} - LogScale")
        plt.ylabel('Eval 1-ExactMatchSetAccuracy - LogScale')
        plt.xscale('log')
        plt.yscale('log')
    else:
        plt.xlabel(f"{xlabel}")
        plt.ylabel('Eval 1-ExactMatchSetAccuracy')
        plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_file}")
    plt.close()
    return a, b, c


def process_csv_file(file_path, e_primes, threshold=1.0, set_a_to_zero=True):
    """
    Process a file
    """
    dataset = file_path.split("/")[-1].split(".")[0]
    with open(file_path) as reader:
        df = pd.read_csv(reader)
        fraction = df["Fraction"]
        examples = df["#Examples"]
        _1minuseasyacc = np.array(1) - np.array(df["Easy"])
        _1minusmediumacc = np.array(1) - np.array(df["Medium"])
        _1minushardacc = np.array(1) - np.array(df["Hard"])
        _1minusextrahardacc = np.array(1) - np.array(df["ExtraHard"])
        _1minusacc = df["1-ExactMatchSetAccuracy"]

        # Plot the logscale line charts
        plot(fraction, _1minusacc, "Training Fraction", f"{os.path.dirname(file_path)}/fraction.{dataset}.png")
        plot(examples, _1minusacc, "Training Examples (Total)",
             f"{os.path.dirname(file_path)}/examples.{dataset}.png")
        plot(examples, _1minuseasyacc, "Easy Level",
             f"{os.path.dirname(file_path)}/easyacc.{dataset}.png")
        plot(examples, _1minusmediumacc, "Medium Level",
             f"{os.path.dirname(file_path)}/mediumacc.{dataset}.png")
        plot(examples, _1minushardacc, "Hard Level",
             f"{os.path.dirname(file_path)}/hardacc.{dataset}.png")
        plot(examples, _1minusextrahardacc, "ExtraHard Level",
             f"{os.path.dirname(file_path)}/extrahardacc.{dataset}.png")

        def _get_curve_fit(examples, acc, e_prime, threshold, name="total"):
            # Get the performance on the full data (which is the last line in the input file)
            full_data_1minusf1 = acc[len(acc) - 1]
            full_data_num_examples = examples[len(acc) - 1]

            # Get the remaining data for curve_fit
            remaining_examples = list(examples[: len(examples) - 1])
            remaining_acc = list(acc[: len(acc) - 1])

            # Normalise data based on the threshold
            remaining_examples = normalise(remaining_examples, remaining_acc, threshold)
            remaining_acc = normalise(remaining_acc, remaining_acc, threshold)

            a, b, c = run_regression(remaining_examples, remaining_acc,
                                     f"{os.path.dirname(file_path)}/curve_fit.{name}.examples.{dataset}.0.{threshold}.{set_a_to_zero}.png",
                                     f"Training Examples {name}", False, set_a_to_zero)

            pred_e = func(full_data_num_examples, a, b, c)
            print(f"\n~~~~Training Examples ({name})~~~~")
            print(f"RSM of the relative residuals: {round(abs(pred_e / full_data_1minusf1 - 1), 4)}")
            print(f"Number {name} examples needed to get e'={e_prime}: {int(get_x_from_y(e_prime, a, b, c))}")

            run_regression(remaining_examples, remaining_acc,
                           f"{os.path.dirname(file_path)}/curve_fit.{name}.examples.logscale.{dataset}.0.{threshold}.{set_a_to_zero}.png",
                           f"Training Examples {name}", True, set_a_to_zero)

        _get_curve_fit(examples=examples, acc=_1minusacc, e_prime=e_primes[0], threshold=threshold, name="total")
        _get_curve_fit(examples=examples, acc=_1minuseasyacc, e_prime=e_primes[1], threshold=threshold, name="easy")
        _get_curve_fit(examples=examples, acc=_1minusmediumacc, e_prime=e_primes[2], threshold=threshold, name="medium")
        try:
            _get_curve_fit(examples=examples, acc=_1minushardacc, e_prime=e_primes[3], threshold=threshold, name="hard")
        except:
            print("Failed to get curve fit for hard examples. Setting new threshold 0.8...")
            _get_curve_fit(examples=examples, acc=_1minushardacc, e_prime=e_primes[3], threshold=0.8, name="hard")

        try:
            _get_curve_fit(examples=examples, acc=_1minusextrahardacc, e_prime=e_primes[4], threshold=threshold,
                           name="extrahard")
        except:
            print("Failed to get curve fit for extrahard examples. Setting new threshold 0.9...")
            _get_curve_fit(examples=examples, acc=_1minusextrahardacc, e_prime=e_primes[4], threshold=0.9,
                           name="extrahard")


if __name__ == '__main__':
    datasets = ["Spider@0.2@0.05@0.2@0.3@0.5", "Sparc@0.3@0.2@0.4@0.6@0.7", "CoSQL@0.45@0.25@0.5@0.7@0.8"]
    folder_path = "./logdir/learning_curve"
    for dataset in datasets:
        dataset_path = f"{folder_path}/{dataset.split('@')[0]}.csv"
        dataset_eprimes = [float(e) for e in dataset.split('@')[1:]]
        assert len(dataset_eprimes) == 5
        if not os.path.exists(dataset_path):
            continue

        print("~~~~~~~~~~~~~" * 2)
        print(dataset)
        print("\nProcessing: a + bn^c")
        process_csv_file(dataset_path, e_primes=dataset_eprimes, threshold=1.0, set_a_to_zero=False)

        print("\nProcessing: bn^c ~ a = 0")
        process_csv_file(dataset_path, e_primes=dataset_eprimes, threshold=1.0)

        print(f"\nProcessing: bn^c ~ a = 0 + threshold = {0.5 if 'CoSQL' not in dataset else 0.6}")
        process_csv_file(dataset_path, e_primes=dataset_eprimes, threshold=0.5 if 'CoSQL' not in dataset else 0.6)
