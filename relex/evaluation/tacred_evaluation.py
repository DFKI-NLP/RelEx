from typing import Optional

import sys
from collections import Counter
from relex.predictors.predictor_utils import load_predictor
from relex.models.model_utils import batched_predict_instances


NO_RELATION = "no_relation"


def score(key, prediction, verbose=False):
    # TODO: state source

    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1:
                sys.stdout.write(" ")
            if prec < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1:
                sys.stdout.write(" ")
            if recall < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1:
                sys.stdout.write(" ")
            if f1 < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = (float(sum(correct_by_relation.values()))
                      / float(sum(guessed_by_relation.values())))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = (float(sum(correct_by_relation.values()))
                        / float(sum(gold_by_relation.values())))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro


def evaluate(model_dir: str,
             test_file: str,
             archive_filename: str = "model.tar.gz",
             cuda_device: int = -1,
             predictor_name: str = "relation_classifier",
             weights_file: Optional[str] = None,
             batch_size: int = 16) -> str:
    predictor = load_predictor(model_dir, predictor_name, cuda_device,
                               archive_filename, weights_file)

    test_instances = predictor._dataset_reader.read(test_file)  # pylint: disable=protected-access
    test_results = batched_predict_instances(predictor, test_instances, batch_size)

    true_labels = [instance["label"].label for instance in test_instances]
    predicted_labels = [result["label"] for result in test_results]

    return score(true_labels, predicted_labels)
