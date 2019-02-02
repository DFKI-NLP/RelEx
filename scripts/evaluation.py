from relex.evaluation.semeval2010_task8_evaluation import evaluate


def main():
    print(
        evaluate(
            model_dir="/home/christoph/experiment_results/relex/basic_test/",
            test_file="../rexplained/data/semeval_2010_task_8/test.jsonl",
            eval_script_file="../lm-transformer-re/analysis/semeval/semeval2010_task8_scorer-v1.2.pl",
        )
    )


if __name__ == "__main__":
    main()
