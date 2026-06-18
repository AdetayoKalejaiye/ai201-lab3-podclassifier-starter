Evaluation Spec — Pod Classifier

Complete this spec before writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for compute_accuracy() and
compute_per_class_accuracy() in evaluate.py.

Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:

Overall: What fraction of episodes did we classify correctly?

Per-class: Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

compute_accuracy(predictions, ground_truth)

What it does

Returns the fraction of predictions that exactly match the ground truth.

Inputs

Parameter

Type

Description

predictions

list[str]

Labels predicted by classify_episode(), one per episode.

ground_truth

list[str]

The correct labels, in the same order as predictions.

Output

Return value

Type

Description

accuracy

float

A value between 0.0 and 1.0.

Spec fields — fill these in before writing code

Formula:

Accuracy measures the total proportion of matches. It is calculated as:
Accuracy = (Number of Correctly Classified Episodes) / (Total Number of Episodes)

A prediction is defined as "correct" if predictions[i] == ground_truth[i].


Step-by-step logic:

1. Validate that the predictions and ground_truth lists have the exact same length.
2. Initialize a counter variable (e.g., `correct_count = 0`).
3. Iterate over both lists simultaneously using a loop or python's built-in `zip(predictions, ground_truth)`.
4. For every pair, compare the elements. If they are an exact match, increment `correct_count` by 1.
5. Compute the final float by dividing `correct_count` by `len(predictions)`.


Edge case — what if both lists are empty?

The function should return 0.0. When both lists are empty, there are zero total episodes to evaluate. To avoid a runtime division-by-zero crash, we explicitly return 0.0 and treat the missing evaluation set as 0% accuracy.


Worked example:

predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Evaluation by index:
- Index 0: Match ("interview" == "interview") -> Correct
- Index 1: Match ("solo" == "solo") -> Correct
- Index 2: Mismatch ("panel" != "solo") -> Incorrect
- Index 3: Mismatch ("interview" != "narrative") -> Incorrect

Total correct = 2
Total items = 4
Accuracy = 2 / 4 = 0.50 (50.0%)


compute_per_class_accuracy(predictions, ground_truth)

What it does

Returns accuracy broken down by each label. For each label in VALID_LABELS,
reports how many episodes with that ground-truth label were classified correctly.

Inputs

Parameter

Type

Description

predictions

list[str]

Labels predicted by classify_episode().

ground_truth

list[str]

Correct labels, in the same order.

Output

A dict keyed by label. Each value is a dict with three keys:

{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}


Spec fields — fill these in before writing code

What does "correct" mean for a given class?

"Correct" for a given class (e.g., "interview") means the number of times the ground-truth label was "interview" AND the model successfully predicted "interview".


What does "total" mean for a given class?

"Total" is the total number of episodes in the test set where the true (ground-truth) label is that class. It does not measure how many times the model guessed the class; it measures how many times the class actually occurred in the real data.


Step-by-step logic:

1. Initialize a nested dictionary with the standard format. For each label in VALID_LABELS, set "correct" to 0, "total" to 0, and "accuracy" to 0.0.
2. Iterate through `predictions` and `ground_truth` in parallel using `zip()`.
3. For each pair (pred, true):
   a. If the `true` label is a valid class tracked in our keys:
      i. Increment its "total" count by 1.
      ii. If `pred == true`, increment its "correct" count by 1.
4. Iterate through all labels in our dictionary:
   a. If "total" > 0, calculate "accuracy" = "correct" / "total".
   b. If "total" is 0, set "accuracy" to 0.0 to prevent division by zero.
5. Return the completed nested dictionary.


Edge case — what if a class has no examples in ground_truth (total == 0)?

If a class has 0 ground-truth examples, we set its accuracy to 0.0. This prevents a ZeroDivisionError when performing the divisions step, cleanly handling cases where a class is entirely absent from a sub-test or split.


Worked example:

predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Step-by-step trace:
- Ep 1: Pred "interview", Truth "interview" -> Correct "interview" (interview counts: corr=1, tot=1)
- Ep 2: Pred "interview", Truth "solo"      -> Incorrect "solo" (solo counts: corr=0, tot=1)
- Ep 3: Pred "solo",      Truth "solo"      -> Correct "solo" (solo counts: corr=1, tot=2)
- Ep 4: Pred "panel",     Truth "panel"     -> Correct "panel" (panel counts: corr=1, tot=1)
- Ep 5: Pred "panel",     Truth "narrative" -> Incorrect "narrative" (narrative counts: corr=0, tot=1)

label       correct  total  accuracy
----------  -------  -----  --------
interview      1       1      1.0
solo           1       2      0.5
panel          1       1      1.0
narrative      0       1      0.0


Reflection questions (discuss at the checkpoint)

Your overall accuracy might be decent even if one class has very low accuracy.
Why is per-class accuracy a more informative metric than overall accuracy alone?

If our test data has a class imbalance (e.g., 75% "interview" episodes and 25% other formats), overall accuracy would remain highly inflated even if the classifier completely failed at identifying non-interview episodes. Per-class accuracy strips away class-distribution bias, letting us diagnose exactly which formats are actually understood and which ones the model struggles to differentiate.

If panel episodes consistently get misclassified as interview, what does
that tell you about your training labels or your prompt?

This points to a weak taxonomy boundary or ambiguous training examples in our few-shot prompt. It suggests that the examples or the instructions provided do not adequately explain the difference between a multi-guest "panel" and a standard conversational "interview". To fix this, our prompt must explicitly emphasize the speaking dynamics: "panel" means multiple peer-level experts collaborating or debating, whereas "interview" is a primary host leading a structured dialogue with a focal guest.

You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
How might the evaluation results change if you had labeled 100 training episodes?
What if you had 200 test episodes?

100 Training Episodes: With more training episodes, we could provide the few-shot prompt with diverse and higher-quality examples representing clear-cut cases and nuanced edge cases, likely boosting precision on tricky boundaries (like solo vs. narrative).

200 Test Episodes: Evaluating on a larger, more diverse test set would significantly reduce statistical variance. A small 20-episode test set can easily suffer from "luck of the draw," where one weird description completely swings class accuracy by 20%. A larger test set establishes a much more robust, statistically significant baseline of the model's true real-world accuracy.