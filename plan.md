# Goal
We start with replication and see how easy that can be done. The hope is to finish this task until end of the day Thursday 20-03-2024. Before this, on Monday 17-03-2024, we thoroughly analyze the paper as preparation for Tuesday, when we start with replication.

If replication seems achievable Tuesday, we keep going till Thursday and try to finish it all. If not, we switch to the existing codebase(s) and keep going with the rest of our plan.

The rest of the plan is to do an **ablation study to compare performance of prototypical networks with varying embedding size** and **to reproduce the last row of table 2**.

In addition to this, we give ourselves the space to do further analyzes in what we find interesting, e.g. hyperparameter analysis, writing a new algorithm varaint, testing on new datasets, etc.

# Important Dates
- **Work on the code:** 19-3-2024 till 20-3-2024
- **Work on results:** 21-3-2024 till 4-4-2024 
- **Work on blog/presentation:** 4-4-2024 till 8-4-2024
- **finish it all before:** 8-4-2024
- **algo exam:** 10-4-2024
- **presentation & blog post:** 11-4-2024




# Important points from BS

For writing your reproducibility blog, draw inspiration from the following sources:

- Earlier projects are available here: https://reproducedpapers.org/ 
- ReScience journal: http://rescience.github.io/ 
- Checklist: https://ai.facebook.com/blog/how-the-ai-community-can-get-serious-about-reproducibility/ 
- Reproducibility workshop: https://sites.google.com/view/icml-reproducibility-workshop/home 
- Follow my writing guidelines: http://jvgemert.github.io/links.html 

If you like the project, consider posting your blog post online to inform deep learning researchers.

---

The project is pass/fail, and you pass if you score above 60% according to these criteria:

    • 20%: Motivation: Good plan prepared and discussed with TA, meets TA sufficiently, attends the presentation session

    • 50%: Content: Show good efforts on trying to reproduce something

    • 30%: Exposition: write a clear blog post; address the value of doing a reproduction; state if your reproduction results uphold the main conclusions of the paper.


### Reproducibility project guidelines
There are multiple criteria for reproducing a paper.The two main options are: 

- Full re-implementation: Implemented the paper contribution without using pre-existing code. This will pass the project without any extra work needed.

- Existing code: Replicate an existing code implementation. Additional criteria will be required. Please consult with your TA and see below.

When you submit your reproduction, you can select multiple criteria, each person should at least be responsible for 1 criteria:

- Replicated: A full implementation from scratch without using any pre-existing code.
- Reproduced: Existing code was evaluated.
- Hyperparams check: Evaluating sensitivity to hyperparameters.
- New data: Evaluating different datasets to obtain similar results.
- New algorithm variant: Evaluating a slightly different variant.
- New code variant: Rewrote or ported existing code to be more efficient/readable.
- Ablation study: Additional ablation studies.

In your submission make clear which criteria you used, and why. Each submission should briefly list what each person contributed (few lines). 