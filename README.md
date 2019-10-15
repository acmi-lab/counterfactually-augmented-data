# Learning the Difference that Makes a Difference with Counterfactually-Augmented Data

### Overview
This repository houses the dataset described in the paper *Learning the Difference that Makes a Difference with Counterfactually-Augmented Data*. Given documents and their initial labels, we tasked humans to (i) revise each document to accord with a counterfactual target label, subject to producing revisions that (ii) result in internally consistent documents and (iii) avoid any gratuitous changes to facts that are semantically unrelated to the applicability of the label.

### Sentiment Analysis

Fields        | Description
------------- | -------------
Text          | A movie review
Sentiment     | Positive or Negative Sentiment associated with the movie review

### Natural Language Inference

Fields        | Description
------------- | -------------
sentence1     | The premise sentence
sentence2     | The hypothesis sentence
gold_label    | The truth value of hypothesis given premise

## Bibligoraphy

If you use our data, please cite the paper that introduced the resource with the following BibTeX:

```
@article{kaushik2019learning,
  title={Learning the Difference that Makes a Difference with Counterfactually-Augmented Data},
  author={Kaushik, Divyansh and Hovy, Eduard and Lipton, Zachary C},
  journal={arXiv preprint arXiv:1909.12434},
  year={2019}
}
```

### Revision platform

We will release the code for the revision platform soon. We are currently cleaning up the codebase to make it easier to use. In the meanwhile, the interface is depicted below:
![Revision platform](https://github.com/dkaushik96/bizarro-data-private/blob/master/platform_screenshot.png)
